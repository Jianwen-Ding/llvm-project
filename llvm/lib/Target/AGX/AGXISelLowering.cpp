//===-- AGXISelLowering.cpp - AGX DAG lowering implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering of LLVM IR to the AGX selection DAG.
//
// Legal types are i16, i32, i64, f16 and f32. f64 values are softened to i64
// (their arithmetic is integer code from AGXSoftF64), and there are no
// vector types: vectors are scalarized. Booleans are 0/1 in 32-bit registers.
// 64-bit integers are native for add, sub, multiply and bitwise operations;
// shifts, compares and bit counts are split into 32-bit halves here.
//
// Kernel arguments arrive in uniform registers, see LowerFormalArguments and
// llvm/docs/AGXUsage.rst. Calls are normally inlined. A narrow, opt-in leaf
// experiment exercises direct call/return code generation.
//
//===----------------------------------------------------------------------===//

#include "AGXISelLowering.h"
#include "AGXTargetTransformInfo.h"
#include "AGX.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXRegisterInfo.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>

using namespace llvm;

static cl::opt<bool> ExperimentalLeafCalls(
    "agx-experimental-leaf-calls", cl::Hidden, cl::init(false),
    cl::desc("Emit direct calls to argument-free void AGX leaf functions"));

bool llvm::agxExperimentalLeafCallsEnabled() { return ExperimentalLeafCalls; }

static cl::opt<bool> AGXPairedHalfStores(
    "agx-paired-half-stores", cl::Hidden, cl::init(true),
    cl::desc("Store 4-byte aligned pairs of 16-bit elements as one 32-bit "
             "word each"));

#define DEBUG_TYPE "agx-lower"

static void diagnose(SelectionDAG &DAG, const SDLoc &DL, const Twine &Msg) {
  const Function &F = DAG.getMachineFunction().getFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(F, Msg, DL.getDebugLoc()));
}

static cl::opt<bool> AGXWordShifts128(
    "agx-word-shifts-128", cl::init(true), cl::Hidden,
    cl::desc("Variable 128-bit shifts on the 32-bit words"));

AGXTargetLowering::AGXTargetLowering(const TargetMachine &TM,
                                     const AGXSubtarget &ST)
    : TargetLowering(TM, ST), ST(ST) {
  addRegisterClass(MVT::i16, &AGX::GPR16RegClass);
  addRegisterClass(MVT::f16, &AGX::GPR16RegClass);
  addRegisterClass(MVT::i32, &AGX::GPR32RegClass);
  addRegisterClass(MVT::f32, &AGX::GPR32RegClass);
  addRegisterClass(MVT::i64, &AGX::GPR64RegClass);
  computeRegisterProperties(ST.getRegisterInfo());

  setBooleanContents(ZeroOrOneBooleanContent);
  setSchedulingPreference(Sched::RegPressure);
  setMinFunctionAlignment(Align(2));
  setPrefFunctionAlignment(Align(2));
  setJumpIsExpensive(true);
  // Wider atomics are expanded with locks by AGXExpandWideAtomics, except
  // the 64-bit operations the chip has (shouldExpandAtomicRMWInIR).
  setMaxAtomicSizeInBitsSupported(ST.hasDeviceAtomicU64MinMax() ? 64 : 32);
  setMinCmpXchgSizeInBits(32);

  // Scalar branches become structured mask operations in
  // AGXAnnotateControlFlow.
  setOperationAction(ISD::BR_CC, {MVT::i16, MVT::i32, MVT::i64, MVT::f16,
                                  MVT::f32}, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  // llvm.agx.texture.sample's <4 x i32> (ReplaceNodeResults)
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v4i32, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction({ISD::STACKSAVE, ISD::STACKRESTORE,
                      ISD::DYNAMIC_STACKALLOC},
                     {MVT::i32, MVT::Other}, Expand);
  setOperationAction(ISD::GlobalAddress, {MVT::i32, MVT::i64}, Custom);
  // trap is an instruction; there is no debugger to stop in.
  setOperationAction({ISD::TRAP, ISD::DEBUGTRAP}, MVT::Other, Legal);
  setOperationAction({ISD::INTRINSIC_WO_CHAIN, ISD::INTRINSIC_VOID},
                     MVT::Other, Custom);
  // Fences become the barrier sequences measured for their scope and ordering.
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i32, Custom);
  // llvm.agx.simd.ballot takes an i1, which is promoted by custom lowering.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i1, Custom);
  setTargetDAGCombine({ISD::SHL, ISD::FP_TO_SINT, ISD::FP_TO_UINT, ISD::FP_TO_SINT_SAT,
                       ISD::FP_TO_UINT_SAT, ISD::SETCC, ISD::SELECT, ISD::BRCOND, ISD::TRUNCATE, ISD::BUILD_PAIR, ISD::SRL,
                       ISD::ZERO_EXTEND, ISD::FMINIMUMNUM, ISD::SRA,
                       ISD::SIGN_EXTEND_INREG, ISD::OR, ISD::AND, ISD::MULHU,
                       ISD::MULHS, ISD::ADD, ISD::FREEZE});
  // Vectors are not legal types; 4-byte aligned loads and stores of 2 to 4
  // 32-bit elements still become one access (lowerVectorLoad/Store).
  for (MVT VT : {MVT::v2i32, MVT::v3i32, MVT::v4i32, MVT::v2f32, MVT::v3f32,
                 MVT::v4f32})
    setOperationAction({ISD::LOAD, ISD::STORE}, VT, Custom);
  // and naturally aligned device loads of 2 or 4 bytes become one access of
  // the byte format (ReplaceNodeResults)
  setOperationAction({ISD::LOAD, ISD::STORE}, {MVT::v2i8, MVT::v4i8}, Custom);
  // 4-byte aligned loads of 2 or 4 16-bit elements: one 32-bit access per
  // pair, whose halves are the elements (ReplaceNodeResults)
  setOperationAction(ISD::LOAD, {MVT::v2i16, MVT::v4i16, MVT::v2f16, MVT::v4f16},
                     Custom);
  // and stores: one 32-bit store per pair, its halves the elements
  // (lowerVectorStore), not a 16-bit store each
  if (AGXPairedHalfStores)
    setOperationAction(ISD::STORE,
                       {MVT::v2i16, MVT::v4i16, MVT::v2f16, MVT::v4f16}, Custom);

  //===--------------------------------------------------------------------===//
  // Integer operations
  //===--------------------------------------------------------------------===//

  for (MVT VT : {MVT::i16, MVT::i32, MVT::i64}) {
    setOperationAction({ISD::SDIV, ISD::UDIV, ISD::SREM, ISD::UREM,
                        ISD::SDIVREM, ISD::UDIVREM, ISD::SMUL_LOHI,
                        ISD::UMUL_LOHI, ISD::BSWAP, ISD::SELECT_CC,
                        ISD::SHL_PARTS, ISD::SRL_PARTS, ISD::SRA_PARTS},
                       VT, Expand);
    setOperationAction({ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX}, VT,
                       VT == MVT::i64 ? Expand : Legal);
  }
  setOperationAction({ISD::MULHU, ISD::MULHS}, MVT::i32, Legal);
  // Variable 128-bit shifts on the 32-bit words (ReplaceNodeResults).
  if (AGXWordShifts128)
    setOperationAction({ISD::SHL, ISD::SRL, ISD::SRA}, MVT::i128, Custom);
  // Both halves of a 32 x 32 -> 64-bit product from one multiply (the
  // expansions of 64-bit multiplications would otherwise take the low half
  // from a second, 32-bit one).
  setOperationAction({ISD::UMUL_LOHI, ISD::SMUL_LOHI}, MVT::i32, Custom);
  setOperationAction({ISD::MULHU, ISD::MULHS}, MVT::i16, Expand);
  // four 32 x 32 -> 64-bit multiply-adds; the signed high half from the
  // unsigned one
  setOperationAction({ISD::MULHU, ISD::MULHS}, MVT::i64, Custom);
  setOperationAction({ISD::ROTL, ISD::ROTR, ISD::FSHL}, {MVT::i16, MVT::i64},
                     Expand);
  // fshl / rotl: extr by 32 - (s mod 32) (an amount of 32 gives the high
  // word, as s = 0 needs)
  setOperationAction({ISD::ROTL, ISD::FSHL}, MVT::i32, Legal);
  setOperationAction(ISD::FSHR, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i64, Custom);
  setOperationAction(ISD::BITREVERSE, MVT::i16, Promote);
  // (ctlz_zero_poison too: expanded it takes 10 instructions of shifts, ors
  // and a popcount, promoted 4. cttz_zero_poison's expansion, a popcount of
  // the low zeros, is as short as the promotion and uses the shared unit
  // once instead of twice.)
  setOperationAction({ISD::CTPOP, ISD::CTLZ, ISD::CTTZ, ISD::CTLZ_ZERO_POISON},
                     MVT::i16, Promote);
  setOperationAction({ISD::CTLZ, ISD::CTTZ}, MVT::i64, Custom);
  // Averages without the widened addition (a 64-bit add on the wide issue
  // slot): (a & b) + ((a ^ b) >> 1), and (a | b) - ((a ^ b) >> 1) rounding
  // up.
  setOperationAction({ISD::AVGFLOORU, ISD::AVGCEILU, ISD::AVGFLOORS,
                      ISD::AVGCEILS},
                     MVT::i32, Custom);
  setOperationAction({ISD::SHL, ISD::SRL, ISD::SRA, ISD::MUL}, MVT::i64,
                     Custom);
  setOperationAction(ISD::SETCC, MVT::i64, Custom);
  // 64-bit logic and selects are done on the halves (BUILD_PAIR joins them),
  // so that constant or known halves fold.
  setOperationAction({ISD::AND, ISD::OR, ISD::XOR, ISD::SELECT}, MVT::i64,
                     Custom);
  setOperationAction(ISD::BUILD_PAIR, MVT::i64, Legal);
  setOperationAction({ISD::UADDSAT, ISD::USUBSAT, ISD::SADDSAT,
                      ISD::SSUBSAT},
                     {MVT::i16, MVT::i32}, Legal);

  //===--------------------------------------------------------------------===//
  // Floating point operations
  //===--------------------------------------------------------------------===//

  for (MVT VT : {MVT::f16, MVT::f32}) {
    setOperationAction({ISD::FFLOOR, ISD::FCEIL, ISD::FTRUNC, ISD::FRINT,
                        ISD::FNEARBYINT, ISD::FROUNDEVEN, ISD::FMINNUM,
                        ISD::FMAXNUM, ISD::FEXP2, ISD::FLOG2},
                       VT, Legal);
    setOperationAction(ISD::FCOPYSIGN, VT, Custom);
    setOperationAction({ISD::FMINIMUMNUM, ISD::FMAXIMUMNUM, ISD::FMINIMUM,
                        ISD::FMAXIMUM},
                       VT, Custom);
    setOperationAction({ISD::FSUB, ISD::SELECT_CC, ISD::FMINNUM_IEEE,
                        ISD::FMAXNUM_IEEE, ISD::FSINCOS},
                       VT, Expand);
    setOperationAction(ISD::FROUND, VT, Custom);
    // Conditions without a direct hardware condition.
    setCondCodeAction({ISD::SETONE, ISD::SETUEQ, ISD::SETO, ISD::SETUO}, VT,
                      Expand);
  }
  setOperationAction({ISD::FDIV, ISD::FREM, ISD::FSIN, ISD::FCOS, ISD::FEXP,
                      ISD::FEXP10, ISD::FLOG, ISD::FLOG10, ISD::FPOW},
                     MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f16, Custom);
  setOperationAction(ISD::FSQRT, {MVT::f32, MVT::f16}, Custom);
  setOperationAction({ISD::FREM, ISD::FEXP, ISD::FEXP10, ISD::FLOG,
                      ISD::FLOG10, ISD::FPOW},
                     MVT::f16, Promote);
  setOperationAction({ISD::FSIN, ISD::FCOS}, MVT::f16, Custom);
  // ldexp of half in float: exact there (half values scaled within float's
  // range, and what underflows float is far below half's denormals), so the
  // rounding to half is the only one. The generic expansion in half scales
  // by at most 2^-6 before clamping n to the exponent range: ldexp(91.6, -37)
  // came out 8.7e-5, not 0.
  setOperationAction(ISD::FLDEXP, MVT::f16, Promote);
  AddPromotedToType(ISD::FLDEXP, MVT::f16, MVT::f32);
  setOperationAction({ISD::FP16_TO_FP, ISD::FP_TO_FP16}, MVT::f32, Expand);
  // bfloat16 is not a legal type: conversions are integer operations.
  setOperationAction({ISD::BF16_TO_FP, ISD::FP_TO_BF16},
                     {MVT::i16, MVT::f16, MVT::f32}, Expand);
  // The generic expansion truncates; round to nearest even instead.
  setOperationAction(ISD::FP_TO_BF16, {MVT::i16, MVT::f16, MVT::f32}, Custom);
  // Constrained bfloat16 arithmetic (promoted to f32) narrows with the strict
  // form. Directed rounding is done in software, so the dynamic rounding
  // mode is round to nearest even and it is lowered the same way.
  setOperationAction(ISD::STRICT_FP_TO_BF16, {MVT::i16, MVT::f16, MVT::f32},
                     Custom);

  // Float-to-integer convert saturates and gives 0 for NaN (measured on G14C,
  // AGX Performance Simulator tasks/ptx-re-blockers.md, B27), so saturating
  // conversions to the convert's own width are legal; the others expand.
  setOperationAction({ISD::FP_TO_SINT_SAT, ISD::FP_TO_UINT_SAT},
                     {MVT::i16, MVT::i32}, Custom);

  // 64-bit conversions are expanded with 32-bit conversions.
  setOperationAction({ISD::FP_TO_SINT, ISD::FP_TO_UINT, ISD::SINT_TO_FP,
                      ISD::UINT_TO_FP},
                     MVT::i64, Custom);

  //===--------------------------------------------------------------------===//
  // Memory
  //===--------------------------------------------------------------------===//

  for (MVT VT : {MVT::i16, MVT::i32, MVT::i64}) {
    setLoadExtAction({ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD}, VT,
                     MVT::i1, Promote);
    setLoadExtAction({ISD::SEXTLOAD}, VT, {MVT::i8, MVT::i16}, Expand);
  }
  setLoadExtAction({ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD}, MVT::i64,
                   {MVT::i8, MVT::i16, MVT::i32}, Expand);
  for (MVT MemVT : {MVT::i8, MVT::i16, MVT::i32})
    setTruncStoreAction(MVT::i64, MemVT, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, {MVT::f16, MVT::bf16}, Expand);
  setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::f32, MVT::bf16, Expand);

  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i32, Expand);

  // There is no library to call: memory intrinsics that AGXCodeGenPrepare
  // leaves alone are always expanded into loads and stores.
  MaxStoresPerMemcpy = MaxStoresPerMemcpyOptSize = ~0U;
  MaxStoresPerMemmove = MaxStoresPerMemmoveOptSize = ~0U;
  MaxStoresPerMemset = MaxStoresPerMemsetOptSize = ~0U;
}

static cl::opt<unsigned> AGXAddressingModes(
    "agx-addressing-modes", cl::init(1), cl::Hidden,
    cl::desc("Addressing modes reported to IR passes (LSR): 0 the generic "
             "default, 1 the hardware's base + immediate only, 2 also base "
             "+ scaled index"));

/// The addresses loads and stores take (SelectDeviceAddr, SelectLocalAddr):
/// a device access is a 64-bit base plus either a constant offset in
/// elements (16-bit signed) or an extended 32-bit index scaled by the
/// element size times 1, 2 or 4; threadgroup and private ones a base plus
/// a constant offset.
bool AGXTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                              const AddrMode &AM, Type *Ty,
                                              unsigned AS,
                                              Instruction *I) const {
  if (AGXAddressingModes == 0)
    return TargetLowering::isLegalAddressingMode(DL, AM, Ty, AS, I);
  if (AM.BaseGV || AM.ScalableOffset)
    return false;
  uint64_t Elem = 1;
  if (Ty && Ty->isSized())
    Elem = std::max<uint64_t>(
        1, DL.getTypeStoreSize(Ty->getScalarType()).getFixedValue());
  bool Device = AS != AGX::THREADGROUP_ADDRESS && AS != AGX::PRIVATE_ADDRESS;
  if (AM.Scale == 0) {
    if (!Device)
      return isInt<16>(AM.BaseOffs);
    return AM.BaseOffs % int64_t(Elem) == 0 &&
           isInt<16>(AM.BaseOffs / int64_t(Elem));
  }
  // A register without a base: base + 0.
  if (AM.Scale == 1 && !AM.HasBaseReg)
    return Device ? AM.BaseOffs % int64_t(Elem) == 0 &&
                        isInt<16>(AM.BaseOffs / int64_t(Elem))
                  : isInt<16>(AM.BaseOffs);
  if (!Device || AGXAddressingModes < 2 || AM.BaseOffs != 0)
    return false;
  return AM.Scale == int64_t(Elem) || AM.Scale == int64_t(2 * Elem) ||
         AM.Scale == int64_t(4 * Elem);
}

EVT AGXTargetLowering::getSetCCResultType(const DataLayout &DL,
                                          LLVMContext &Context,
                                          EVT VT) const {
  if (VT.isVector())
    return EVT::getVectorVT(Context, MVT::i32, VT.getVectorElementCount());
  return MVT::i32;
}

MVT AGXTargetLowering::getScalarShiftAmountTy(const DataLayout &DL,
                                              EVT VT) const {
  return VT == MVT::i16 ? MVT::i16 : MVT::i32;
}

bool AGXTargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                   EVT VT) const {
  VT = VT.getScalarType();
  return VT == MVT::f32 || VT == MVT::f16;
}

static cl::opt<unsigned> AGXSignedBytesALU(
    "agx-signed-bytes-alu", cl::Hidden, cl::init(3),
    cl::desc("Sign-extend bytes with an and (or a field extract), an xor and "
             "a subtraction instead of two shifts: 0 never, 1 the low byte, "
             "2 every byte, 3 every byte in functions with register headroom "
             "(-agx-headroom-gprs), else the low byte"));

static cl::opt<unsigned> AGXHeadroomGPRs(
    "agx-headroom-gprs", cl::Hidden, cl::init(24),
    cl::desc(
        "Functions whose estimated register pressure (agxEstimatePressure, "
        "32-bit registers) is at most this take the forms that trade "
        "registers for speed: their allocation stays well below the "
        "registers at which threads are lost"));

static cl::opt<bool> AGXDecomposeMul(
    "agx-decompose-mul", cl::Hidden, cl::init(true),
    cl::desc("Multiply 32-bit values by 3, 5, 9 or 17 with a shifted iadd"));

bool AGXTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                               SDValue C) const {
  // x * (2^k + 1), k <= 4: iadd x, x << k (an ALU instruction) instead of
  // imadd (SFU).
  auto *N = dyn_cast<ConstantSDNode>(C);
  if (!AGXDecomposeMul || VT != MVT::i32 || !N)
    return false;
  const APInt &V = N->getAPIntValue();
  return V == 3 || V == 5 || V == 9 || V == 17;
}

bool AGXTargetLowering::isTruncateFree(EVT SrcVT, EVT DstVT) const {
  // Truncation reads a subregister.
  return SrcVT.isScalarInteger() && DstVT.isScalarInteger() &&
         SrcVT.getSizeInBits() > DstVT.getSizeInBits() &&
         DstVT.getSizeInBits() >= 16;
}

TargetLowering::AtomicExpansionKind
AGXTargetLowering::shouldExpandAtomicRMWInIR(const AtomicRMWInst *RMW) const {
  if (RMW->getType()->isIntegerTy(32)) {
    switch (RMW->getOperation()) {
    case AtomicRMWInst::Add:
    case AtomicRMWInst::Sub:
    case AtomicRMWInst::Xchg:
    case AtomicRMWInst::Min:
    case AtomicRMWInst::Max:
    case AtomicRMWInst::UMin:
    case AtomicRMWInst::UMax:
    case AtomicRMWInst::And:
    case AtomicRMWInst::Or:
    case AtomicRMWInst::Xor:
      return AtomicExpansionKind::None;
    default:
      break;
    }
  }
  // A native float add on device memory where the chip has one; the
  // threadgroup atomic has none.
  if (RMW->getOperation() == AtomicRMWInst::FAdd && RMW->getType()->isFloatTy() &&
      ST.hasDeviceAtomicFAdd() &&
      RMW->getPointerAddressSpace() != AGX::THREADGROUP_ADDRESS)
    return AtomicExpansionKind::None;
  // A native u64 min / max on device memory when the old value is unused.
  if ((RMW->getOperation() == AtomicRMWInst::UMin ||
       RMW->getOperation() == AtomicRMWInst::UMax) &&
      RMW->getType()->isIntegerTy(64) && RMW->use_empty() &&
      ST.hasDeviceAtomicU64MinMax() &&
      RMW->getPointerAddressSpace() != AGX::THREADGROUP_ADDRESS)
    return AtomicExpansionKind::None;
  return AtomicExpansionKind::CmpXChg;
}

TargetLowering::AtomicExpansionKind
AGXTargetLowering::shouldExpandAtomicLoadInIR(LoadInst *LI) const {
  // System-scope 32-bit loads of device memory stay loads, encoded coherent
  // after an f59d: the only form that sees host writes during a dispatch
  // (ptx-re-blockers B7).
  if (isSystemScopeDeviceAccess(LI))
    return AtomicExpansionKind::None;
  // Other atomic loads: a compare and swap that never succeeds in changing
  // memory.
  return AtomicExpansionKind::CmpXChg;
}

TargetLowering::AtomicExpansionKind
AGXTargetLowering::shouldExpandAtomicStoreInIR(StoreInst *SI) const {
  // System-scope 32-bit stores of device memory: a coherent store and an
  // f533, the only form the host sees during a dispatch.
  if (isSystemScopeDeviceAccess(SI))
    return AtomicExpansionKind::None;
  // Atomic stores become atomic exchanges.
  return AtomicExpansionKind::Expand;
}

bool AGXTargetLowering::isSystemScopeDeviceAccess(const Instruction *I) {
  Type *Ty = isa<LoadInst>(I) ? I->getType()
                              : cast<StoreInst>(I)->getValueOperand()->getType();
  const DataLayout &DL = I->getDataLayout();
  return DL.getTypeSizeInBits(Ty) == 32 && AGX::isHostCoherentAccess(*I) &&
         getLoadStoreAddressSpace(I) == AGX::DEVICE_ADDRESS;
}

//===----------------------------------------------------------------------===//
// Calling convention
//===----------------------------------------------------------------------===//

// Experimental device calls pass legalized values in r2..r29. Sixteen-bit
// values occupy one low half; i64 values start at an even register and occupy
// two. The same window carries return values. This is intentionally separate
// from the kernel uniform-register convention.
static MCRegister callValueReg(MVT VT, unsigned &Slot,
                               const MCRegisterInfo &MRI) {
  unsigned Words = VT == MVT::i64 ? 2 : 1;
  if (VT == MVT::i64)
    Slot = alignTo(Slot, 2);
  if (Slot + Words > 28)
    return MCRegister();
  AGX::RegKind Kind = VT == MVT::i16 || VT == MVT::f16
                          ? AGX::RegKind::GPR16
                      : VT == MVT::i32 || VT == MVT::f32
                          ? AGX::RegKind::GPR32
                      : VT == MVT::i64 ? AGX::RegKind::GPR64
                                       : AGX::RegKind::Invalid;
  if (Kind == AGX::RegKind::Invalid)
    return MCRegister();
  MCRegister Reg = AGX::getRegister(Kind, (2 + Slot) * 2, MRI);
  Slot += Words;
  return Reg;
}

template <typename Args>
static bool canUseCallRegisters(const Args &Values,
                                const MCRegisterInfo &MRI) {
  unsigned Slot = 0;
  for (const auto &Value : Values)
    if (!callValueReg(Value.VT, Slot, MRI))
      return false;
  return true;
}

SDValue AGXTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto *MFI = MF.getInfo<AGXMachineFunctionInfo>();
  const MCRegisterInfo &MRI = *ST.getRegisterInfo();

  if (CallConv != CallingConv::AGX_KERNEL && ExperimentalLeafCalls) {
    if (IsVarArg || !canUseCallRegisters(Ins, MRI)) {
      diagnose(DAG, DL, "experimental AGX call arguments exceed r2..r29");
      for (const ISD::InputArg &In : Ins)
        InVals.push_back(DAG.getUNDEF(In.VT));
      return Chain;
    }
    unsigned Slot = 0;
    for (const ISD::InputArg &In : Ins) {
      MCRegister PReg = callValueReg(In.VT, Slot, MRI);
      Register VReg = MF.addLiveIn(PReg, getRegClassFor(In.VT));
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, In.VT));
    }
    return Chain;
  }

  if (IsVarArg)
    diagnose(DAG, DL, "variadic AGX kernels are not supported");

  // Arguments are packed into consecutive uniform registers in 16-bit units,
  // each aligned to its size (64-bit values to 32 bits), after the texture
  // heap address in a module that samples textures: texture_sample's bindless
  // base is a uniform below u64 at a multiple of four halves.
  unsigned Half = 0;
  if (AGX::usesTextureHeap(*MF.getFunction().getParent())) {
    MFI->setTextureHeap();
    Half = 4;
  }
  bool Overflow = false;
  auto Allocate = [&](MVT VT) -> MCRegister {
    unsigned Bits = VT.getSizeInBits();
    AGX::RegKind Kind = Bits == 16   ? AGX::RegKind::UGPR16
                        : Bits == 32 ? AGX::RegKind::UGPR32
                                     : AGX::RegKind::UGPR64;
    Half = alignTo(Half, Bits == 16 ? 1 : 2);
    MCRegister Reg = AGX::getRegister(Kind, Half, MRI);
    Half += Bits / 16;
    if (!Reg)
      Overflow = true;
    return Reg;
  };

  for (const ISD::InputArg &In : Ins) {
    MVT VT = In.VT;
    if (In.Flags.isByVal() || !(VT == MVT::i16 || VT == MVT::f16 ||
                                VT == MVT::i32 || VT == MVT::f32 ||
                                VT == MVT::i64)) {
      diagnose(DAG, DL, "unsupported AGX kernel argument type");
      InVals.push_back(DAG.getUNDEF(VT));
      continue;
    }
    MCRegister PReg = Allocate(VT);
    if (In.isOrigArg()) {
      unsigned Arg = In.getOrigArgIndex();
      if (MFI->ArgHalves.size() <= Arg)
        MFI->ArgHalves.resize(Arg + 1, ~0u);
      if (MFI->ArgHalves[Arg] == ~0u)
        MFI->ArgHalves[Arg] = Half - VT.getSizeInBits() / 16;
    }
    if (!PReg) {
      InVals.push_back(DAG.getUNDEF(VT));
      continue;
    }
    Register VReg = MF.addLiveIn(PReg, getRegClassFor(VT));
    InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, VT));
  }

  // Implicit arguments: threadgroups_per_grid.xyz.
  Half = alignTo(Half, 2);
  MFI->setImplicitArgHalfIndex(Half);
  for (unsigned Dim = 0; Dim < 3; ++Dim) {
    MCRegister PReg = Allocate(MVT::i32);
    if (PReg)
      MFI->setImplicitArgVReg(Dim,
                              MF.addLiveIn(PReg, &AGX::GPR32RegClass));
  }
  // The global data buffer address, if the kernel uses device globals (a
  // module's other globals, such as checked mode's site table, need none).
  if (AGX::usesGlobalData(MF.getFunction())) {
    unsigned GlobalHalf = alignTo(Half, 2);
    MCRegister PReg = Allocate(MVT::i64);
    if (PReg)
      MFI->setGlobalData(MF.addLiveIn(PReg, &AGX::GPR64RegClass), GlobalHalf);
  }
  // Values the launch computes from argument words (llvm.agx.launch.value).
  for (const Instruction &I : instructions(MF.getFunction())) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || II->getIntrinsicID() != Intrinsic::agx_launch_value)
      continue;
    unsigned Arg = cast<ConstantInt>(II->getArgOperand(0))->getZExtValue();
    unsigned Word = cast<ConstantInt>(II->getArgOperand(1))->getZExtValue();
    unsigned Kind = cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
    if (MFI->getLaunchValueVReg(Arg, Word, Kind))
      continue;
    Half = alignTo(Half, 2);
    unsigned At = Half;
    MCRegister PReg = Allocate(MVT::i32);
    if (PReg)
      MFI->LaunchValues.push_back(
          {Arg, Word, Kind, MF.addLiveIn(PReg, &AGX::GPR32RegClass), At});
  }
  MFI->setUniformHalfCount(Half);

  if (Overflow)
    diagnose(DAG, DL, "AGX kernel arguments exceed the uniform registers");
  return Chain;
}

SDValue
AGXTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SDLoc &DL, SelectionDAG &DAG) const {
  if (CallConv != CallingConv::AGX_KERNEL && ExperimentalLeafCalls) {
    const MCRegisterInfo &MRI = *ST.getRegisterInfo();
    if (!canUseCallRegisters(Outs, MRI))
      diagnose(DAG, DL, "experimental AGX call results exceed r2..r29");
    unsigned Slot = 0;
    SDValue Glue;
    SmallVector<SDValue, 8> Ops = {Chain};
    for (unsigned I = 0; I < Outs.size(); ++I) {
      MCRegister PReg = callValueReg(Outs[I].VT, Slot, MRI);
      if (PReg) {
        Chain = DAG.getCopyToReg(Chain, DL, PReg, OutVals[I], Glue);
        Glue = Chain.getValue(1);
        Ops.push_back(DAG.getRegister(PReg, Outs[I].VT));
      }
    }
    Ops[0] = Chain;
    if (Glue)
      Ops.push_back(Glue);
    return DAG.getNode(AGXISD::RET, DL, MVT::Other, Ops);
  }
  if (!Outs.empty())
    diagnose(DAG, DL, "AGX kernels cannot return values");
  return DAG.getNode(AGXISD::STOP, DL, MVT::Other, Chain);
}

SDValue AGXTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  const MCRegisterInfo &MRI = *ST.getRegisterInfo();
  bool ABIShape = canUseCallRegisters(CLI.Outs, MRI) &&
                  canUseCallRegisters(CLI.Ins, MRI);
  if (ExperimentalLeafCalls && ABIShape) {
    if (auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee)) {
      if (const auto *F = dyn_cast<Function>(G->getGlobal())) {
        const Function &Caller = DAG.getMachineFunction().getFunction();
        // (the call lowered is the caller's call of F, when it has one: no
        // walk of the whole caller for each of its calls)
        bool HasCall = CLI.CB && CLI.CB->getCalledFunction() == F;
        if (!HasCall)
          for (const BasicBlock &BB : Caller)
            for (const Instruction &I : BB)
              if (const auto *CB = dyn_cast<CallBase>(&I))
                if (CB->getCalledFunction() == F)
                  HasCall = true;
        // F's returns and whether its allocas and calls are supported, once
        // per callee while one caller is selected (only the caller's IR
        // changes meanwhile): a walk of F for each call was quadratic in
        // large modules with calls.
        auto &CalleeShapes = DAG.getMachineFunction()
                                 .getInfo<AGXMachineFunctionInfo>()
                                 ->CalleeShapes;
        auto [ShapeIt, NewShape] = CalleeShapes.try_emplace(F);
        if (NewShape) {
          bool Supported = true;
          unsigned Returns = 0;
          for (const BasicBlock &BB : *F)
            for (const Instruction &I : BB)
              if (isa<ReturnInst>(I))
                ++Returns;
              else if (const auto *AI = dyn_cast<AllocaInst>(&I))
                Supported &= AI->isStaticAlloca();
              else if (const auto *CB = dyn_cast<CallBase>(&I))
                Supported &= CB->getCalledFunction() != nullptr;
          ShapeIt->second = {Returns, Supported};
        }
        auto [Returns, Supported] = ShapeIt->second;
        bool Signature = !F->isVarArg() && !CLI.IsVarArg &&
                         !CLI.IsTailCall &&
                         CLI.Outs.empty() == F->arg_empty() &&
                         CLI.Ins.empty() == F->getReturnType()->isVoidTy() &&
                         !F->hasStructRetAttr();
        if (!F->isDeclaration() && Returns == 1 && HasCall && Supported &&
            Signature && F->hasFnAttribute(Attribute::NoInline)) {
          SDValue Chain = CLI.Chain;
          SDValue Glue;
          unsigned Slot = 0;
          SmallVector<SDValue, 8> ArgRegs;
          for (unsigned I = 0; I < CLI.OutVals.size(); ++I) {
            MCRegister PReg = callValueReg(CLI.Outs[I].VT, Slot, MRI);
            Chain = DAG.getCopyToReg(Chain, CLI.DL, PReg, CLI.OutVals[I], Glue);
            Glue = Chain.getValue(1);
            ArgRegs.push_back(DAG.getRegister(PReg, CLI.Outs[I].VT));
          }
          SDValue Target = DAG.getTargetGlobalAddress(F, CLI.DL, MVT::i64);
          const uint32_t *Mask = ST.getRegisterInfo()->getCallPreservedMask(
              DAG.getMachineFunction(), CallingConv::C);
          SmallVector<SDValue, 12> Ops = {Chain, Target,
                                          DAG.getRegisterMask(Mask)};
          Ops.append(ArgRegs);
          if (Glue)
            Ops.push_back(Glue);
          Chain = DAG.getNode(AGXISD::CALL, CLI.DL,
                              DAG.getVTList(MVT::Other, MVT::Glue), Ops);
          Glue = Chain.getValue(1);
          Slot = 0;
          for (const ISD::InputArg &In : CLI.Ins) {
            MCRegister PReg = callValueReg(In.VT, Slot, MRI);
            SDValue Result =
                DAG.getCopyFromReg(Chain, CLI.DL, PReg, In.VT, Glue);
            InVals.push_back(Result.getValue(0));
            Chain = Result.getValue(1);
            Glue = Result.getValue(2);
          }
          return Chain;
        }
      }
    }
  }
  StringRef Name = "<indirect>";
  if (auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee))
    Name = G->getGlobal()->getName();
  else if (auto *E = dyn_cast<ExternalSymbolSDNode>(CLI.Callee))
    Name = E->getSymbol();
  diagnose(DAG, CLI.DL, "unsupported AGX call to '" + Name + "'");
  for (const ISD::InputArg &In : CLI.Ins)
    InVals.push_back(DAG.getUNDEF(In.VT));
  return CLI.Chain;
}

//===----------------------------------------------------------------------===//
// Custom lowering
//===----------------------------------------------------------------------===//

/// The high half of a 64-bit value.
static SDValue hi32(SelectionDAG &DAG, const SDLoc &DL, SDValue V) {
  if (auto *C = dyn_cast<ConstantSDNode>(V))
    return DAG.getConstant(C->getZExtValue() >> 32, DL, MVT::i32);
  switch (V.getOpcode()) {
  case ISD::BUILD_PAIR:
    return V.getOperand(1);
  case ISD::ZERO_EXTEND:
    return DAG.getConstant(0, DL, MVT::i32);
  case ISD::SIGN_EXTEND:
    if (V.getOperand(0).getValueType() == MVT::i32)
      return DAG.getNode(ISD::SRA, DL, MVT::i32, V.getOperand(0),
                         DAG.getConstant(31, DL, MVT::i32));
    break;
  case ISD::ANY_EXTEND:
    return DAG.getUNDEF(MVT::i32);
  default:
    break;
  }
  return DAG.getNode(AGXISD::HI32, DL, MVT::i32, V);
}

/// Low and high halves of a 64-bit value, as nodes the combiner folds.
static std::pair<SDValue, SDValue> split64(SelectionDAG &DAG, const SDLoc &DL,
                                           SDValue V) {
  if (V.getOpcode() == ISD::BUILD_PAIR)
    return {V.getOperand(0), V.getOperand(1)};
  return {DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, V), hi32(DAG, DL, V)};
}

static SDValue join64(SelectionDAG &DAG, const SDLoc &DL, SDValue Lo,
                      SDValue Hi) {
  return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Lo, Hi);
}

static SDValue lowerLogicSelect64(SDValue Op, SelectionDAG &DAG) {
  SDLoc DL(Op);
  unsigned Opc = Op.getOpcode();
  if (Opc == ISD::SELECT) {
    auto [XLo, XHi] = split64(DAG, DL, Op.getOperand(1));
    auto [YLo, YHi] = split64(DAG, DL, Op.getOperand(2));
    SDValue C = Op.getOperand(0);
    return join64(DAG, DL, DAG.getSelect(DL, MVT::i32, C, XLo, YLo),
                  DAG.getSelect(DL, MVT::i32, C, XHi, YHi));
  }
  auto [ALo, AHi] = split64(DAG, DL, Op.getOperand(0));
  auto [BLo, BHi] = split64(DAG, DL, Op.getOperand(1));
  return join64(DAG, DL, DAG.getNode(Opc, DL, MVT::i32, ALo, BLo),
                DAG.getNode(Opc, DL, MVT::i32, AHi, BHi));
}

static cl::opt<bool> Convert8(
    "agx-convert8", cl::init(true), cl::Hidden,
    cl::desc("a float saturated to an 8-bit integer as the convert's 8-bit "
             "mode"));

static cl::opt<bool> SatConvertClamp(
    "agx-sat-convert-clamp",
    cl::desc("narrow saturating float conversions as the 16-bit convert and "
             "an integer clamp"),
    cl::init(true), cl::Hidden);

SDValue AGXTargetLowering::LowerOperation(SDValue Op,
                                          SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::STORE:
    return lowerVectorStore(Op, DAG);
  case ISD::LOAD:
    // reached only for loads ReplaceNodeResults leaves as they are
    return SDValue();
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN:
    return lowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID:
    return lowerINTRINSIC_VOID(Op, DAG);
  case ISD::ATOMIC_FENCE:
    return lowerATOMIC_FENCE(Op, DAG);
  case ISD::ATOMIC_LOAD_SUB:
    return lowerATOMIC_LOAD_SUB(Op, DAG);
  case ISD::BRCOND:
    return lowerBRCOND(Op, DAG);
  case ISD::SETCC:
    return lowerSETCC64(Op, DAG);
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SELECT:
    return lowerLogicSelect64(Op, DAG);
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    return lowerShift64(Op, DAG);
  case ISD::MUL:
    return lowerMUL64(Op, DAG);
  case ISD::CTLZ:
  case ISD::CTTZ:
    return lowerCTLZ_CTTZ64(Op, DAG);
  case ISD::AVGFLOORU:
  case ISD::AVGCEILU:
  case ISD::AVGFLOORS:
  case ISD::AVGCEILS: {
    SDLoc DL(Op);
    unsigned Opc = Op.getOpcode();
    bool Ceil = Opc == ISD::AVGCEILU || Opc == ISD::AVGCEILS;
    bool Signed = Opc == ISD::AVGFLOORS || Opc == ISD::AVGCEILS;
    SDValue A = Op.getOperand(0), B = Op.getOperand(1);
    SDValue Half = DAG.getNode(Signed ? ISD::SRA : ISD::SRL, DL, MVT::i32,
                               DAG.getNode(ISD::XOR, DL, MVT::i32, A, B),
                               DAG.getConstant(1, DL, MVT::i32));
    return DAG.getNode(Ceil ? ISD::SUB : ISD::ADD, DL, MVT::i32,
                       DAG.getNode(Ceil ? ISD::OR : ISD::AND, DL, MVT::i32, A,
                                   B),
                       Half);
  }
  case ISD::MULHU: {
    // With a = a1 2^32 + a0, b likewise, no sum overflows 64 bits:
    // t = a1 b0 + hi(a0 b0), u = a0 b1 + lo(t),
    // mulhu = a1 b1 + hi(t) + hi(u)
    SDLoc DL(Op);
    auto [A0, A1] = split64(DAG, DL, Op.getOperand(0));
    auto [B0, B1] = split64(DAG, DL, Op.getOperand(1));
    auto Wide = [&](SDValue X, SDValue Y) {
      return DAG.getNode(AGXISD::MUL_WIDE_U, DL, MVT::i64, X, Y);
    };
    SDValue C32 = DAG.getConstant(32, DL, MVT::i32);
    auto Hi = [&](SDValue X) {
      return DAG.getNode(ISD::SRL, DL, MVT::i64, X, C32);
    };
    auto Add = [&](SDValue X, SDValue Y) {
      return DAG.getNode(ISD::ADD, DL, MVT::i64, X, Y);
    };
    SDValue T = Add(Wide(A1, B0), Hi(Wide(A0, B0)));
    SDValue U = Add(Wide(A0, B1),
                    DAG.getNode(ISD::AND, DL, MVT::i64, T,
                                DAG.getConstant(0xffffffffu, DL, MVT::i64)));
    return Add(Add(Wide(A1, B1), Hi(T)), Hi(U));
  }
  case ISD::MULHS: {
    // mulhs(a, b) = mulhu(a, b) - (a < 0 ? b : 0) - (b < 0 ? a : 0)
    SDLoc DL(Op);
    SDValue A = Op.getOperand(0), B = Op.getOperand(1);
    SDValue C63 = DAG.getConstant(63, DL, MVT::i32);
    SDValue R = DAG.getNode(ISD::MULHU, DL, MVT::i64, A, B);
    R = DAG.getNode(ISD::SUB, DL, MVT::i64, R,
                    DAG.getNode(ISD::AND, DL, MVT::i64, B,
                                DAG.getNode(ISD::SRA, DL, MVT::i64, A, C63)));
    return DAG.getNode(
        ISD::SUB, DL, MVT::i64, R,
        DAG.getNode(ISD::AND, DL, MVT::i64, A,
                    DAG.getNode(ISD::SRA, DL, MVT::i64, B, C63)));
  }
  case ISD::UMUL_LOHI:
  case ISD::SMUL_LOHI: {
    SDLoc DL(Op);
    SDValue W =
        DAG.getNode(Op.getOpcode() == ISD::UMUL_LOHI ? AGXISD::MUL_WIDE_U
                                                     : AGXISD::MUL_WIDE_S,
                    DL, MVT::i64, Op.getOperand(0), Op.getOperand(1));
    auto [Lo, Hi] = split64(DAG, DL, W);
    return DAG.getMergeValues({Lo, Hi}, DL);
  }
  case ISD::BITREVERSE: {
    // the halves reversed by the 32-bit instruction, and swapped
    SDLoc DL(Op);
    auto [Lo, Hi] = split64(DAG, DL, Op.getOperand(0));
    return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64,
                       DAG.getNode(ISD::BITREVERSE, DL, MVT::i32, Hi),
                       DAG.getNode(ISD::BITREVERSE, DL, MVT::i32, Lo));
  }
  case ISD::FP_TO_BF16:
    return lowerFP_TO_BF16(Op, DAG);
  case ISD::STRICT_FP_TO_BF16:
    return DAG.getMergeValues({lowerFP_TO_BF16(Op, DAG), Op.getOperand(0)},
                              SDLoc(Op));
  case ISD::FDIV:
    return lowerFDIV(Op, DAG);
  case ISD::FSQRT:
    return lowerFSQRT(Op, DAG);
  case ISD::FREM:
    return lowerFREM(Op, DAG);
  case ISD::FCOPYSIGN:
    return lowerFCOPYSIGN(Op, DAG);
  case ISD::FMINIMUMNUM:
  case ISD::FMAXIMUMNUM:
  case ISD::FMINIMUM:
  case ISD::FMAXIMUM:
    return lowerFMinMaxNum(Op, DAG);
  case ISD::FROUND:
    return lowerFROUND(Op, DAG);
  case ISD::FSIN:
  case ISD::FCOS:
    return lowerTrig(Op, DAG);
  case ISD::FEXP:
  case ISD::FEXP10:
  case ISD::FLOG:
  case ISD::FLOG10:
  case ISD::FPOW:
    return lowerExpLog(Op, DAG);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return lowerINT_TO_FP64(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return lowerFP_TO_INT64(Op, DAG);
  case ISD::FP_TO_SINT_SAT:
  case ISD::FP_TO_UINT_SAT: {
    EVT Src = Op.getOperand(0).getValueType();
    EVT Sat = cast<VTSDNode>(Op.getOperand(1))->getVT();
    EVT VT = Op.getValueType();
    if (Src != MVT::f32 && Src != MVT::f16)
      return SDValue();
    if (Sat == VT)
      return Op;
    // A narrower saturation (an i8 result is promoted to i16 with an i8
    // saturation): the convert saturating to 16 bits, then for i8 the integer
    // clamp, instead of clamping the float against both bounds, testing it
    // for NaN and converting (the convert gives 0 for NaN, which the clamp
    // keeps). 2 instructions unsigned, 4 signed (one bound a mov_imm),
    // instead of 6 / 7.
    if (!SatConvertClamp || Sat.getSizeInBits() > 16)
      return SDValue();
    bool Signed = Op.getOpcode() == ISD::FP_TO_SINT_SAT;
    SDLoc DL(Op);
    unsigned W = Sat.getSizeInBits();
    EVT SrcVT = Op.getOperand(0).getValueType();
    if (W == 8 && Convert8 && (SrcVT == MVT::f32 || SrcVT == MVT::f16)) {
      // The convert's 8-bit modes saturate to the byte's range (NaN 0) and
      // zero-extend it (B27): unsigned is that, signed its sign extension
      // (which a byte store, reading only the byte, drops).
      // The convert truncates: a ftrunc before it (round's lowering) goes.
      SDValue Src = Op.getOperand(0);
      if (Src.getOpcode() == ISD::FTRUNC)
        Src = Src.getOperand(0);
      SDValue C = DAG.getNode(Signed ? AGXISD::CVT_F_TO_S8 : AGXISD::CVT_F_TO_U8,
                              DL, MVT::i16, Src);
      if (Signed)
        C = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i16, C,
                        DAG.getValueType(MVT::i8));
      return Signed ? DAG.getSExtOrTrunc(C, DL, VT) : DAG.getZExtOrTrunc(C, DL, VT);
    }
    SDValue C = DAG.getNode(Op.getOpcode(), DL, MVT::i16, Op.getOperand(0),
                            DAG.getValueType(MVT::i16));
    if (W < 16) {
      if (Signed) {
        C = DAG.getNode(ISD::SMAX, DL, MVT::i16, C,
                        DAG.getSignedConstant(-(int64_t(1) << (W - 1)), DL,
                                              MVT::i16));
        C = DAG.getNode(ISD::SMIN, DL, MVT::i16, C,
                        DAG.getConstant((1u << (W - 1)) - 1, DL, MVT::i16));
      } else {
        C = DAG.getNode(ISD::UMIN, DL, MVT::i16, C,
                        DAG.getConstant((1u << W) - 1, DL, MVT::i16));
      }
    }
    return Signed ? DAG.getSExtOrTrunc(C, DL, VT) : DAG.getZExtOrTrunc(C, DL, VT);
  }
  default:
    llvm_unreachable("unexpected custom lowering");
  }
}

/// Whether a load or store of VT is one xy / xyz / xyzw access: 2 to 4
/// elements of 32 bits, 4-byte aligned, in device or threadgroup memory, and
/// neither volatile nor atomic (those keep the scalar path).
static bool isVectorAccess(EVT VT, const MemSDNode *M) {
  unsigned AS = M->getAddressSpace();
  return VT.isVector() && VT.getScalarSizeInBits() == 32 &&
         VT.getVectorNumElements() >= 2 && VT.getVectorNumElements() <= 4 &&
         M->isSimple() && M->getAlign() >= Align(4) &&
         (AS == AGX::DEVICE_ADDRESS || AS == AGX::THREADGROUP_ADDRESS);
}

static unsigned vectorNode(bool Load, unsigned N) {
  static const unsigned Loads[] = {AGXISD::LOAD_V2, AGXISD::LOAD_V3,
                                   AGXISD::LOAD_V4};
  static const unsigned Stores[] = {AGXISD::STORE_V2, AGXISD::STORE_V3,
                                    AGXISD::STORE_V4};
  return (Load ? Loads : Stores)[N - 2];
}

static cl::opt<bool> AGXElementAlignedVectors(
    "agx-element-aligned-vectors", cl::init(false), cl::Hidden,
    cl::desc("Vector accesses of 8- and 16-bit elements need only the "
             "element's alignment (not measured on hardware; 32-bit elements "
             "at 4 bytes are: ptx-re-blockers B33)"));

/// A 128-bit shift by a variable amount n (below 128: more is poison) on the
/// 32-bit words: the words moved by 64 where n's bit 6 is set, then by 32
/// where bit 5 is, then each result word a funnel shift (extr) of two
/// neighbours by n mod 32 (the type legalizer's expansion through 64-bit
/// halves takes about 23 instructions, this about 15).
static SDValue lowerShift128(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  unsigned Opc = N->getOpcode();
  SDValue X = N->getOperand(0);
  SDValue Amt = DAG.getZExtOrTrunc(N->getOperand(1), DL, MVT::i32);
  auto C32 = [&](uint64_t V) { return DAG.getConstant(V, DL, MVT::i32); };
  SDValue W[4];
  for (unsigned I = 0; I < 4; ++I)
    W[I] = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32,
                       DAG.getNode(ISD::SRL, DL, MVT::i128, X,
                                   DAG.getConstant(32 * I, DL, MVT::i32)));
  SDValue Fill = Opc == ISD::SRA
                     ? DAG.getNode(ISD::SRA, DL, MVT::i32, W[3], C32(31))
                     : C32(0);
  auto Bit = [&](unsigned B) {
    return DAG.getSetCC(DL, MVT::i32,
                        DAG.getNode(ISD::AND, DL, MVT::i32, Amt, C32(B)),
                        C32(0), ISD::SETNE);
  };
  auto Sel = [&](SDValue C, SDValue T, SDValue F) {
    return DAG.getSelect(DL, MVT::i32, C, T, F);
  };
  // word moves by 64 and 32 places (K words), filled with Fill
  auto Move = [&](SDValue C, unsigned K) {
    SDValue V[4];
    for (int I = 0; I < 4; ++I) {
      int From = Opc == ISD::SHL ? I - int(K) : I + int(K);
      SDValue Moved = From >= 0 && From < 4 ? W[From] : Fill;
      V[I] = Sel(C, Moved, W[I]);
    }
    for (int I = 0; I < 4; ++I)
      W[I] = V[I];
  };
  // (n below 128: bit 6 is n >= 64, which the select's compare takes)
  Move(DAG.getSetCC(DL, MVT::i32, Amt, C32(64), ISD::SETUGE), 2);
  Move(Bit(32), 1);
  // (the funnel shifts take n mod 32 themselves: one and for all)
  SDValue S = DAG.getNode(ISD::AND, DL, MVT::i32, Amt, C32(31));
  SDValue R[4];
  if (Opc == ISD::SHL) {
    for (int I = 3; I > 0; --I)
      R[I] = DAG.getNode(ISD::FSHL, DL, MVT::i32, W[I], W[I - 1], Amt);
    R[0] = DAG.getNode(ISD::SHL, DL, MVT::i32, W[0], S);
  } else {
    for (int I = 0; I < 3; ++I)
      R[I] = DAG.getNode(ISD::FSHR, DL, MVT::i32, W[I + 1], W[I], Amt);
    R[3] = DAG.getNode(Opc, DL, MVT::i32, W[3], S);
  }
  auto Pair = [&](SDValue Lo, SDValue Hi) {
    return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Lo, Hi);
  };
  return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i128, Pair(R[0], R[1]),
                     Pair(R[2], R[3]));
}

void AGXTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  if ((N->getOpcode() == ISD::SHL || N->getOpcode() == ISD::SRL ||
       N->getOpcode() == ISD::SRA) &&
      N->getValueType(0) == MVT::i128 &&
      !isa<ConstantSDNode>(N->getOperand(1))) {
    Results.push_back(lowerShift128(N, DAG));
    return;
  }
  // llvm.agx.texture.sample: one i32 result per channel.
  if (N->getOpcode() == ISD::INTRINSIC_W_CHAIN &&
      N->getConstantOperandVal(1) == Intrinsic::agx_texture_sample) {
    SDLoc DL(N);
    SDValue Sample = DAG.getNode(
        AGXISD::TEXTURE_SAMPLE, DL,
        DAG.getVTList({MVT::i32, MVT::i32, MVT::i32, MVT::i32, MVT::Other}),
        {N->getOperand(0), N->getOperand(2), N->getOperand(3), N->getOperand(4),
         N->getOperand(5),
         DAG.getTargetConstant(N->getConstantOperandVal(6), DL, MVT::i32)});
    Results.push_back(DAG.getBuildVector(
        MVT::v4i32, DL,
        {Sample.getValue(0), Sample.getValue(1), Sample.getValue(2),
         Sample.getValue(3)}));
    Results.push_back(Sample.getValue(4));
    return;
  }
  // Byte vector loads: one access of the byte format, a 16-bit result per
  // byte. The PTX alignment of ld.v2.u8 / ld.v4.u8 is the vector's size.
  if (auto *L = dyn_cast<LoadSDNode>(N);
      L && L->getExtensionType() == ISD::NON_EXTLOAD && !L->isIndexed() &&
      L->isSimple() && L->getAddressSpace() == AGX::DEVICE_ADDRESS &&
      (L->getValueType(0) == MVT::v2i8 || L->getValueType(0) == MVT::v4i8) &&
      (AGXElementAlignedVectors ||
       L->getAlign() >= Align(L->getValueType(0).getStoreSize()))) {
    SDLoc DL(N);
    EVT VT = L->getValueType(0);
    unsigned NumElts = VT.getVectorNumElements();
    SmallVector<EVT, 5> VTs(NumElts, MVT::i16);
    VTs.push_back(MVT::Other);
    SDValue Load = DAG.getMemIntrinsicNode(
        NumElts == 2 ? AGXISD::LOAD_V2_I8 : AGXISD::LOAD_V4_I8, DL,
        DAG.getVTList(VTs), {L->getChain(), L->getBasePtr()},
        L->getMemoryVT(), L->getMemOperand());
    SmallVector<SDValue, 4> Elts;
    for (unsigned I = 0; I < NumElts; ++I)
      Elts.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i8, Load.getValue(I)));
    Results.push_back(DAG.getBuildVector(VT, DL, Elts));
    Results.push_back(Load.getValue(NumElts));
    return;
  }
  // 16-bit elements below 4-byte alignment (-agx-element-aligned-vectors):
  // one device access of the i16 format into consecutive 16-bit halves.
  if (auto *L = dyn_cast<LoadSDNode>(N);
      AGXElementAlignedVectors && L &&
      L->getExtensionType() == ISD::NON_EXTLOAD && !L->isIndexed() &&
      L->isSimple() && L->getAlign() < Align(4) &&
      L->getAddressSpace() == AGX::DEVICE_ADDRESS &&
      (L->getValueType(0) == MVT::v2i16 || L->getValueType(0) == MVT::v4i16 ||
       L->getValueType(0) == MVT::v2f16 || L->getValueType(0) == MVT::v4f16)) {
    SDLoc DL(N);
    EVT VT = L->getValueType(0);
    unsigned NumElts = VT.getVectorNumElements();
    SmallVector<EVT, 5> VTs(NumElts, MVT::i16);
    VTs.push_back(MVT::Other);
    SDValue Load = DAG.getMemIntrinsicNode(
        NumElts == 2 ? AGXISD::LOAD_V2_I8 : AGXISD::LOAD_V4_I8, DL,
        DAG.getVTList(VTs), {L->getChain(), L->getBasePtr()},
        L->getMemoryVT(), L->getMemOperand());
    SmallVector<SDValue, 4> Elts;
    for (unsigned I = 0; I < NumElts; ++I)
      Elts.push_back(DAG.getBitcast(VT.getScalarType(), Load.getValue(I)));
    Results.push_back(DAG.getBuildVector(VT, DL, Elts));
    Results.push_back(Load.getValue(NumElts));
    return;
  }
  // Pairs of 16-bit elements: a 32-bit access each, its low half the first
  // element (a 16-bit half register, no instruction to split it).
  if (auto *L = dyn_cast<LoadSDNode>(N);
      L && L->getExtensionType() == ISD::NON_EXTLOAD && !L->isIndexed() &&
      L->isSimple() && L->getAlign() >= Align(4) &&
      (L->getAddressSpace() == AGX::DEVICE_ADDRESS ||
       L->getAddressSpace() == AGX::THREADGROUP_ADDRESS) &&
      (L->getValueType(0) == MVT::v2i16 || L->getValueType(0) == MVT::v4i16 ||
       L->getValueType(0) == MVT::v2f16 || L->getValueType(0) == MVT::v4f16)) {
    SDLoc DL(N);
    EVT VT = L->getValueType(0);
    unsigned Words = VT.getVectorNumElements() / 2;
    EVT WordsVT = Words == 1 ? EVT(MVT::i32) : EVT(MVT::v2i32);
    SDValue Load = DAG.getLoad(WordsVT, DL, L->getChain(), L->getBasePtr(),
                               L->getMemOperand());
    SmallVector<SDValue, 4> Elts;
    for (unsigned W = 0; W < Words; ++W) {
      SDValue Word = Words == 1 ? Load
                                : DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                                              MVT::i32, Load,
                                              DAG.getVectorIdxConstant(W, DL));
      SDValue Hi = DAG.getNode(ISD::SRL, DL, MVT::i32, Word,
                               DAG.getConstant(16, DL, MVT::i32));
      for (SDValue Half : {Word, Hi})
        Elts.push_back(DAG.getBitcast(
            VT.getScalarType(), DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, Half)));
    }
    Results.push_back(DAG.getBuildVector(VT, DL, Elts));
    Results.push_back(Load.getValue(1));
    return;
  }
  // Vector loads: one access with a 32-bit result per element.
  auto *L = dyn_cast<LoadSDNode>(N);
  if (!L || L->getExtensionType() != ISD::NON_EXTLOAD || L->isIndexed() ||
      !isVectorAccess(L->getValueType(0), L))
    return;
  SDLoc DL(N);
  EVT VT = L->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();
  SmallVector<EVT, 5> VTs(NumElts, MVT::i32);
  VTs.push_back(MVT::Other);
  SDValue Load = DAG.getMemIntrinsicNode(
      vectorNode(true, NumElts), DL, DAG.getVTList(VTs),
      {L->getChain(), L->getBasePtr()}, L->getMemoryVT(), L->getMemOperand());
  SmallVector<SDValue, 4> Elts;
  for (unsigned I = 0; I < NumElts; ++I)
    Elts.push_back(DAG.getBitcast(VT.getScalarType(), Load.getValue(I)));
  Results.push_back(DAG.getBuildVector(VT, DL, Elts));
  Results.push_back(Load.getValue(NumElts));
}

/// Device and threadgroup accesses of up to 128 bits need only 4-byte
/// alignment: xy / xyzw accesses at every byte offset 0, 4, 8, 12 mod 16, also
/// across 64-byte lines and 4 KiB pages, match the byte-address model on the
/// M2 Max (ptx-re-blockers B33).
bool AGXTTIImpl::elementAlignedVectors() { return AGXElementAlignedVectors; }

static cl::opt<bool> AGXBytePairChains(
    "agx-byte-pair-chains", cl::init(true), cl::Hidden,
    cl::desc("Merge two neighbouring 2-byte aligned device bytes into one "
             "access of the byte format"));
bool AGXTTIImpl::bytePairChains() { return AGXBytePairChains; }

bool AGXTargetLowering::allowsMisalignedMemoryAccesses(
    EVT VT, unsigned AS, Align Alignment, MachineMemOperand::Flags Flags,
    unsigned *Fast) const {
  Align Need(4);
  if (AGXElementAlignedVectors && VT.isVector() &&
      VT.getScalarSizeInBits() < 32)
    Need = Align(VT.getScalarSizeInBits() / 8);
  // The load/store vectorizer asks for its chain as one integer: with the
  // option, chains of 8- or 16-bit elements at their own alignment.
  if (AGXElementAlignedVectors && VT.isScalarInteger())
    Need = Align(1);
  bool OK = (AS == AGX::DEVICE_ADDRESS || AS == AGX::THREADGROUP_ADDRESS) &&
            Alignment >= Need && VT.getStoreSize() <= 16;
  if (Fast)
    *Fast = OK;
  return OK;
}

static cl::opt<bool> AGXByteVectorStores(
    "agx-byte-vector-stores", cl::init(true), cl::Hidden,
    cl::desc("Store 2 or 4 bytes with one device_store of the byte format"));

SDValue AGXTargetLowering::lowerVectorStore(SDValue Op,
                                            SelectionDAG &DAG) const {
  auto *S = cast<StoreSDNode>(Op);
  SDValue Val = S->getValue();
  EVT VT = Val.getValueType();
  SDLoc DL(Op);
  // 2 or 4 bytes, naturally aligned (PTX st.v2.u8 / st.v4.u8): one access of
  // the byte format from 16-bit values
  if (AGXByteVectorStores && (VT == MVT::v2i8 || VT == MVT::v4i8) &&
      !S->isTruncatingStore() &&
      !S->isIndexed() && S->isSimple() &&
      S->getAddressSpace() == AGX::DEVICE_ADDRESS &&
      S->getAlign() >= Align(VT.getStoreSize())) {
    unsigned NumElts = VT.getVectorNumElements();
    SmallVector<SDValue, 6> Ops{S->getChain()};
    for (unsigned I = 0; I < NumElts; ++I)
      Ops.push_back(DAG.getNode(
          ISD::ANY_EXTEND, DL, MVT::i16,
          DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i8, Val,
                      DAG.getVectorIdxConstant(I, DL))));
    Ops.push_back(S->getBasePtr());
    return DAG.getMemIntrinsicNode(
        NumElts == 2 ? AGXISD::STORE_V2_I8 : AGXISD::STORE_V4_I8, DL,
        DAG.getVTList(MVT::Other), Ops, S->getMemoryVT(), S->getMemOperand());
  }
  // Pairs of 16-bit elements, 4-byte aligned: a 32-bit word each, whose
  // halves are the elements (a register's halves, no instruction to pack).
  if ((VT == MVT::v2i16 || VT == MVT::v4i16 || VT == MVT::v2f16 ||
       VT == MVT::v4f16) &&
      !S->isTruncatingStore() && !S->isIndexed() && S->isSimple() &&
      S->getAlign() >= Align(4) &&
      (S->getAddressSpace() == AGX::DEVICE_ADDRESS ||
       S->getAddressSpace() == AGX::THREADGROUP_ADDRESS)) {
    unsigned Words = VT.getVectorNumElements() / 2;
    SmallVector<SDValue, 2> W;
    for (unsigned I = 0; I < Words; ++I) {
      SDValue Half[2];
      for (unsigned K = 0; K < 2; ++K)
        Half[K] = DAG.getNode(
            ISD::ANY_EXTEND, DL, MVT::i32,
            DAG.getBitcast(MVT::i16,
                           DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                                       VT.getScalarType(), Val,
                                       DAG.getVectorIdxConstant(2 * I + K, DL))));
      W.push_back(DAG.getNode(AGXISD::PACK16, DL, MVT::i32, Half[0], Half[1]));
    }
    if (Words == 1)
      return DAG.getStore(S->getChain(), DL, W[0], S->getBasePtr(),
                          S->getMemOperand());
    return DAG.getMemIntrinsicNode(vectorNode(false, 2), DL,
                                   DAG.getVTList(MVT::Other),
                                   {S->getChain(), W[0], W[1], S->getBasePtr()},
                                   S->getMemoryVT(), S->getMemOperand());
  }
  if (S->isTruncatingStore() || S->isIndexed() || !isVectorAccess(VT, S))
    return SDValue();
  SmallVector<SDValue, 6> Ops{S->getChain()};
  for (unsigned I = 0; I < VT.getVectorNumElements(); ++I)
    Ops.push_back(DAG.getBitcast(
        MVT::i32, DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT.getScalarType(),
                              Val, DAG.getVectorIdxConstant(I, DL))));
  Ops.push_back(S->getBasePtr());
  return DAG.getMemIntrinsicNode(vectorNode(false, VT.getVectorNumElements()),
                                 DL, DAG.getVTList(MVT::Other), Ops,
                                 S->getMemoryVT(), S->getMemOperand());
}

SDValue AGXTargetLowering::lowerGlobalAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *GA = cast<GlobalAddressSDNode>(Op);
  const auto *GV = dyn_cast<GlobalVariable>(GA->getGlobal());
  MachineFunction &MF = DAG.getMachineFunction();
  auto *MFI = MF.getInfo<AGXMachineFunctionInfo>();

  // Device and constant variables: an offset into the global data buffer.
  if (GV && AGX::isGlobalDataVariable(*GV)) {
    std::optional<uint64_t> Offset =
        AGX::computeGlobalDataLayout(*MF.getFunction().getParent())
            .getOffset(GV);
    if (!Offset || !MFI->getGlobalDataVReg()) {
      diagnose(DAG, DL,
               "unsupported external global '" + GV->getName() + "'");
      return DAG.getUNDEF(Op.getValueType());
    }
    SDValue Base = DAG.getCopyFromReg(DAG.getEntryNode(), DL,
                                      MFI->getGlobalDataVReg(), MVT::i64);
    SDValue Addr =
        DAG.getNode(ISD::ADD, DL, MVT::i64, Base,
                    DAG.getConstant(*Offset + GA->getOffset(), DL, MVT::i64));
    return DAG.getZExtOrTrunc(Addr, DL, Op.getValueType());
  }

  if (!GV || GA->getAddressSpace() != AGX::THREADGROUP_ADDRESS) {
    diagnose(DAG, DL,
             "unsupported global '" + GA->getGlobal()->getName() + "'");
    return DAG.getUNDEF(Op.getValueType());
  }
  if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer()))
    diagnose(DAG, DL, "threadgroup variables cannot have initializers");

  // Threadgroup variables are allocated statically per kernel; their
  // address is a constant offset into threadgroup memory.
  unsigned Offset =
      MFI->allocateThreadgroupVariable(DAG.getMachineFunction().getFunction(), *GV);
  return DAG.getConstant(Offset + GA->getOffset(), DL, Op.getValueType());
}

SDValue AGXTargetLowering::lowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  unsigned IID = Op.getConstantOperandVal(0);
  switch (IID) {
  case Intrinsic::agx_threadgroups_per_grid_x:
  case Intrinsic::agx_threadgroups_per_grid_y:
  case Intrinsic::agx_threadgroups_per_grid_z: {
    auto *MFI = DAG.getMachineFunction().getInfo<AGXMachineFunctionInfo>();
    Register VReg =
        MFI->getImplicitArgVReg(IID - Intrinsic::agx_threadgroups_per_grid_x);
    if (!VReg)
      return DAG.getUNDEF(MVT::i32);
    return DAG.getCopyFromReg(DAG.getEntryNode(), DL, VReg, MVT::i32);
  }
  case Intrinsic::agx_launch_value: {
    auto *MFI = DAG.getMachineFunction().getInfo<AGXMachineFunctionInfo>();
    Register VReg = MFI->getLaunchValueVReg(Op.getConstantOperandVal(1),
                                            Op.getConstantOperandVal(2),
                                            Op.getConstantOperandVal(3));
    if (!VReg)
      return DAG.getUNDEF(MVT::i32);
    return DAG.getCopyFromReg(DAG.getEntryNode(), DL, VReg, MVT::i32);
  }
  case Intrinsic::agx_sin:
  case Intrinsic::agx_cos:
  case Intrinsic::agx_exp2:
  case Intrinsic::agx_log2: {
    // the nodes of llvm.sin and the others, which the DAG does not fold
    unsigned Opc = IID == Intrinsic::agx_sin   ? ISD::FSIN
                   : IID == Intrinsic::agx_cos ? ISD::FCOS
                   : IID == Intrinsic::agx_exp2 ? ISD::FEXP2
                                                : ISD::FLOG2;
    return DAG.getNode(Opc, DL, Op.getValueType(), Op.getOperand(1));
  }
  case Intrinsic::agx_simd_ballot:
  case Intrinsic::agx_quad_ballot: {
    SDValue Cond = DAG.getZExtOrTrunc(Op.getOperand(1), DL, MVT::i32);
    return DAG.getNode(IID == Intrinsic::agx_simd_ballot
                           ? AGXISD::BALLOT
                           : AGXISD::QUAD_BALLOT,
                       DL, MVT::i32, Cond);
  }
  default:
    return SDValue();
  }
}

SDValue AGXTargetLowering::lowerINTRINSIC_VOID(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  switch (Op.getConstantOperandVal(1)) {
  case Intrinsic::agx_end_cf:
    return DAG.getNode(AGXISD::END_CF, DL, MVT::Other, Op.getOperand(0));
  case Intrinsic::agx_loop_begin:
    return DAG.getNode(AGXISD::LOOP_BEGIN, DL, MVT::Other, Op.getOperand(0));
  default:
    return SDValue();
  }
}

// The memory barriers of G13/G14, by their 16-bit encodings (memory_barrier
// a, b, c is 0xf5 | b << 8 | a << 10 | c << 12). Their effect was measured
// with cross-core litmus tests on G14C (M2 Max) under Asahi Linux
// (Apple-M2-Playground novel-results 2026-10-11-memory-ordering,
// 2026-10-18-sb-lb, 2026-10-18-atomic-release, 2026-10-18-fence-cost):
//   f596 (Mesa's memory_barrier) does not order stores between cores;
//   f533 is the store release (also needed before an atomic flag);
//   wait 0 then f59d is the acquire;
//   store buffering needs the whole sequence Mesa uses at queue-family scope.
// Threads of one threadgroup run on one core, where no reordering was seen.
enum AGXBarrier : unsigned { F596, F533, F59D, F59A, F59E, F503, F599 };

static SDValue emitBarrier(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                           AGXBarrier B) {
  static const unsigned Ops[][3] = {
      {1, 2, 9}, {0, 3, 3}, {3, 1, 9}, {2, 2, 9}, {3, 2, 9}, {0, 3, 0},
      {2, 1, 9}};
  SDValue Args[] = {
      Chain, DAG.getTargetConstant(Intrinsic::agx_memory_barrier, DL, MVT::i32),
      DAG.getTargetConstant(Ops[B][0], DL, MVT::i32),
      DAG.getTargetConstant(Ops[B][1], DL, MVT::i32),
      DAG.getTargetConstant(Ops[B][2], DL, MVT::i32)};
  return DAG.getNode(ISD::INTRINSIC_VOID, DL, MVT::Other, Args);
}

static SDValue emitWait(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain) {
  SDValue Args[] = {Chain,
                    DAG.getTargetConstant(Intrinsic::agx_wait, DL, MVT::i32),
                    DAG.getTargetConstant(0, DL, MVT::i32)};
  return DAG.getNode(ISD::INTRINSIC_VOID, DL, MVT::Other, Args);
}

SDValue AGXTargetLowering::lowerATOMIC_FENCE(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  auto Ordering = static_cast<AtomicOrdering>(Op.getConstantOperandVal(1));
  auto Scope = static_cast<SyncScope::ID>(Op.getConstantOperandVal(2));
  if (Scope == SyncScope::SingleThread)
    return Chain;
  LLVMContext &Ctx = *DAG.getContext();
  if (Scope == Ctx.getOrInsertSyncScopeID("workgroup") ||
      Scope == Ctx.getOrInsertSyncScopeID("workgroup-one-as"))
    return emitBarrier(DAG, DL, Chain, F596);
  // Device and system scope.
  switch (Ordering) {
  case AtomicOrdering::Acquire:
    return emitBarrier(DAG, DL, emitWait(DAG, DL, Chain), F59D);
  case AtomicOrdering::Release:
    return emitBarrier(DAG, DL, Chain, F533);
  case AtomicOrdering::AcquireRelease:
    Chain = emitBarrier(DAG, DL, Chain, F533);
    return emitBarrier(DAG, DL, emitWait(DAG, DL, Chain), F59D);
  default:
    for (AGXBarrier B : {F596, F533, F59A, F59E, F503, F599})
      Chain = emitBarrier(DAG, DL, Chain, B);
    return emitBarrier(DAG, DL, emitWait(DAG, DL, Chain), F59D);
  }
}

Instruction *AGXTargetLowering::emitLeadingFence(IRBuilderBase &Builder,
                                                 Instruction *Inst,
                                                 AtomicOrdering Ord) const {
  if (isReleaseOrStronger(Ord) && Inst->hasAtomicStore())
    return Builder.CreateFence(isAcquireOrStronger(Ord) ? AtomicOrdering::Release
                                                        : Ord,
                               getAtomicSyncScopeID(Inst).value_or(
                                   SyncScope::System));
  return nullptr;
}

Instruction *AGXTargetLowering::emitTrailingFence(IRBuilderBase &Builder,
                                                  Instruction *Inst,
                                                  AtomicOrdering Ord) const {
  if (isAcquireOrStronger(Ord))
    return Builder.CreateFence(
        Ord == AtomicOrdering::SequentiallyConsistent
            ? AtomicOrdering::SequentiallyConsistent
            : AtomicOrdering::Acquire,
        getAtomicSyncScopeID(Inst).value_or(SyncScope::System));
  return nullptr;
}

SDValue AGXTargetLowering::lowerATOMIC_LOAD_SUB(SDValue Op,
                                               SelectionDAG &DAG) const {
  // x - v is x + (-v). The add keeps the meaning of the opcode with the
  // coherency bit set, which turns the sub opcode into a float add on G14.
  auto *N = cast<AtomicSDNode>(Op.getNode());
  SDLoc DL(Op);
  SDValue Neg = DAG.getNode(ISD::SUB, DL, MVT::i32,
                            DAG.getConstant(0, DL, MVT::i32), N->getVal());
  return DAG.getAtomic(ISD::ATOMIC_LOAD_ADD, DL, N->getMemoryVT(),
                       N->getChain(), N->getBasePtr(), Neg,
                       N->getMemOperand());
}

SDValue AGXTargetLowering::lowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  // Every conditional branch is rewritten into execution mask operations by
  // performBRCONDCombine; anything else is unstructured control flow.
  diagnose(DAG, SDLoc(Op),
           "unstructured conditional branch (AGXAnnotateControlFlow did not "
           "run)");
  return Op.getOperand(0);
}

/// Rewrite
///
///   brcond (llvm.agx.if/else/loop ...), %a
///   br %b
///
/// to the target node, branching to the block executed when no thread
/// remains active, and an unconditional branch to the other block. This is
/// done before type legalization so that the i1 results of the control flow
/// intrinsics never need to be legalized.
SDValue AGXTargetLowering::performBRCONDCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  SDValue Cond = N->getOperand(1);
  SDValue Target = N->getOperand(2);

  bool Negated = false;
  if (Cond.getOpcode() == ISD::XOR && isOneConstant(Cond.getOperand(1))) {
    Negated = true;
    Cond = Cond.getOperand(0);
  } else if (Cond.getOpcode() == ISD::SETCC &&
             isa<ConstantSDNode>(Cond.getOperand(1))) {
    ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
    bool IsOne = isOneConstant(Cond.getOperand(1));
    bool IsZero = isNullConstant(Cond.getOperand(1));
    if ((CC == ISD::SETEQ && IsZero) || (CC == ISD::SETNE && IsOne))
      Negated = true;
    else if (!((CC == ISD::SETNE && IsZero) || (CC == ISD::SETEQ && IsOne)))
      return SDValue();
    Cond = Cond.getOperand(0);
  }

  if (Cond.getOpcode() != ISD::INTRINSIC_W_CHAIN)
    return SDValue();
  unsigned IID = Cond.getConstantOperandVal(1);
  unsigned Opc;
  switch (IID) {
  case Intrinsic::agx_if:
    Opc = AGXISD::IF;
    break;
  case Intrinsic::agx_else:
    Opc = AGXISD::ELSE;
    break;
  case Intrinsic::agx_loop:
    Opc = AGXISD::LOOP;
    break;
  default:
    return SDValue();
  }
  SDNode *Intr = Cond.getNode();

  // The block to skip to. Without negation, it is the destination of the
  // unconditional branch that follows, which is redirected to the original
  // destination.
  SDNode *BR = nullptr;
  if (!Negated) {
    for (SDNode *U : N->users())
      if (U->getOpcode() == ISD::BR)
        BR = U;
    if (!BR)
      return SDValue();
    Target = BR->getOperand(1);
  }

  SmallVector<SDValue, 3> Ops = {N->getOperand(0)};
  if (Opc != AGXISD::ELSE)
    Ops.push_back(DAG.getZExtOrTrunc(Intr->getOperand(2), DL, MVT::i32));
  Ops.push_back(Target);
  SDValue Result = DAG.getNode(Opc, DL, MVT::Other, Ops);

  if (BR) {
    SDValue NewBR = DAG.getNode(ISD::BR, SDLoc(BR), MVT::Other,
                                BR->getOperand(0), N->getOperand(2));
    DAG.ReplaceAllUsesWith(BR, NewBR.getNode());
  }

  // Remove the intrinsic from the chain.
  DAG.ReplaceAllUsesOfValueWith(SDValue(Intr, Intr->getNumValues() - 1),
                                Intr->getOperand(0));
  return Result;
}

/// Whether V (i16) is a boolean expression: compares, 0 / 1 constants,
/// truncations of 32-bit 0 / 1 values, and and / or / xor of those, each used
/// once.
static bool isBoolean16(SDValue V, SelectionDAG &DAG, unsigned Depth) {
  if (auto *C = dyn_cast<ConstantSDNode>(V))
    return C->getZExtValue() <= 1;
  if (Depth > 4 || !V.hasOneUse())
    return false;
  switch (V.getOpcode()) {
  case ISD::SETCC:
    return true;
  case ISD::TRUNCATE:
    // Of a value that is 0 or 1.
    return V.getOperand(0).getValueType() == MVT::i32 &&
           DAG.computeKnownBits(V.getOperand(0)).countMaxActiveBits() <= 1;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    return isBoolean16(V.getOperand(0), DAG, Depth + 1) &&
           isBoolean16(V.getOperand(1), DAG, Depth + 1);
  default:
    return false;
  }
}

/// A boolean expression accepted by isBoolean16, rebuilt in i32.
static SDValue widenBoolean(SDValue V, SelectionDAG &DAG, const SDLoc &DL) {
  if (auto *C = dyn_cast<ConstantSDNode>(V))
    return DAG.getConstant(C->getZExtValue(), DL, MVT::i32);
  if (V.getOpcode() == ISD::TRUNCATE)
    return V.getOperand(0);
  if (V.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::SETCC, DL, MVT::i32, V.getOperand(0),
                       V.getOperand(1), V.getOperand(2));
  return DAG.getNode(V.getOpcode(), DL, MVT::i32,
                     widenBoolean(V.getOperand(0), DAG, DL),
                     widenBoolean(V.getOperand(1), DAG, DL));
}

/// Byte fields inserted by bfi where the or-of-masks form would take two or
/// three instructions: (x & m) | (y & ~m), m the low w bits (w a multiple of
/// 8), is bfi(y, x, 0, w), y needing no mask (bfi keeps it outside the
/// field); and in 16 bits (x << s) | q, q known zero from bit s, is bfi(q, x,
/// s, 16 - s), q's mask of its low s bits dropped. (32-bit shifted fields are
/// the combine below; the 16-bit low half of a 32-bit value is its register
/// half, no bfi.)
static SDValue performBfiInsertCombine(SDNode *N,
                                       TargetLowering::DAGCombinerInfo &DCI) {
  EVT VT = N->getValueType(0);
  if (!DCI.isAfterLegalizeDAG() || (VT != MVT::i32 && VT != MVT::i16))
    return SDValue();
  SelectionDAG &DAG = DCI.DAG;
  unsigned BW = VT.getSizeInBits();
  SDLoc DL(N);
  auto MaskOf = [](SDValue V) -> std::optional<APInt> {
    if (V.getOpcode() != ISD::AND || !isa<ConstantSDNode>(V.getOperand(1)))
      return std::nullopt;
    return V.getConstantOperandAPInt(1);
  };
  for (unsigned I = 0; I < 2; ++I) {
    SDValue P = N->getOperand(I), Q = N->getOperand(1 - I);
    auto MP = MaskOf(P), MQ = MaskOf(Q);
    // (or x known zero above its low w bits, x >> s among them)
    if (MQ && (~*MQ).isMask() && Q.hasOneUse()) {
      unsigned W = (~*MQ).countr_one();
      bool Masked = MP && *MP == ~*MQ && P.hasOneUse();
      if (W % 8 == 0 && W < BW && !(BW == 32 && W == 16) &&
          (Masked || APInt::getBitsSetFrom(BW, W).isSubsetOf(
                         DAG.computeKnownBits(P).Zero)))
        return DAG.getNode(AGXISD::BFI_FIELD, DL, VT, Q.getOperand(0),
                           Masked ? P.getOperand(0) : P,
                           DAG.getConstant(0, DL, VT),
                           DAG.getTargetConstant(W, DL, MVT::i32));
    }
    if (BW != 16 || P.getOpcode() != ISD::SHL ||
        !isa<ConstantSDNode>(P.getOperand(1)) || !P.hasOneUse())
      continue;
    uint64_t S = P.getConstantOperandVal(1);
    if (S == 0 || S >= 16 || S % 8 != 0)
      continue;
    APInt High = APInt::getBitsSetFrom(16, S);
    if (!High.isSubsetOf(DAG.computeKnownBits(Q).Zero))
      continue;
    SDValue Base = Q;
    if (MQ && *MQ == ~High && Q.hasOneUse())
      Base = Q.getOperand(0);
    return DAG.getNode(AGXISD::BFI_FIELD, DL, VT, Base, P.getOperand(0),
                       DAG.getConstant(S, DL, VT),
                       DAG.getTargetConstant(16 - S, DL, MVT::i32));
  }
  return SDValue();
}

static cl::opt<bool> SelectBooleanLogic(
    "agx-select-boolean-logic",
    cl::desc("and / or of a comparison and a 0/1 value as one select"),
    cl::init(true), cl::Hidden);

/// and(setcc, b) and or(setcc, b) of a 0/1 value b: setcc ? b : 0 and
/// setcc ? 1 : b, one icmpsel / fcmpsel (the comparison's), instead of
/// materializing the comparison's 0/1 and then the and / or. The select
/// reads the compared pair and b: at most one issue cycle more than the
/// comparison alone (three 32-bit registers), which is what the and took.
static SDValue performBooleanLogicCombine(SDNode *N,
                                          TargetLowering::DAGCombinerInfo &DCI) {
  if (!SelectBooleanLogic || !DCI.isAfterLegalizeDAG() ||
      N->getValueType(0) != MVT::i32)
    return SDValue();
  SelectionDAG &DAG = DCI.DAG;
  for (unsigned I = 0; I < 2; ++I) {
    SDValue C = N->getOperand(I), B = N->getOperand(1 - I);
    if (C.getOpcode() != ISD::SETCC || !C.hasOneUse())
      continue;
    EVT CT = C.getOperand(0).getValueType();
    if (CT != MVT::i32 && CT != MVT::f32 && CT != MVT::i16 && CT != MVT::f16)
      continue;
    if (DAG.computeKnownBits(B).countMinLeadingZeros() < 31)
      continue;
    SDLoc DL(N);
    SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
    SDValue One = DAG.getConstant(1, DL, MVT::i32);
    return N->getOpcode() == ISD::AND
               ? DAG.getSelect(DL, MVT::i32, C, B, Zero)
               : DAG.getSelect(DL, MVT::i32, C, One, B);
  }
  return SDValue();
}

static cl::opt<bool> SharedByteBias(
    "agx-shared-byte-bias",
    cl::desc("signed bytes of one word through one xor of the word"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> Byte1Asr(
    "agx-byte1-asr",
    cl::desc("the signed byte 1 of a word as an asr of its low half"),
    cl::init(true), cl::Hidden);

/// Whether X is a word with two or more signed bytes taken from it (the
/// shl / sra pairs and sign_extend_inregs of the type legalized sext of its
/// <4 x i8>), or one already taken through x ^ 0x80808080.
static bool hasSharedSignedBytes(SDValue X) {
  unsigned Fields = 0;
  for (SDNode *U : X->users()) {
    if (U->getOpcode() == ISD::XOR && isa<ConstantSDNode>(U->getOperand(1)) &&
        U->getConstantOperandVal(1) == 0x80808080)
      return true;
    auto IsByteExt = [](const SDNode *E) {
      return E->getOpcode() == ISD::SIGN_EXTEND_INREG &&
             cast<VTSDNode>(E->getOperand(1))->getVT() == MVT::i8;
    };
    if (IsByteExt(U))
      ++Fields;
    else if (U->getOpcode() == ISD::SRL && isa<ConstantSDNode>(U->getOperand(1)) &&
             U->getConstantOperandVal(1) <= 24 &&
             llvm::any_of(U->users(), IsByteExt))
      ++Fields;
    else if (U->getOpcode() == ISD::SHL && isa<ConstantSDNode>(U->getOperand(1)) &&
             U->hasOneUse() && U->user_begin()->getOpcode() == ISD::SRA &&
             isa<ConstantSDNode>(U->user_begin()->getOperand(1)) &&
             U->user_begin()->getConstantOperandVal(1) == 24)
      ++Fields;
  }
  return Fields >= 2;
}

/// The signed byte at bit P of X: ((x >> p) & 255) ^ 128) - 128 on the ALUs,
/// or, where X gives several, ((x ^ 0x80808080) >> p & 255) - 128, the xor
/// shared by all of them (one per word instead of one per byte).
static SDValue signedByteALU(SelectionDAG &DAG, const SDLoc &DL, SDValue X,
                             unsigned P) {
  auto C = [&](uint64_t V) { return DAG.getConstant(V, DL, MVT::i32); };
  // Byte 1: one asr of the low half by 8 (asr sign-extends its 16-bit
  // source), the shift a bfeil would take anyway, without the subtraction.
  if (P == 8 && Byte1Asr)
    return DAG.getNode(ISD::SRA, DL, MVT::i32,
                       DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i32, X,
                                   DAG.getValueType(MVT::i16)),
                       C(8));
  bool Shared = SharedByteBias && hasSharedSignedBytes(X);
  // Opaque, so that demanded bits do not narrow each byte's xor to its own
  // bit (which would make one xor per byte again).
  SDValue F =
      Shared ? DAG.getNode(ISD::XOR, DL, MVT::i32, X,
                           DAG.getConstant(0x80808080, DL, MVT::i32,
                                           /*isTarget=*/false,
                                           /*isOpaque=*/true))
             : X;
  if (P)
    F = DAG.getNode(ISD::SRL, DL, MVT::i32, F, C(P));
  F = DAG.getNode(ISD::AND, DL, MVT::i32, F, C(255));
  if (!Shared)
    F = DAG.getNode(ISD::XOR, DL, MVT::i32, F, C(128));
  return DAG.getNode(ISD::SUB, DL, MVT::i32, F, C(128));
}

static cl::opt<bool> BranchLowBitXor(
    "agx-branch-low-bit-xor",
    cl::desc("a branch on (a ^ x) & 1, a boolean, as a != (x & 1)"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> FreezeTruthLogic(
    "agx-freeze-truth-logic",
    cl::desc("push a freeze of an and / or / xor / select of truth values "
             "into its operands"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> SelectOfSelect(
    "agx-select-of-select",
    cl::desc("a select on an or / and of truth values as two nested "
             "selects"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> XorShiftCommute(
    "agx-xor-shift-commute",
    cl::desc("(x ^ c) << s as (x << s) ^ (c << s)"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> MulLoHiPairs(
    "agx-mul-lohi-pairs", cl::init(true), cl::Hidden,
    cl::desc("a mulhu / mulhs and a mul of the same operands as one "
             "widening multiply"));

static cl::opt<bool> SignSelect(
    "agx-sign-select",
    cl::desc("select on x > -1 as a select on x < 0, arms swapped"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> WideByteLoads(
    "agx-wide-byte-loads", cl::init(true), cl::Hidden,
    cl::desc("A byte or 16-bit value loaded into 16 bits and zero-extended "
             "further is loaded into a 32-bit register"));

static cl::opt<bool> KeepShiftPairs(
    "agx-keep-shift-pairs", cl::init(true), cl::Hidden,
    cl::desc("Keep a 64-bit (x >> a) << b rather than a shift and a mask the "
             "8-bit immediate does not hold"));

/// A 64-bit (x >> a) << b is a shift and an and with ((-1 >> a) << b)
/// otherwise. Where the mask is beyond the 8-bit immediate, the and is a
/// mov_imm and a (32-bit) and, and the offset no longer a shifted index: the
/// shift left of an address offset is free, the device address's index shift
/// ((i / 32) * 4 bytes: one bfeil, the index register). (32-bit masks stay:
/// one mov_imm serves every and, and x - ((x >> a) << a) is x & (2^a - 1).)
bool AGXTargetLowering::shouldFoldConstantShiftPairToMask(
    const SDNode *N) const {
  if (!KeepShiftPairs || N->getOpcode() != ISD::SHL ||
      N->getValueType(0) != MVT::i64)
    return true;
  SDValue X = N->getOperand(0);
  auto *B = dyn_cast<ConstantSDNode>(N->getOperand(1));
  auto *A = X.getNumOperands() > 1
                ? dyn_cast<ConstantSDNode>(X.getOperand(1))
                : nullptr;
  if (X.getOpcode() != ISD::SRL || !A || !B)
    return true;
  unsigned W = N->getValueType(0).getScalarSizeInBits();
  if (A->getZExtValue() >= W || B->getZExtValue() >= W)
    return true;
  APInt M = APInt::getAllOnes(W).lshr(A->getZExtValue()).shl(B->getZExtValue());
  return M.isIntN(8);
}

static cl::opt<bool> MulRotates(
    "agx-mul-rotates", cl::init(true), cl::Hidden,
    cl::desc("Recover the rotate of a product that InstCombine split into a "
             "second multiply"));

/// rotl(x * c1, k) is (x * c1 << k) | (x * c1 >> (32 - k)), which InstCombine
/// turns into (x * (c1 << k)) | ((x * c1) >> (32 - k)): a second multiply (on
/// the unit integer multiplies share) where the rotate was one extr (a hash's
/// mix step). or(x * c2, (x * c1) >> r) with c2 = c1 * 2^(32 - r) mod 2^32 is
/// rotl(x * c1, 32 - r) exactly: x * c1 * 2^(32 - r) mod 2^32 is x * c2.
static SDValue performMulRotateCombine(SDNode *N,
                                       TargetLowering::DAGCombinerInfo &DCI) {
  if (!MulRotates || N->getValueType(0) != MVT::i32)
    return SDValue();
  for (unsigned I = 0; I < 2; ++I) {
    SDValue A = N->getOperand(I), B = N->getOperand(1 - I);
    if (A.getOpcode() != ISD::MUL || B.getOpcode() != ISD::SRL)
      continue;
    SDValue M = B.getOperand(0);
    auto *R = dyn_cast<ConstantSDNode>(B.getOperand(1));
    if (M.getOpcode() != ISD::MUL || !R || R->getZExtValue() == 0 ||
        R->getZExtValue() >= 32)
      continue;
    SDValue X = M.getOperand(0);
    auto *C1 = dyn_cast<ConstantSDNode>(M.getOperand(1));
    auto *C2 = dyn_cast<ConstantSDNode>(A.getOperand(1));
    if (!C1 || !C2 || A.getOperand(0) != X)
      continue;
    unsigned K = 32 - R->getZExtValue();
    if (C2->getAPIntValue() != C1->getAPIntValue().shl(K))
      continue;
    SDLoc DL(N);
    SelectionDAG &DAG = DCI.DAG;
    SDValue Rot = DAG.getNode(ISD::ROTL, DL, MVT::i32, M,
                              DAG.getConstant(K, DL, MVT::i32));
    // Other uses (x * c2) >> s, s >= k, are rot >> s (the rotate's low k
    // bits, x * c1's high ones, shifted out): the second multiply goes.
    SmallVector<SDNode *, 4> Shifts;
    for (SDNode *U : A->users())
      if (U != N && U->getOpcode() == ISD::SRL &&
          isa<ConstantSDNode>(U->getOperand(1)) &&
          U->getConstantOperandVal(1) >= K &&
          U->getConstantOperandVal(1) < 32)
        Shifts.push_back(U);
    for (SDNode *U : Shifts)
      DCI.CombineTo(U, DAG.getNode(ISD::SRL, SDLoc(U), MVT::i32, Rot,
                                   U->getOperand(1)));
    return Rot;
  }
  return SDValue();
}

static cl::opt<unsigned> AGXOffsetIndexMax(
    "agx-offset-index-max", cl::init(1), cl::Hidden,
    cl::desc("Most constant offsets from one address that each become an index"));

static cl::opt<bool> OffsetIndices(
    "agx-offset-indices", cl::init(true), cl::Hidden,
    cl::desc("A device address base + (zext(x) << s) + c with no other "
             "constant offset is base + (zext((x << s - t) + c >> t) << t)"));

/// The element size, log2, of a device load or store whose address is A.
static std::optional<unsigned> deviceAccessElemLog2(SDNode *U, SDValue A) {
  auto *M = dyn_cast<MemSDNode>(U);
  if (!M || M->getAddressSpace() != AGX::DEVICE_ADDRESS || !M->isSimple())
    return std::nullopt;
  if (auto *L = dyn_cast<LoadSDNode>(U)) {
    if (L->getBasePtr() != A || L->isIndexed())
      return std::nullopt;
  } else if (auto *St = dyn_cast<StoreSDNode>(U)) {
    if (St->getBasePtr() != A || St->isIndexed())
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  uint64_t Bytes = M->getMemoryVT().getScalarSizeInBits() / 8;
  if (Bytes != 1 && Bytes != 2 && Bytes != 4 && Bytes != 8)
    return std::nullopt;
  return Log2_64(std::min<uint64_t>(Bytes, 4));
}

/// (b + (zext(x) << s)) + c: device accesses at b + (zext(x) << s) take b and
/// the index register x, but the one at + c (the second half of a split
/// vector, a field of a struct element) took the 64-bit sum b + (x << s) as
/// its base: a 64-bit add, and copies of a uniform b to registers. Where c is
/// one of at most three constant offsets from that sum, and (x << (s - t)) + (c >> t), t
/// the trailing zeros of c (at most s), is below 2^32 (known bits), it is
/// b + (zext of that 32-bit sum << t): one 32-bit add, the index register.
static SDValue performOffsetIndexCombine(SDNode *N,
                                         TargetLowering::DAGCombinerInfo &DCI) {
  if (!OffsetIndices || N->getValueType(0) != MVT::i64)
    return SDValue();
  auto *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  SDValue P = N->getOperand(0);
  if (!C || P.getOpcode() != ISD::ADD || C->isZero() ||
      C->getAPIntValue().isNegative())
    return SDValue();
  SelectionDAG &DAG = DCI.DAG;
  for (unsigned I = 0; I < 2; ++I) {
    SDValue B = P.getOperand(I), Sh = P.getOperand(1 - I);
    if (Sh.getOpcode() != ISD::SHL || !isa<ConstantSDNode>(Sh.getOperand(1)) ||
        Sh.getOperand(0).getOpcode() != ISD::ZERO_EXTEND ||
        Sh.getOperand(0).getOperand(0).getValueType() != MVT::i32)
      continue;
    unsigned S = Sh.getConstantOperandVal(1);
    uint64_t CV = C->getZExtValue();
    unsigned T = std::min<unsigned>(S, llvm::countr_zero(CV));
    SDValue X = Sh.getOperand(0).getOperand(0);
    // (x << (s - t)) + (c >> t) below 2^32.
    APInt Max = DAG.computeKnownBits(X).getMaxValue().zext(64);
    // A grid position: below the kernel's known grid extent.
    if (X.getOpcode() == ISD::INTRINSIC_WO_CHAIN)
      if (auto E = AGX::knownGridExtent(DAG.getMachineFunction().getFunction(),
                                        X.getConstantOperandVal(0)))
        Max = APIntOps::umin(Max, APInt(64, *E - 1));
    if (S - T >= 32 || (CV >> T) >= (uint64_t(1) << 32) ||
        Max.ugt(APInt::getAllOnes(64).lshr(32 + S - T)))
      return SDValue();
    if ((Max.shl(S - T) + (CV >> T)).uge(APInt(64, 1).shl(32)))
      return SDValue();
    // Every user of p: this add or an access at p; of this add: accesses,
    // each of an element size t indexes (t in [e, e + 2]).
    auto Fits = [&](SDValue A, SDNode *U) {
      std::optional<unsigned> E = deviceAccessElemLog2(U, A);
      return E && T >= *E && T <= *E + 2;
    };
    bool OK = !N->use_empty();
    for (SDNode *U : N->users())
      OK &= Fits(SDValue(N, 0), U);
    // p's other users: accesses at p, or at most two more constant offsets
    // (each one 32-bit add, against the 64-bit sum and, for a uniform b, its
    // two copies).
    unsigned Offsets = 1;
    for (SDNode *U : P->users()) {
      if (U == N || deviceAccessElemLog2(U, P))
        continue;
      if (U->getOpcode() == ISD::ADD && U->getOperand(0) == P &&
          isa<ConstantSDNode>(U->getOperand(1)) && ++Offsets <= AGXOffsetIndexMax)
        continue;
      OK = false;
    }
    if (!OK)
      return SDValue();
    SDLoc DL(N);
    SDValue Y = DAG.getNode(
        ISD::ADD, DL, MVT::i32,
        DAG.getNode(ISD::SHL, DL, MVT::i32, X,
                    DAG.getConstant(S - T, DL, MVT::i32)),
        DAG.getConstant(CV >> T, DL, MVT::i32), SDNodeFlags::NoUnsignedWrap);
    return DAG.getNode(
        ISD::ADD, DL, MVT::i64, B,
        DAG.getNode(ISD::SHL, DL, MVT::i64,
                    DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, Y),
                    DAG.getConstant(T, DL, MVT::i32)));
  }
  return SDValue();
}

SDValue AGXTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case ISD::ADD:
    return performOffsetIndexCombine(N, DCI);
  case ISD::SHL: {
    // (x ^ c) << s -> (x << s) ^ (c << s) for c << s an 8-bit immediate: the
    // shifted x is the one other indices use (tg[t] beside tg[t ^ 1]), so
    // one xor instead of an xor and a shift (add and or fold into the
    // shifted iadd already).
    SDValue X = N->getOperand(0);
    auto *S = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (!XorShiftCommute || N->getValueType(0) != MVT::i32 || !S ||
        X.getOpcode() != ISD::XOR || !X.hasOneUse() ||
        !isa<ConstantSDNode>(X.getOperand(1)))
      return SDValue();
    uint64_t Sh = S->getZExtValue(), C = X.getConstantOperandVal(1);
    if (Sh >= 32 || (C << Sh) > 255)
      return SDValue();
    SelectionDAG &DAG = DCI.DAG;
    SDLoc DL(N);
    return DAG.getNode(ISD::XOR, DL, MVT::i32,
                       DAG.getNode(ISD::SHL, DL, MVT::i32, X.getOperand(0),
                                   N->getOperand(1)),
                       DAG.getConstant(C << Sh, DL, MVT::i32));
  }
  case ISD::SETCC:
    return performClassCompareCombine(N, DCI);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT_SAT:
  case ISD::FP_TO_UINT_SAT:
    return performRoundClampConvertCombine(N, DCI);
  case ISD::SELECT: {
    SDValue C = N->getOperand(0);
    // select (select (c, 1, d)), x, y -> select c, x, (select d, x, y) and
    // select (select (c, d, 0)), x, y -> select c, (select d, x, y), y: a
    // condition that is a truth value or another (the or / and of two
    // compares after legalization, as one / ueq float compares expand to)
    // takes two compare-selects instead of the inner select's truth value
    // and a select on it.
    if (SelectOfSelect && DCI.isAfterLegalizeDAG() &&
        C.getOpcode() == ISD::SELECT && C.hasOneUse()) {
      SelectionDAG &DAG = DCI.DAG;
      SDLoc DL(N);
      SDValue X = N->getOperand(1), Y = N->getOperand(2);
      SDValue CC = C.getOperand(0), T = C.getOperand(1), F = C.getOperand(2);
      if (isOneConstant(T) && F.getOpcode() == ISD::SETCC)
        return DAG.getSelect(DL, N->getValueType(0), CC, X,
                             DAG.getSelect(DL, N->getValueType(0), F, X, Y));
      if (isNullConstant(F) && T.getOpcode() == ISD::SETCC)
        return DAG.getSelect(DL, N->getValueType(0), CC,
                             DAG.getSelect(DL, N->getValueType(0), T, X, Y),
                             Y);
    }
    // select (x > -1), a, b -> select (x < 0), b, a: icmpsel compares with
    // an immediate 0 but needs -1 in a register (a mov_imm). x >= 0 is
    // canonicalized to x > -1.
    if (!SignSelect || !DCI.isAfterLegalizeDAG() ||
        C.getOpcode() != ISD::SETCC || !C.hasOneUse() ||
        !C.getOperand(0).getValueType().isScalarInteger() ||
        !isAllOnesConstant(C.getOperand(1)) ||
        cast<CondCodeSDNode>(C.getOperand(2))->get() != ISD::SETGT)
      return SDValue();
    SelectionDAG &DAG = DCI.DAG;
    SDLoc DL(N);
    SDValue Lt = DAG.getSetCC(
        DL, C.getValueType(), C.getOperand(0),
        DAG.getConstant(0, DL, C.getOperand(0).getValueType()), ISD::SETLT);
    return DAG.getSelect(DL, N->getValueType(0), Lt, N->getOperand(2),
                         N->getOperand(1));
  }
  case ISD::FREEZE: {
    // freeze (and a, b) -> and (freeze a), (freeze b), likewise or, xor and
    // select, for i1: the generic combine pushes a freeze only into an
    // operation with one operand that may be poison (or a compare), and
    // `select c, true, d` -> `or c, freeze d` leaves a frozen and of two
    // compares, whose known bits (0 or 1) are lost. Each operand is frozen
    // here, without replacing its other uses; the compares then take the
    // freezes into their own operands.
    SDValue V = N->getOperand(0);
    if (!FreezeTruthLogic || V.getValueType() != MVT::i1 || !V.hasOneUse())
      return SDValue();
    unsigned Opc = V.getOpcode();
    if (Opc != ISD::AND && Opc != ISD::OR && Opc != ISD::XOR &&
        Opc != ISD::SELECT)
      return SDValue();
    SelectionDAG &DAG = DCI.DAG;
    SmallVector<SDValue, 3> Ops;
    for (SDValue Op : V->op_values())
      Ops.push_back(DAG.getFreeze(Op));
    return DAG.getNode(Opc, SDLoc(N), MVT::i1, Ops);
  }
  case ISD::BRCOND:
    return performBRCONDCombine(N, DCI);
  case ISD::AND:
    return performBooleanLogicCombine(N, DCI);
  case ISD::OR: {
    if (SDValue V = performMulRotateCombine(N, DCI))
      return V;
    if (SDValue V = performBooleanLogicCombine(N, DCI))
      return V;
    if (SDValue V = performBfiInsertCombine(N, DCI))
      return V;
    // or(Q, P) where P is x << s (with masks around the shift) whose bits
    // that may be set are one field [s, s + w) and Q is known zero there:
    // one bfi of x into Q, instead of the shift, the masks and the or.
    if (!DCI.isAfterLegalizeDAG() || N->getValueType(0) != MVT::i32)
      return SDValue();
    SelectionDAG &DAG = DCI.DAG;
    for (unsigned I = 0; I < 2; ++I) {
      SDValue P = N->getOperand(I), Q = N->getOperand(1 - I);
      SDValue Sh = P;
      if (Sh.getOpcode() == ISD::AND && isa<ConstantSDNode>(Sh.getOperand(1)))
        Sh = Sh.getOperand(0);
      if (Sh.getOpcode() != ISD::SHL ||
          !isa<ConstantSDNode>(Sh.getOperand(1)) || !P.hasOneUse())
        continue;
      uint64_t S = Sh.getConstantOperandVal(1);
      if (S == 0 || S >= 32)
        continue;
      SDValue X = Sh.getOperand(0);
      if (X.getOpcode() == ISD::AND && isa<ConstantSDNode>(X.getOperand(1)))
        X = X.getOperand(0);
      // (y >> (32 - s)) | (x << s), y perhaps the low word of a 64-bit
      // shift, is a funnel shift: one extr
      SDValue QS = Q;
      while (QS.getOpcode() == ISD::TRUNCATE || QS.getOpcode() == ISD::AND ||
             QS.getOpcode() == ISD::BUILD_PAIR)
        QS = QS.getOperand(0);
      if ((QS.getOpcode() == ISD::SRL || QS.getOpcode() == ISD::SRA) &&
          isa<ConstantSDNode>(QS.getOperand(1)) &&
          QS.getConstantOperandVal(1) % 32 == 32 - S)
        continue;
      APInt Field = ~DAG.computeKnownBits(P).Zero;
      if (Field.isZero() || !Field.isShiftedMask() || Field.countr_zero() != S)
        continue;
      if (!Field.isSubsetOf(DAG.computeKnownBits(Q).Zero))
        continue;
      // bytes only (packing 8-bit values): the fields of soft-f64 and the
      // like are left to the shifts, ors and funnel shifts they are now
      // (or any field where q is masked by exactly its complement, the
      // insert idiom of PTX bfi)
      bool Insert = Q.getOpcode() == ISD::AND &&
                    isa<ConstantSDNode>(Q.getOperand(1)) &&
                    Q.getConstantOperandAPInt(1) == ~Field;
      if (!Insert && (S % 8 != 0 || Field.popcount() % 8 != 0))
        continue;
      if (Insert)
        Q = Q.getOperand(0);
      SDLoc DL(N);
      // A 16-bit value under another: the register's halves. The patterns
      // take the common forms; here an assembled low part (an insert or an
      // or of fields), never a shifted one (a funnel shift's).
      if (S == 16) {
        // (or a 16-bit value zero-extended, an assembled half)
        bool Half16 = Q.getOpcode() == ISD::ZERO_EXTEND &&
                      Q.getOperand(0).getValueType() == MVT::i16;
        if (Field.popcount() != 16 ||
            (Q.getOpcode() != AGXISD::BFI_FIELD && Q.getOpcode() != ISD::OR &&
             !Half16))
          continue;
        if (Half16)
          Q = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Q.getOperand(0));
        return DAG.getNode(AGXISD::PACK16, DL, MVT::i32, Q, X);
      }
      return DAG.getNode(AGXISD::BFI_FIELD, DL, MVT::i32, Q, X,
                         DAG.getConstant(S, DL, MVT::i32),
                         DAG.getTargetConstant(Field.popcount(), DL, MVT::i32));
    }
    return SDValue();
  }
  case ISD::SIGN_EXTEND_INREG: {
    if (!AGXSignedBytesALU)
      return SDValue();
    // The low byte signed: ((x & 255) ^ 128) - 128, as below. (16 bits are
    // one addition with a sign-extended 16-bit source.)
    if (!DCI.isAfterLegalizeDAG() || N->getValueType(0) != MVT::i32 ||
        cast<VTSDNode>(N->getOperand(1))->getVT() != MVT::i8)
      return SDValue();
    SDValue X = N->getOperand(0);
    unsigned P = 0;
    if (X.getOpcode() == ISD::SRL && isa<ConstantSDNode>(X.getOperand(1)) &&
        X.getConstantOperandVal(1) <= 24 && SharedByteBias &&
        hasSharedSignedBytes(X.getOperand(0))) {
      P = X.getConstantOperandVal(1);
      X = X.getOperand(0);
    }
    return signedByteALU(DCI.DAG, SDLoc(N), X, P);
  }
  case AGXISD::STORE_V2_I8:
  case AGXISD::STORE_V4_I8: {
    // Only each value's low byte is stored: a sign or zero extension of it
    // (a saturating 8-bit conversion's) goes.
    unsigned NumElts = N->getOpcode() == AGXISD::STORE_V2_I8 ? 2 : 4;
    SmallVector<SDValue, 6> Ops(N->op_begin(), N->op_end());
    bool Changed = false;
    APInt Low = APInt::getLowBitsSet(16, 8);
    for (unsigned I = 1; I <= NumElts; ++I)
      if (SDValue V = SimplifyMultipleUseDemandedBits(Ops[I], Low, DCI.DAG)) {
        Ops[I] = V;
        Changed = true;
      }
    if (!Changed)
      return SDValue();
    return SDValue(DCI.DAG.UpdateNodeOperands(N, Ops), 0);
  }
  case ISD::MULHU:
  case ISD::MULHS: {
    // A multiply-high whose low half is also computed (mul of the same
    // operands, as Philox's mul.hi / mul.lo pairs): one 32 x 32 -> 64-bit
    // multiply for both.
    if (!MulLoHiPairs || N->getValueType(0) != MVT::i32)
      return SDValue();
    SelectionDAG &DAG = DCI.DAG;
    SDValue A = N->getOperand(0), B = N->getOperand(1);
    SDNode *Lo = DAG.getNodeIfExists(ISD::MUL, DAG.getVTList(MVT::i32), {A, B},
                                     /*AllowCommute=*/true);
    if (!Lo)
      return SDValue();
    SDValue LoHi = DAG.getNode(N->getOpcode() == ISD::MULHU ? ISD::UMUL_LOHI
                                                            : ISD::SMUL_LOHI,
                               SDLoc(N), DAG.getVTList(MVT::i32, MVT::i32), A,
                               B);
    DCI.CombineTo(Lo, LoHi.getValue(0));
    return LoHi.getValue(1);
  }
  case ISD::SRA: {
    if (AGXSignedBytesALU < 2 ||
        (AGXSignedBytesALU == 3 &&
         DCI.DAG.getMachineFunction()
                 .getInfo<AGXMachineFunctionInfo>()
                 ->getEstimatedPressure(
                     DCI.DAG.getMachineFunction().getFunction()) >
             AGXHeadroomGPRs))
      return SDValue();
    // A signed byte, sra(shl(x, 24 - p), 24): the field (one extract, or an
    // and at p = 0) and (f ^ 128) - 128 on the ALUs, instead of two shifts on
    // the unit that integer multiplies, bit fields and conversions share
    // (the sign-extended bytes of dp4a). A 16-bit field: its shift and a
    // sign-extending addition, one shift fewer.
    SDValue A = N->getOperand(0);
    auto *S2 = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (!DCI.isAfterLegalizeDAG() || N->getValueType(0) != MVT::i32 || !S2 ||
        A.getOpcode() != ISD::SHL || !A.hasOneUse())
      return SDValue();
    auto *S1 = dyn_cast<ConstantSDNode>(A.getOperand(1));
    if (!S1)
      return SDValue();
    uint64_t Sh1 = S1->getZExtValue(), Sh2 = S2->getZExtValue();
    unsigned W = 32 - Sh2;
    if (Sh2 >= 32 || Sh1 > Sh2 || (W != 8 && W != 16))
      return SDValue();
    SDLoc DL(N);
    SelectionDAG &DAG = DCI.DAG;
    auto C = [&](uint64_t V) { return DAG.getConstant(V, DL, MVT::i32); };
    SDValue X = A.getOperand(0), F = X;
    if (unsigned P = Sh2 - Sh1)
      F = DAG.getNode(ISD::SRL, DL, MVT::i32, X, C(P));
    if (W == 16)
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i32, F,
                         DAG.getValueType(MVT::i16));
    if (W == 8)
      return signedByteALU(DAG, DL, X, Sh2 - Sh1);
    F = DAG.getNode(ISD::AND, DL, MVT::i32, F, C((1u << W) - 1));
    uint64_t Sign = 1u << (W - 1);
    return DAG.getNode(ISD::SUB, DL, MVT::i32,
                       DAG.getNode(ISD::XOR, DL, MVT::i32, F, C(Sign)),
                       C(Sign));
  }
  case ISD::FMINIMUMNUM: {
    // minimumnum(maximumnum(x, +0), 1): the clamp of .sat (a NaN becomes +0
    // and -0 becomes +0, as maximumnum(-0, +0) is +0), before the two become
    // hardware min / max and their signed-zero fixes. Under ieee-fp32 a float
    // must keep its denormals, which the hardware's .sat flushes.
    EVT VT = N->getValueType(0);
    if (DCI.isAfterLegalizeDAG() || (VT != MVT::f16 && VT != MVT::f32) ||
        (VT == MVT::f32 && DCI.DAG.getSubtarget<AGXSubtarget>().hasIEEEFP32()))
      return SDValue();
    auto IsConst = [](SDValue V, double C) {
      auto *F = dyn_cast<ConstantFPSDNode>(V);
      return F && F->getValueAPF().isPosZero() == (C == 0.0) &&
             F->isExactlyValue(C);
    };
    SDValue M = N->getOperand(0), One = N->getOperand(1);
    if (!IsConst(One, 1.0))
      std::swap(M, One);
    if (!IsConst(One, 1.0) || M.getOpcode() != ISD::FMAXIMUMNUM ||
        !M.hasOneUse())
      return SDValue();
    SDValue X = M.getOperand(0), Zero = M.getOperand(1);
    if (!IsConst(Zero, 0.0))
      std::swap(X, Zero);
    if (!IsConst(Zero, 0.0))
      return SDValue();
    return DCI.DAG.getNode(AGXISD::FSAT, SDLoc(N), VT, X);
  }
  case ISD::ZERO_EXTEND: {
    SDValue A = N->getOperand(0);
    // A byte or 16-bit value loaded into 16 bits (an i64 zero-extending load
    // of either is expanded so) and zero-extended further: loaded into the
    // 32-bit register, zero-extended, instead of a 16-bit half and a move.
    if (auto *L = dyn_cast<LoadSDNode>(A);
        WideByteLoads && L && A.getValueType() == MVT::i16 &&
        (N->getValueType(0) == MVT::i32 || N->getValueType(0) == MVT::i64) &&
        ((L->getExtensionType() == ISD::ZEXTLOAD &&
          L->getMemoryVT() == MVT::i8) ||
         (L->getExtensionType() == ISD::NON_EXTLOAD &&
          L->getMemoryVT() == MVT::i16)) &&
        !L->isIndexed() && L->isSimple() && A.hasOneUse()) {
      // (Device, threadgroup and stack loads of a byte or 16 bits into 32
      // bits all have patterns.)
      SelectionDAG &DAG = DCI.DAG;
      SDLoc DL(N);
      SDValue W =
          DAG.getExtLoad(ISD::ZEXTLOAD, DL, MVT::i32, L->getChain(),
                         L->getBasePtr(), L->getMemoryVT(), L->getMemOperand());
      DAG.ReplaceAllUsesOfValueWith(SDValue(L, 1), W.getValue(1));
      return N->getValueType(0) == MVT::i32
                 ? W
                 : DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, W);
    }
    // Booleans (i1 promoted to i16) combined with and / or / xor and then
    // extended to 32 bits (for a branch): compute them in 32 bits, where
    // compares produce them directly, instead of a 16-bit chain and a move.
    if (N->getValueType(0) != MVT::i32 || A.getValueType() != MVT::i16 ||
        !isBoolean16(A, DCI.DAG, 0))
      return SDValue();
    return widenBoolean(A, DCI.DAG, SDLoc(N));
  }
  case ISD::SRL: {
    SDValue A = N->getOperand(0);
    auto *S = dyn_cast<ConstantSDNode>(N->getOperand(1));
    // A 16-bit value zero-extended, then shifted: shifted in its 16-bit
    // register half (no copy into a 32-bit register first).
    if (S && N->getValueType(0) == MVT::i32 &&
        A.getOpcode() == ISD::ZERO_EXTEND &&
        A.getOperand(0).getValueType() == MVT::i16 && S->getZExtValue() < 16) {
      SDLoc DL(N);
      SDValue Narrow =
          DCI.DAG.getNode(ISD::SRL, DL, MVT::i16, A.getOperand(0),
                          DCI.DAG.getConstant(S->getZExtValue(), DL, MVT::i16));
      return DCI.DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, Narrow);
    }
    // (x & (m << s)) >> s = (x >> s) & m: one bfeil (a field extract).
    auto *M = A.getOpcode() == ISD::AND
                  ? dyn_cast<ConstantSDNode>(A.getOperand(1))
                  : nullptr;
    if (!DCI.isAfterLegalizeDAG() || !S || !M || !A.hasOneUse())
      return SDValue();
    uint64_t Mask = M->getZExtValue(), Shift = S->getZExtValue();
    if (!isShiftedMask_64(Mask) || llvm::countr_zero(Mask) != Shift)
      return SDValue();
    SDLoc DL(N);
    EVT VT = N->getValueType(0);
    SDValue Shifted = DCI.DAG.getNode(ISD::SRL, DL, VT, A.getOperand(0),
                                      N->getOperand(1));
    return DCI.DAG.getNode(ISD::AND, DL, VT, Shifted,
                           DCI.DAG.getConstant(Mask >> Shift, DL, VT));
  }
  case ISD::TRUNCATE:
    // The low half of a pair.
    if (N->getOperand(0).getOpcode() == ISD::BUILD_PAIR) {
      SDValue Lo = N->getOperand(0).getOperand(0);
      return DCI.DAG.getZExtOrTrunc(Lo, SDLoc(N), N->getValueType(0));
    }
    // The low half of a 64-bit sign extension in register from at most the
    // result's width: the extension of the low half (sext i16 -> i64 masked
    // to 32 bits, as zext (sext i16 -> i32) becomes, is then one extension
    // instead of a 64-bit one and its truncation).
    if (N->getOperand(0).getOpcode() == ISD::SIGN_EXTEND_INREG &&
        N->getOperand(0).hasOneUse()) {
      SDValue In = N->getOperand(0);
      EVT VT = N->getValueType(0);
      EVT From = cast<VTSDNode>(In.getOperand(1))->getVT();
      if (VT.isScalarInteger() && From.bitsLT(VT)) {
        SDLoc DL(N);
        return DCI.DAG.getNode(
            ISD::SIGN_EXTEND_INREG, DL, VT,
            DCI.DAG.getNode(ISD::TRUNCATE, DL, VT, In.getOperand(0)),
            In.getOperand(1));
      }
    }
    return SDValue();
  case ISD::BUILD_PAIR: {
    // A value split and joined again.
    SDValue Lo = N->getOperand(0), Hi = N->getOperand(1);
    if (Lo.getOpcode() == ISD::TRUNCATE && Hi.getOpcode() == AGXISD::HI32 &&
        Lo.getOperand(0) == Hi.getOperand(0))
      return Lo.getOperand(0);
    return SDValue();
  }
  case AGXISD::IF:
  case AGXISD::LOOP: {
    // On (b ^ 1) for a boolean b: on b == 0, which the branch compares
    // itself (AGXFoldOperands), instead of an xor and a compare with zero.
    // Likewise on a ^ b, a != b, and on (a ^ b) == 0 or != 0, a == b or
    // a != b: one compare of the two instead of an xor and a compare.
    SDValue Cond = N->getOperand(1);
    SelectionDAG &DAG = DCI.DAG;
    SDLoc DL(N);
    auto Branch = [&](SDValue C) {
      return DAG.getNode(N->getOpcode(), DL, MVT::Other, N->getOperand(0), C,
                         N->getOperand(2));
    };
    auto IsBool = [&](SDValue V) {
      return DAG.computeKnownBits(V).countMaxActiveBits() <= 1;
    };
    if (Cond.getOpcode() == ISD::XOR && Cond.hasOneUse()) {
      SDValue A = Cond.getOperand(0), B = Cond.getOperand(1);
      if (isOneConstant(B) && IsBool(A))
        return Branch(DAG.getSetCC(DL, MVT::i32, A,
                                   DAG.getConstant(0, DL, MVT::i32),
                                   ISD::SETEQ));
      if (!isa<ConstantSDNode>(B) && IsBool(A) && IsBool(B))
        return Branch(DAG.getSetCC(DL, MVT::i32, A, B, ISD::SETNE));
      return SDValue();
    }
    // On (a ^ x) & 1 for a boolean a: on a != (x & 1), a compare the
    // branch takes, and x & 1 often computed already (bitonic's direction
    // bit, i1 trunc of the thread index), instead of an xor, an and and a
    // compare with zero.
    if (BranchLowBitXor && Cond.getOpcode() == ISD::AND && Cond.hasOneUse() &&
        isOneConstant(Cond.getOperand(1)) &&
        Cond.getOperand(0).getOpcode() == ISD::XOR &&
        Cond.getOperand(0).hasOneUse() && Cond.getValueType() == MVT::i32) {
      SDValue X = Cond.getOperand(0);
      for (unsigned I = 0; I < 2; ++I) {
        SDValue A = X.getOperand(I), B = X.getOperand(1 - I);
        if (!IsBool(A) || isa<ConstantSDNode>(B))
          continue;
        return Branch(DAG.getSetCC(
            DL, MVT::i32, A,
            DAG.getNode(ISD::AND, DL, MVT::i32, B,
                        DAG.getConstant(1, DL, MVT::i32)),
            ISD::SETNE));
      }
    }
    if (Cond.getOpcode() == ISD::SETCC && Cond.hasOneUse() &&
        isNullConstant(Cond.getOperand(1)) &&
        Cond.getOperand(0).getOpcode() == ISD::XOR &&
        Cond.getOperand(0).hasOneUse()) {
      ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
      SDValue X = Cond.getOperand(0);
      if ((CC == ISD::SETEQ || CC == ISD::SETNE) &&
          X.getValueType() == MVT::i32)
        return Branch(DAG.getSetCC(DL, Cond.getValueType(), X.getOperand(0),
                                   X.getOperand(1), CC));
    }
    return SDValue();
  }
  case AGXISD::BFI7:
  case AGXISD::EXTR7:
  case AGXISD::SRA7:
    // Shifts of zero.
    if (all_of(drop_end(N->ops()), [](const SDUse &U) { return isNullConstant(U.get()); }))
      return DCI.DAG.getConstant(0, SDLoc(N), MVT::i32);
    return SDValue();
  case AGXISD::HI32: {
    SDValue V = N->getOperand(0);
    SDLoc DL(N);
    SDValue H = hi32(DCI.DAG, DL, V);
    if (H.getOpcode() != AGXISD::HI32)
      return H;
    KnownBits Known = DCI.DAG.computeKnownBits(SDValue(N, 0));
    if (Known.isConstant())
      return DCI.DAG.getConstant(Known.getConstant(), DL, MVT::i32);
    return SDValue();
  }
  default:
    return SDValue();
  }
}

void AGXTargetLowering::computeKnownBitsForTargetNode(
    const SDValue Op, KnownBits &Known, const APInt &DemandedElts,
    const SelectionDAG &DAG, unsigned Depth) const {
  Known.resetAll();
  switch (Op.getOpcode()) {
  case AGXISD::BFI_FIELD: {
    // the base outside the field, the inserted bits in it
    unsigned S = Op.getConstantOperandVal(2), W = Op.getConstantOperandVal(3);
    unsigned BW = Op.getValueSizeInBits();
    APInt Field = APInt::getBitsSet(BW, S, std::min(S + W, BW));
    KnownBits B = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    KnownBits X = DAG.computeKnownBits(Op.getOperand(1), Depth + 1);
    X = KnownBits::shl(X, KnownBits::makeConstant(APInt(BW, S)));
    Known.Zero = (B.Zero & ~Field) | (X.Zero & Field);
    Known.One = (B.One & ~Field) | (X.One & Field);
    break;
  }
  case AGXISD::HI32:
    Known = DAG.computeKnownBits(Op.getOperand(0), Depth + 1).extractBits(32, 32);
    break;
  case AGXISD::LOAD_V2_I8:
  case AGXISD::LOAD_V4_I8: {
    // each byte zero-extended into its 16-bit half (bytes in memory; 16-bit
    // elements fill their halves)
    auto *M = cast<MemSDNode>(Op.getNode());
    if (Op.getResNo() < Op.getNode()->getNumValues() - 1 &&
        M->getMemoryVT().getScalarSizeInBits() == 8)
      Known.Zero.setBitsFrom(8);
    break;
  }
  case AGXISD::MUL_WIDE_U:
  case AGXISD::MUL_WIDE_S: {
    bool S = Op.getOpcode() == AGXISD::MUL_WIDE_S;
    KnownBits A = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    KnownBits B = DAG.computeKnownBits(Op.getOperand(1), Depth + 1);
    Known = KnownBits::mul(S ? A.sext(64) : A.zext(64), S ? B.sext(64) : B.zext(64));
    break;
  }
  default:
    break;
  }
}

static ISD::CondCode getUnsignedCondCode(ISD::CondCode CC) {
  switch (CC) {
  case ISD::SETLT:
    return ISD::SETULT;
  case ISD::SETLE:
    return ISD::SETULE;
  case ISD::SETGT:
    return ISD::SETUGT;
  case ISD::SETGE:
    return ISD::SETUGE;
  default:
    return CC;
  }
}

static cl::opt<bool> AGXHighHalfCompares(
    "agx-high-half-compares", cl::init(true), cl::Hidden,
    cl::desc("Compare only the high halves of a 64-bit value and a constant "
             "whose low half decides nothing"));

SDValue AGXTargetLowering::lowerSETCC64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  auto [ALo, AHi] = split64(DAG, DL, Op.getOperand(0));
  auto [BLo, BHi] = split64(DAG, DL, Op.getOperand(1));

  if (CC == ISD::SETEQ || CC == ISD::SETNE) {
    SDValue Lo = DAG.getSetCC(DL, VT, ALo, BLo, CC);
    SDValue Hi = DAG.getSetCC(DL, VT, AHi, BHi, CC);
    return DAG.getNode(CC == ISD::SETEQ ? ISD::AND : ISD::OR, DL, VT, Lo, Hi);
  }

  // The high halves decide unless they are equal; the low halves always
  // compare unsigned.
  SDValue HiCmp = DAG.getSetCC(DL, VT, AHi, BHi, CC);
  // A constant whose low half fixes the low comparison to what CC gives for
  // equal values (x < c, c's low half 0: never; x <= c, all ones: always;
  // > and >= alike, signed or not; x < 0 and x > -1 the sign tests): the high
  // halves alone.
  if (AGXHighHalfCompares)
    if (auto *C = dyn_cast<ConstantSDNode>(BLo)) {
      uint64_t L = C->getZExtValue() & 0xffffffffu;
      ISD::CondCode U = getUnsignedCondCode(CC);
      if (((U == ISD::SETULT || U == ISD::SETUGE) && L == 0) ||
          ((U == ISD::SETUGT || U == ISD::SETULE) && L == 0xffffffffu))
        return HiCmp;
    }
  SDValue HiEq = DAG.getSetCC(DL, VT, AHi, BHi, ISD::SETEQ);
  SDValue LoCmp = DAG.getSetCC(DL, VT, ALo, BLo, getUnsignedCondCode(CC));
  return DAG.getSelect(DL, VT, HiEq, LoCmp, HiCmp);
}

SDValue AGXTargetLowering::lowerShift64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  unsigned Opc = Op.getOpcode();
  auto [Lo, Hi] = split64(DAG, DL, Op.getOperand(0));
  SDValue Amt = DAG.getZExtOrTrunc(Op.getOperand(1), DL, MVT::i32);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  auto C32 = [&](uint64_t V) { return DAG.getConstant(V, DL, MVT::i32); };

  if (auto *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
    uint64_t S = C->getZExtValue();
    // Small left shifts stay for address mode matching (iadd with a shift);
    // instruction selection expands what is left.
    SDValue X = Op.getOperand(0);
    bool Address =
        X.getOpcode() == ISD::ZERO_EXTEND || X.getOpcode() == ISD::SIGN_EXTEND ||
        all_of(Op->users(), [](SDNode *U) { return U->getOpcode() == ISD::ADD; });
    if (S >= 64 || (Opc == ISD::SHL && S <= 4 && Address))
      return Op;
    if (S == 0)
      return X;
    SDValue NewLo, NewHi;
    if (S >= 32) {
      if (Opc == ISD::SHL) {
        NewLo = Zero;
        NewHi = DAG.getNode(ISD::SHL, DL, MVT::i32, Lo, C32(S - 32));
      } else {
        NewLo = DAG.getNode(Opc, DL, MVT::i32, Hi, C32(S - 32));
        NewHi = Opc == ISD::SRA ? DAG.getNode(ISD::SRA, DL, MVT::i32, Hi, C32(31))
                                : Zero;
      }
    } else if (Opc == ISD::SHL) {
      NewLo = DAG.getNode(ISD::SHL, DL, MVT::i32, Lo, C32(S));
      NewHi = DAG.getNode(ISD::FSHR, DL, MVT::i32, Hi, Lo, C32(32 - S));
    } else {
      NewLo = DAG.getNode(ISD::FSHR, DL, MVT::i32, Hi, Lo, C32(S));
      NewHi = DAG.getNode(Opc, DL, MVT::i32, Hi, C32(S));
    }
    return join64(DAG, DL, NewLo, NewHi);
  }

  // Variable shifts (by 0 to 63) with the 7-bit shift amounts of bfi, extr
  // and asr, which shift a register's bits out completely from 32 on:
  //   srl: lo = extr lo, hi, s            hi = extr hi, 0, s
  //   shl: lo = bfi 0, lo, s              hi = bfi (extr 0, lo, 64 - s), hi, s
  //   sra: lo = s < 32 ? extr lo, hi, s : asr hi, s - 32      hi = asr hi, s
  // In shl, extr 0, lo, 64 - s is the part of lo that moves into hi, for s
  // below and above 32 (and 0 for s = 0).
  SDValue NewLo, NewHi;
  if (Opc == ISD::SRL) {
    NewLo = DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Lo, Hi, Amt);
    NewHi = DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Hi, Zero, Amt);
  } else if (Opc == ISD::SHL) {
    NewLo = DAG.getNode(AGXISD::BFI7, DL, MVT::i32, Zero, Lo, Amt);
    SDValue Up = DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Zero, Lo,
                             DAG.getNode(ISD::SUB, DL, MVT::i32, C32(64), Amt));
    NewHi = DAG.getNode(AGXISD::BFI7, DL, MVT::i32, Up, Hi, Amt);
  } else {
    NewHi = DAG.getNode(AGXISD::SRA7, DL, MVT::i32, Hi, Amt);
    SDValue Big = DAG.getSetCC(DL, MVT::i32, Amt, C32(31), ISD::SETUGT);
    NewLo = DAG.getSelect(
        DL, MVT::i32, Big,
        DAG.getNode(AGXISD::SRA7, DL, MVT::i32, Hi,
                    DAG.getNode(ISD::SUB, DL, MVT::i32, Amt, C32(32))),
        DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Lo, Hi, Amt));
  }
  return join64(DAG, DL, NewLo, NewHi);
}

SDValue AGXTargetLowering::lowerMUL64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  auto [ALo, AHi] = split64(DAG, DL, A);
  auto [BLo, BHi] = split64(DAG, DL, B);
  // 32 x 32 -> 64-bit products are a single imadd.
  if (DAG.computeKnownBits(A).countMinLeadingZeros() >= 32 &&
      DAG.computeKnownBits(B).countMinLeadingZeros() >= 32)
    return DAG.getNode(AGXISD::MUL_WIDE_U, DL, MVT::i64, ALo, BLo);
  if (DAG.ComputeNumSignBits(A) > 32 && DAG.ComputeNumSignBits(B) > 32)
    return DAG.getNode(AGXISD::MUL_WIDE_S, DL, MVT::i64, ALo, BLo);
  // x * (y << k) = (x * y) << k (mod 2^64): an extended 32-bit value shifted
  // (a byte stride, shared by several products, so the DAG combiner's
  // one-use form does not apply) multiplies as the 32 x 32 -> 64-bit
  // product, the shift left for the address addition.
  for (unsigned I = 0; I < 2; ++I) {
    SDValue S = I ? B : A, O = I ? A : B;
    if (S.getOpcode() != ISD::SHL || !isa<ConstantSDNode>(S.getOperand(1)))
      continue;
    SDValue X = S.getOperand(0);
    bool Wide = (DAG.ComputeNumSignBits(X) > 32 && DAG.ComputeNumSignBits(O) > 32) ||
                (DAG.computeKnownBits(X).countMinLeadingZeros() >= 32 &&
                 DAG.computeKnownBits(O).countMinLeadingZeros() >= 32);
    if (!Wide || S.getConstantOperandVal(1) >= 64)
      continue;
    return DAG.getNode(ISD::SHL, DL, MVT::i64,
                       DAG.getNode(ISD::MUL, DL, MVT::i64, X, O),
                       S.getOperand(1));
  }

  // lo(a) * lo(b) as a 64-bit product, plus the cross products in the high
  // half.
  SDValue P = DAG.getNode(AGXISD::MUL_WIDE_U, DL, MVT::i64, ALo, BLo);
  auto [PLo, PHi] = split64(DAG, DL, P);
  SDValue Cross = DAG.getNode(ISD::ADD, DL, MVT::i32,
                              DAG.getNode(ISD::MUL, DL, MVT::i32, ALo, BHi),
                              DAG.getNode(ISD::MUL, DL, MVT::i32, AHi, BLo));
  return join64(DAG, DL, PLo, DAG.getNode(ISD::ADD, DL, MVT::i32, PHi, Cross));
}

static cl::opt<bool> SelectCountHalf(
    "agx-select-count-half",
    cl::desc("64-bit ctlz / cttz: select the half, then one ffs"),
    cl::init(true), cl::Hidden);

SDValue AGXTargetLowering::lowerCTLZ_CTTZ64(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  bool IsCTLZ = Op.getOpcode() == ISD::CTLZ;
  auto [Lo, Hi] = split64(DAG, DL, Op.getOperand(0));
  // Count in the half that decides; fall back to the other half plus 32.
  SDValue First = IsCTLZ ? Hi : Lo;
  SDValue Second = IsCTLZ ? Lo : Hi;
  unsigned Opc = Op.getOpcode();
  SDValue FirstZero =
      DAG.getSetCC(DL, MVT::i32, First, DAG.getConstant(0, DL, MVT::i32),
                   ISD::SETEQ);
  if (SelectCountHalf &&
      DAG.getMachineFunction()
              .getInfo<AGXMachineFunctionInfo>()
              ->getEstimatedPressure(DAG.getMachineFunction().getFunction()) <=
          AGXHeadroomGPRs) {
    // The half selected first and one ffs (the index of the highest set
    // bit, -1 for zero, as an M2 Max gives it): 31 - ffs(first), or
    // 63 - ffs(second) where the first half is zero (64 for x = 0), instead
    // of counting both halves. Both halves stay live up to the select, so
    // only where the function has register headroom (ddot, without it:
    // 64 -> 66 GPRs, 832 -> 768 threads per threadgroup).
    SDValue V = DAG.getSelect(DL, MVT::i32, FirstZero, Second, First);
    if (!IsCTLZ)
      V = DAG.getNode(ISD::BITREVERSE, DL, MVT::i32, V);
    SDValue Off = DAG.getSelect(DL, MVT::i32, FirstZero,
                                DAG.getConstant(63, DL, MVT::i32),
                                DAG.getConstant(31, DL, MVT::i32));
    SDValue F(DAG.getMachineNode(AGX::FFS_I32, DL, MVT::i32,
                                 {DAG.getTargetConstant(0, DL, MVT::i32), V}),
              0);
    return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64,
                       DAG.getNode(ISD::SUB, DL, MVT::i32, Off, F));
  }
  SDValue R = DAG.getSelect(
      DL, MVT::i32, FirstZero,
      DAG.getNode(ISD::ADD, DL, MVT::i32,
                  DAG.getNode(Opc, DL, MVT::i32, Second),
                  DAG.getConstant(32, DL, MVT::i32)),
      DAG.getNode(Opc, DL, MVT::i32, First));
  return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, R);
}

static SDValue getAGXIntrinsic(SelectionDAG &DAG, const SDLoc &DL,
                               Intrinsic::ID IID, EVT VT, SDValue Arg) {
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                     DAG.getTargetConstant(IID, DL, MVT::i32), Arg);
}

/// float division without arcp / afn, correctly rounded (with denormal
/// operands and results flushed, as the ALU does). The hardware's rcp is
/// within 1 ulp; the quotient of the significands, a1 in [1, 2) and d in
/// [2, 4) with the signs of a and b, takes the correctly rounded reciprocal
///   y = y0 + y0 (1 - d y0), y0 = rcp(d)
/// (correctly rounded for every significand with the recorded rcp table) and
/// one correction of q0 = a1 y:
///   q = q0 + (a1 - d q0) y,
/// correctly rounded by Markstein's theorem (y is the rounded 1 / d, q0
/// within 1 ulp, the remainder exact). The exponents come back as factors
///   pa = 2^ea, pb = 2^(128 - eb),
/// 0 / inf for a zero (or denormal) / infinite a, and inf / 0 for b, so that
/// zeros and infinities need no other test; the product is exact or rounds
/// once into the flushed or infinite range. pa first could flush a tiny a's
/// intermediate, pb first a huge b's, so a below 2^-124 takes pb first (both
/// at once flush the result anyway). NaNs are selected at the end.
static cl::opt<bool> ConstDivisorFold(
    "agx-fdiv-const-divisor", cl::init(true), cl::Hidden,
    cl::desc("a constant divisor's significand, reciprocal and exponent "
             "factor at compile time in the correctly rounded division"));

SDValue AGXTargetLowering::lowerFDIV32(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const MVT VT = MVT::f32;
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  auto C = [&](uint32_t V) { return DAG.getConstant(V, DL, MVT::i32); };
  auto K = [&](float V) { return DAG.getConstantFP(V, DL, VT); };
  auto Bits = [&](SDValue X) { return DAG.getBitcast(MVT::i32, X); };
  auto Flt = [&](SDValue X) { return DAG.getBitcast(VT, X); };
  auto And = [&](SDValue X, uint32_t M) {
    return DAG.getNode(ISD::AND, DL, MVT::i32, X, C(M));
  };
  auto Fma = [&](SDValue X, SDValue Y, SDValue Z) {
    return DAG.getNode(ISD::FMA, DL, VT, X, Y, Z);
  };
  auto Mul = [&](SDValue X, SDValue Y) {
    return DAG.getNode(ISD::FMUL, DL, VT, X, Y);
  };
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);

  SDValue D, NegD, Y, PB;
  SmallVector<SDValue, 2> NaNOperands;
  auto *CB = dyn_cast<ConstantFPSDNode>(B);
  if (CB && ConstDivisorFold && CB->getValueAPF().isNormal()) {
    // A constant (normal) divisor: d, its correctly rounded reciprocal and
    // the factor 2^(128 - eb) are constants, as rcp and the Newton step
    // would make them (y is the rounded 1 / d either way), and b is no NaN.
    uint32_t Bb = CB->getValueAPF().bitcastToAPInt().getZExtValue();
    uint32_t Db = (Bb & 0x807fffff) | 0x40000000;
    APFloat DV(APFloat::IEEEsingle(), APInt(32, Db));
    APFloat YV(APFloat::IEEEsingle(), "1");
    YV.divide(DV, APFloat::rmNearestTiesToEven);
    D = DAG.getConstantFP(DV, DL, VT);
    NegD = DAG.getConstantFP(-DV, DL, VT);
    Y = DAG.getConstantFP(YV, DL, VT);
    PB = Flt(C(0x7f800000 - (Bb & 0x7f800000)));
  } else {
    SDValue DB = Bits(B);
    D = Flt(DAG.getNode(ISD::OR, DL, MVT::i32, And(DB, 0x807fffff),
                        C(0x40000000)));
    NegD = DAG.getNode(ISD::FNEG, DL, VT, D);
    SDValue Y0 = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rcp, VT, D);
    Y = Fma(Fma(NegD, Y0, K(1.0f)), Y0, Y0);
    PB = Flt(DAG.getNode(ISD::SUB, DL, MVT::i32, C(0x7f800000),
                         And(DB, 0x7f800000)));
    NaNOperands.push_back(B);
  }
  SDValue R;
  auto *CA = dyn_cast<ConstantFPSDNode>(A);
  if (CA && CA->isExactlyValue(1.0)) {
    // 1 / b: y is already the rounded quotient.
    R = Mul(Y, PB);
  } else {
    SDValue AB = Bits(A);
    SDValue A1 = Flt(DAG.getNode(ISD::OR, DL, MVT::i32, And(AB, 0x807fffff),
                                 C(0x3f800000)));
    SDValue Q;
    if (CA && CA->getValueAPF().isNormal() &&
        (CA->getValueAPF().bitcastToAPInt().getZExtValue() & 0x7fffff) == 0) {
      // a = +-2^k: a1 = +-1, so q = +-y, already the rounded quotient.
      Q = CA->isNegative() ? DAG.getNode(ISD::FNEG, DL, VT, Y) : Y;
    } else {
      SDValue Q0 = Mul(A1, Y);
      Q = Fma(Fma(NegD, Q0, A1), Y, Q0);
    }
    SDValue PA = Flt(And(AB, 0x7f800000));
    auto *CPB = dyn_cast<ConstantFPSDNode>(PB);
    if (CPB && CPB->getValueAPF().convertToFloat() >= 0x1p-100f) {
      // A constant b of moderate size: q (in [1/4, 1)) times pb is exact
      // (neither flushed nor infinite), so pb first is right for every a and
      // the product with pa rounds once; no test of a's size.
      if (!CPB->isExactlyValue(1.0))
        Q = Mul(Q, PB);
      R = Mul(Q, PA);
    } else {
      SDValue Small = DAG.getSetCC(DL, CCVT, DAG.getNode(ISD::FABS, DL, VT, A),
                                   K(0x1p-124f), ISD::SETOLT);
      R = Mul(Mul(Q, DAG.getSelect(DL, VT, Small, PB, PA)),
              DAG.getSelect(DL, VT, Small, PA, PB));
    }
    NaNOperands.push_back(A);
  }
  if (Op->getFlags().hasNoNaNs() || NaNOperands.empty())
    return R;
  // One compare-and-select per operand.
  SDValue NaN =
      DAG.getConstantFP(APFloat::getQNaN(APFloat::IEEEsingle()), DL, VT);
  for (SDValue X : NaNOperands)
    R = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, X, X, ISD::SETUO), NaN,
                      R);
  return R;
}

static cl::opt<bool> PreciseHalfDiv(
    "agx-precise-half-div",
    cl::desc("correctly rounded half division (without arcp / afn)"),
    cl::init(true), cl::Hidden);

SDValue AGXTargetLowering::lowerFDIV(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDNodeFlags Flags = Op->getFlags();
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  bool Approx = Flags.hasAllowReciprocal() || Flags.hasApproximateFuncs();
  if (VT == MVT::f32 && !Approx)
    return lowerFDIV32(Op, DAG);
  if (VT == MVT::f16 && !Approx && PreciseHalfDiv) {
    // Correctly rounded: the half rcp times a is up to an ulp off (a fifth of
    // random operand pairs on the simulator). In float, the correctly rounded
    // reciprocal y = y0 + y0 (1 - b y0) and one correction of q1 = a y,
    // q = q1 + (a - b q1) y, give the correctly rounded float quotient
    // (lowerFDIV32's steps: halves are normal floats and their quotients
    // far from the float's flushed and infinite ranges, so no scaling), and
    // float has the 2 * 11 + 2 bits for which rounding that to half is the
    // correctly rounded half quotient. Zeros, infinities and NaNs (a NaN q,
    // or a zero a whose correction can lose the sign) take a y0, which is
    // exact for them.
    auto F = [&](SDValue V) {
      return DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, V);
    };
    auto K = [&](float V) { return DAG.getConstantFP(V, DL, MVT::f32); };
    SDValue A32 = F(A), B32 = F(B);
    SDValue NegB = DAG.getNode(ISD::FNEG, DL, MVT::f32, B32);
    SDValue Y0, Y;
    auto *CB = dyn_cast<ConstantFPSDNode>(B);
    if (CB && ConstDivisorFold && CB->getValueAPF().isNormal()) {
      // A constant b: y is its rounded float reciprocal (what the Newton
      // step gives), and serves as y0 too (exact for a's zeros, infinities
      // and NaNs, with b's sign).
      APFloat BV = CB->getValueAPF();
      bool LosesInfo;
      BV.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                 &LosesInfo);
      APFloat YV(APFloat::IEEEsingle(), "1");
      YV.divide(BV, APFloat::rmNearestTiesToEven);
      Y0 = Y = DAG.getConstantFP(YV, DL, MVT::f32);
    } else {
      Y0 = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rcp, MVT::f32, B32);
      Y = DAG.getNode(ISD::FMA, DL, MVT::f32,
                      DAG.getNode(ISD::FMA, DL, MVT::f32, NegB, Y0, K(1.0f)),
                      Y0, Y0);
    }
    SDValue Q1 = DAG.getNode(ISD::FMUL, DL, MVT::f32, A32, Y);
    SDValue Q = DAG.getNode(
        ISD::FMA, DL, MVT::f32,
        DAG.getNode(ISD::FMA, DL, MVT::f32, NegB, Q1, A32), Y, Q1);
    SDValue Simple = DAG.getNode(ISD::FMUL, DL, MVT::f32, A32, Y0);
    EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(),
                                  MVT::f32);
    SDValue R = DAG.getSelect(DL, MVT::f32,
                              DAG.getSetCC(DL, CCVT, Q, Q, ISD::SETUO), Simple, Q);
    R = DAG.getSelect(
        DL, MVT::f32,
        DAG.getSetCC(DL, CCVT, A, DAG.getConstantFP(0.0, DL, MVT::f16),
                     ISD::SETOEQ),
        Simple, R);
    return DAG.getNode(ISD::FP_ROUND, DL, MVT::f16, R,
                       DAG.getTargetConstant(0, DL, MVT::i32));
  }
  SDValue R = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rcp, VT, B);
  auto *CA = dyn_cast<ConstantFPSDNode>(A);
  if (CA && CA->isExactlyValue(1.0))
    return R;
  return DAG.getNode(ISD::FMUL, DL, VT, A, R, Flags);
}

/// float sqrt without afn, correctly rounded (denormals flushed): from
/// y = rsqrt_special(x) (within 1 ulp; 1 for 0 and infinity),
///   s0 = x y, s = s0 + (x - s0^2) y / 2
/// is correctly rounded for every x in [1, 4) with the recorded table, and
/// so for every x whose remainder x - s0^2 is not flushed: x below 2^-80 is
/// scaled by 2^64 first. +-0 gives +-0, and the NaN of +inf's remainder
/// (s0 = inf), negative values and NaNs are s0.
SDValue AGXTargetLowering::lowerFSQRT(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getValueType() == MVT::f16 && !Op->getFlags().hasApproximateFuncs() &&
      PreciseHalfDiv) {
    // Correctly rounded: x rsqrt_special(x) in half is an ulp off for 8,520
    // of the 65,536 halves (simulator). The float square root below, rounded
    // to half, is the correctly rounded half one (float has the 2 * 11 + 2
    // bits for which the double rounding is exact).
    // A half is never below the 2^-80 that the float path scales first.
    SDLoc DL(Op);
    const MVT VT = MVT::f32;
    SDValue X = DAG.getNode(ISD::FP_EXTEND, DL, VT, Op.getOperand(0));
    EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
    SDValue Y = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rsqrt_special, VT, X);
    SDValue S0 = DAG.getNode(ISD::FMUL, DL, VT, X, Y);
    SDValue H = DAG.getNode(ISD::FMUL, DL, VT, Y, DAG.getConstantFP(0.5, DL, VT));
    SDValue Rem = DAG.getNode(ISD::FMA, DL, VT,
                              DAG.getNode(ISD::FNEG, DL, VT, S0), S0, X);
    SDValue S = DAG.getNode(ISD::FMA, DL, VT, Rem, H, S0);
    S = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, S, S, ISD::SETUO), S0, S);
    return DAG.getNode(ISD::FP_ROUND, DL, MVT::f16, S,
                       DAG.getTargetConstant(0, DL, MVT::i32));
  }
  if (Op.getValueType() != MVT::f32 || Op->getFlags().hasApproximateFuncs())
    return Op;
  SDLoc DL(Op);
  const MVT VT = MVT::f32;
  SDValue X = Op.getOperand(0);
  auto K = [&](float V) { return DAG.getConstantFP(V, DL, VT); };
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue Small = DAG.getSetCC(DL, CCVT, X, K(0x1p-80f), ISD::SETOLT);
  SDValue XS = DAG.getSelect(
      DL, VT, Small, DAG.getNode(ISD::FMUL, DL, VT, X, K(0x1p64f)), X);
  SDValue Y = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rsqrt_special, VT, XS);
  SDValue S0 = DAG.getNode(ISD::FMUL, DL, VT, XS, Y);
  SDValue H = DAG.getNode(ISD::FMUL, DL, VT, Y, K(0.5f));
  SDValue Rem = DAG.getNode(ISD::FMA, DL, VT,
                            DAG.getNode(ISD::FNEG, DL, VT, S0), S0, XS);
  SDValue S = DAG.getNode(ISD::FMA, DL, VT, Rem, H, S0);
  S = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, S, S, ISD::SETUO), S0, S);
  return DAG.getSelect(DL, VT, Small,
                       DAG.getNode(ISD::FMUL, DL, VT, S, K(0x1p-32f)), S);
}

/// copysign(m, s) = (m & ~sign) | (s & sign), on the bits.
/// minimumnum / maximumnum: the hardware's NaN-ignoring min / max (fminnum /
/// fmaxnum, one fcmpsel with an "n" condition), except for operands that
/// compare equal: there the bits of both ORed (min) or ANDed (max), which
/// orders -0 below +0 and leaves equal nonzero values as they are.
/// minimum / maximum: the same, and the default NaN for a NaN operand.
static cl::opt<bool> SelectMinimumMaximum(
    "agx-select-minimum-maximum",
    cl::desc("minimum / maximum as unordered compare-selects"),
    cl::init(true), cl::Hidden);

/// minimum / maximum (NaN if either operand is one, -0 below +0) as
/// select(a ugt b, a, b) for maximum (ult for minimum): a NaN a is taken by
/// the unordered comparison, a NaN b by one more select (any NaN operand is
/// a NaN result LLVM allows, so no default NaN constant), and equal operands
/// of two signs by the and / or of their bits. A constant goes second, a
/// non-NaN one needs no NaN select, a nonzero one no zeros, and +0 for
/// maximum (-0 for minimum) none either (maximum(x, +0) is x ugt 0 ? x : +0,
/// one fcmpsel: relu).
static SDValue lowerFMinimumMaximum(SelectionDAG &DAG, const SDLoc &DL, EVT VT,
                                    SDValue A, SDValue B, bool Max,
                                    bool NoSignedZeros) {
  EVT CCVT = MVT::i32;
  if (isa<ConstantFPSDNode>(A) && !isa<ConstantFPSDNode>(B))
    std::swap(A, B);
  auto *CB = dyn_cast<ConstantFPSDNode>(B);
  ISD::CondCode CC = Max ? ISD::SETUGT : ISD::SETULT;
  bool Zeros = !NoSignedZeros && !(CB && !CB->isZero());
  // maximum(a, +0) and minimum(a, -0): the strict comparison already gives
  // b for every a that compares equal to it (+-0, and denormals, which the
  // comparison flushes). maximum(a, -0) and minimum(a, +0) keep the and / or
  // of the bits: an inclusive comparison would give a, a denormal one too
  // (agx-tv: maximum(-denormal, -0) is -0, not the denormal).
  if (Zeros && CB && Max == !CB->isNegative())
    Zeros = false;
  SDValue R = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, A, B, CC), A, B);
  if (Zeros) {
    EVT IntVT = VT.changeTypeToInteger();
    // Equal to a zero b (a = +-0 or a denormal the comparison flushes): the
    // zero of a's sign, a & sign; a | +0 would give a denormal a back
    // (agx-tv: minimum(+denormal, +0) is +0).
    SDValue Bits =
        CB ? DAG.getNode(ISD::AND, DL, IntVT, DAG.getBitcast(IntVT, A),
                         DAG.getConstant(APInt::getSignMask(
                                             IntVT.getSizeInBits()),
                                         DL, IntVT))
           : DAG.getNode(Max ? ISD::AND : ISD::OR, DL, IntVT,
                         DAG.getBitcast(IntVT, A), DAG.getBitcast(IntVT, B));
    SDValue Eq = DAG.getBitcast(VT, Bits);
    // Two denormals the comparison flushes compare equal, and their and /
    // or is a third denormal: where f32 arithmetic flushes (no ieee-fp32,
    // the contract the backend compiles for), the canonical value of the
    // bits (a zero of their sign; others unchanged, and no NaN is equal).
    // agx-tv: maximum of two denormals is +-0 there.
    if (!CB && VT == MVT::f32 && !DAG.getSubtarget<AGXSubtarget>().hasIEEEFP32())
      Eq = DAG.getNode(ISD::FCANONICALIZE, DL, VT, Eq);
    R = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, A, B, ISD::SETOEQ), Eq,
                      R);
  }
  if (!(CB && !CB->isNaN()))
    R = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, B, B, ISD::SETUO), B, R);
  return R;
}

static cl::opt<bool> MaxNumZeroSelect(
    "agx-maxnum-zero-select",
    cl::desc("maximumnum(x, +0) as one select on x > 0"), cl::init(true),
    cl::Hidden);

SDValue AGXTargetLowering::lowerFMinMaxNum(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue A = Op.getOperand(0), B = Op.getOperand(1);
  unsigned Opc = Op.getOpcode();
  bool Max = Opc == ISD::FMAXIMUMNUM || Opc == ISD::FMAXIMUM;
  bool PropagateNaN = Opc == ISD::FMINIMUM || Opc == ISD::FMAXIMUM;
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  if (PropagateNaN && SelectMinimumMaximum && !Op->getFlags().hasNoNaNs())
    return lowerFMinimumMaximum(DAG, DL, VT, A, B, Max,
                                Op->getFlags().hasNoSignedZeros());
  // maximumnum(x, +0) is (x > 0) ? x : +0: a NaN, -0 and the zeros give +0
  // (as does a positive denormal where f32 compares flush, its flushed
  // value). One fcmpsel instead of the max and its fix for zeros.
  if (!PropagateNaN && Max && MaxNumZeroSelect &&
      (VT == MVT::f16 ||
       !DAG.getSubtarget<AGXSubtarget>().hasIEEEFP32())) {
    auto PosZero = [](SDValue V) {
      auto *C = dyn_cast<ConstantFPSDNode>(V);
      return C && C->isZero() && !C->isNegative();
    };
    if (PosZero(A))
      std::swap(A, B);
    if (PosZero(B))
      return DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, A, B, ISD::SETOGT),
                           A, B);
  }
  SDValue R = DAG.getNode(Max ? ISD::FMAXNUM : ISD::FMINNUM, DL, VT, A, B,
                          Op->getFlags());
  // (equal operands with different bits are zeros of two signs: none where
  // one is a nonzero constant, as in a clamp)
  auto NonZeroConst = [](SDValue V) {
    auto *C = dyn_cast<ConstantFPSDNode>(V);
    return C && !C->isZero();
  };
  if (!Op->getFlags().hasNoSignedZeros() && !NonZeroConst(A) &&
      !NonZeroConst(B)) {
    EVT IntVT = VT.changeTypeToInteger();
    SDValue Bits = DAG.getNode(Max ? ISD::AND : ISD::OR, DL, IntVT,
                               DAG.getBitcast(IntVT, A),
                               DAG.getBitcast(IntVT, B));
    SDValue Equal = DAG.getSetCC(DL, CCVT, A, B, ISD::SETOEQ);
    SDValue Eq = DAG.getBitcast(VT, Bits);
    // Where f32 compares flush, two different denormals of a sign compare
    // equal to each other (and to a zero), and the and / or of their bits
    // would be a third denormal: flushed, it is the zero of the result's
    // sign, which the denormal mode allows.
    if (VT == MVT::f32 && !DAG.getSubtarget<AGXSubtarget>().hasIEEEFP32())
      Eq = DAG.getNode(ISD::FCANONICALIZE, DL, VT, Eq);
    R = DAG.getSelect(DL, VT, Equal, Eq, R);
  }
  if (!PropagateNaN || Op->getFlags().hasNoNaNs())
    return R;
  // minimum / maximum: the default NaN when either operand is a NaN.
  SDValue NaN = DAG.getConstantFP(
      APFloat::getQNaN(VT.getFltSemantics()), DL, VT);
  for (SDValue X : {B, A})
    R = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, X, X, ISD::SETOEQ), R,
                      NaN);
  return R;
}

static cl::opt<bool> CopysignBfi("agx-copysign-bfi",
                                 cl::desc("copysign as one bfi"),
                                 cl::init(true), cl::Hidden);

SDValue AGXTargetLowering::lowerFCOPYSIGN(SDValue Op,
                                          SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  EVT IntVT = VT == MVT::f16 ? MVT::i16 : MVT::i32;
  SDValue Mag = Op.getOperand(0);
  SDValue Sign = Op.getOperand(1);
  if (Sign.getValueType() != VT)
    Sign = Sign.getValueType().bitsLT(VT)
               ? DAG.getNode(ISD::FP_EXTEND, DL, VT, Sign)
               : DAG.getNode(ISD::FP_ROUND, DL, VT, Sign,
                             DAG.getTargetConstant(0, DL, MVT::i32));
  if (CopysignBfi && !isa<ConstantFPSDNode>(Mag)) {
    // The magnitude's low bits inserted into the sign's value: one bfi,
    // instead of two masks (one of them a constant in a register, the other
    // a bfeil on the same unit as the bfi) and an or. A constant magnitude
    // keeps the masks, which stay on the ALUs (copysign(0, x) is one and).
    unsigned BW = IntVT.getSizeInBits();
    return DAG.getBitcast(
        VT, DAG.getNode(AGXISD::BFI_FIELD, DL, IntVT,
                        DAG.getBitcast(IntVT, Sign), DAG.getBitcast(IntVT, Mag),
                        DAG.getConstant(0, DL, IntVT),
                        DAG.getTargetConstant(BW - 1, DL, MVT::i32)));
  }
  APInt SignMask = APInt::getSignMask(IntVT.getSizeInBits());
  SDValue M = DAG.getNode(ISD::AND, DL, IntVT,
                          DAG.getBitcast(IntVT, Mag),
                          DAG.getConstant(~SignMask, DL, IntVT));
  SDValue S = DAG.getNode(ISD::AND, DL, IntVT, DAG.getBitcast(IntVT, Sign),
                          DAG.getConstant(SignMask, DL, IntVT));
  return DAG.getBitcast(VT, DAG.getNode(ISD::OR, DL, IntVT, M, S));
}

static cl::opt<bool> ClassCompare(
    "agx-fpclass-compare",
    cl::desc("finiteness / infinity tests of a float's bits as float "
             "compares"),
    cl::init(true), cl::Hidden);

/// The bit tests of a float's class that one float compare answers, as the
/// generic is.fpclass expansion (and hand-written code) writes them:
///
///   (bits << 1) <u (all-ones exponent << 1)   finite      |x| <  inf
///   (bits << 1) >u that - 1                   not finite  |x| >= inf (uge)
///   (bits & ~sign) == inf                     infinite    |x| == inf
///   (bits & ~sign) != inf                     not         |x| != inf (une)
///
/// An fcmpsel of x with the .abs modifier and +inf (a mov_imm, as the bit
/// form's constant is) instead of the shift or mask on the shared unit and an
/// icmpsel. A flushed subnormal is a zero, finite as the subnormal is.
SDValue AGXTargetLowering::performClassCompareCombine(
    SDNode *N, DAGCombinerInfo &DCI) const {
  if (!ClassCompare)
    return SDValue();
  SDValue L = N->getOperand(0);
  auto *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  EVT IntVT = L.getValueType();
  if (!C || (IntVT != MVT::i32 && IntVT != MVT::i16))
    return SDValue();
  unsigned BW = IntVT.getSizeInBits();
  EVT FVT = BW == 32 ? MVT::f32 : MVT::f16;
  APInt Inf = APFloat::getInf(FVT.getFltSemantics()).bitcastToAPInt();
  const APInt &K = C->getAPIntValue();
  SDValue Bits;
  ISD::CondCode FCC;
  bool Shifted = (L.getOpcode() == ISD::SHL && isOneConstant(L.getOperand(1))) ||
                 (L.getOpcode() == ISD::ADD && L.getOperand(0) == L.getOperand(1));
  if (Shifted && CC == ISD::SETULT && K == Inf.shl(1)) {
    Bits = L.getOperand(0);
    FCC = ISD::SETOLT;
  } else if (Shifted && CC == ISD::SETUGT && K == Inf.shl(1) - 1) {
    Bits = L.getOperand(0);
    FCC = ISD::SETUGE;
  } else if (L.getOpcode() == ISD::AND && isa<ConstantSDNode>(L.getOperand(1)) &&
             L.getConstantOperandAPInt(1) == ~APInt::getSignMask(BW) &&
             K == Inf && (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    Bits = L.getOperand(0);
    FCC = CC == ISD::SETEQ ? ISD::SETOEQ : ISD::SETUNE;
  } else {
    return SDValue();
  }
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  SDValue X = DAG.getBitcast(FVT, Bits);
  return DAG.getSetCC(DL, N->getValueType(0), DAG.getNode(ISD::FABS, DL, FVT, X),
                      DAG.getConstantFP(APFloat::getInf(FVT.getFltSemantics()),
                                        DL, FVT),
                      FCC);
}

static cl::opt<bool> RoundOutOfClamp(
    "agx-round-out-of-clamp",
    cl::desc("move a rounding out of an integral clamp feeding a float to "
             "integer conversion"),
    cl::init(true), cl::Hidden);

/// fptosi(clamp(rnd(x))) with integral bounds -> fptosi(rnd(clamp(x))): a
/// monotone rounding that keeps integers commutes with min / max against an
/// integer (also for a NaN, which the min / max replace by the bound), and
/// outermost it is the convert's own (rint, roundeven, nearbyint: the
/// round-to-nearest-even convert; trunc: the plain one, so it goes) or ends
/// in a trunc that is (round). A quantization's round-then-clamp-then-convert
/// loses its rounding instruction.
SDValue AGXTargetLowering::performRoundClampConvertCombine(
    SDNode *N, DAGCombinerInfo &DCI) const {
  if (!RoundOutOfClamp)
    return SDValue();
  SDValue X = N->getOperand(0);
  // The convert truncates: fptosi(trunc(y)) is fptosi(y) (round's
  // expansion, trunc(y + bias), ends in one).
  if (X.getOpcode() == ISD::FTRUNC) {
    SmallVector<SDValue, 2> Ops = {X.getOperand(0)};
    if (N->getNumOperands() > 1)
      Ops.push_back(N->getOperand(1));
    return DCI.DAG.getNode(N->getOpcode(), SDLoc(N), N->getValueType(0), Ops);
  }
  SmallVector<std::pair<unsigned, SDValue>, 4> Chain;
  auto IntegralConst = [](SDValue V) {
    auto *C = dyn_cast<ConstantFPSDNode>(V);
    return C && C->getValueAPF().isInteger();
  };
  while (X.hasOneUse() &&
         (X.getOpcode() == ISD::FMAXNUM || X.getOpcode() == ISD::FMINNUM ||
          X.getOpcode() == ISD::FMAXIMUMNUM ||
          X.getOpcode() == ISD::FMINIMUMNUM)) {
    unsigned K = IntegralConst(X.getOperand(1)) ? 1
                 : IntegralConst(X.getOperand(0)) ? 0
                                                  : 2;
    if (K == 2)
      return SDValue();
    Chain.push_back({X.getOpcode(), X.getOperand(K)});
    X = X.getOperand(1 - K);
  }
  unsigned R = X.getOpcode();
  if (Chain.empty() || !X.hasOneUse() ||
      (R != ISD::FRINT && R != ISD::FROUNDEVEN && R != ISD::FNEARBYINT &&
       R != ISD::FTRUNC && R != ISD::FROUND))
    return SDValue();
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = X.getValueType();
  SDValue V = X.getOperand(0);
  for (auto It = Chain.rbegin(); It != Chain.rend(); ++It)
    V = DAG.getNode(It->first, DL, VT, V, It->second);
  if (R != ISD::FTRUNC)
    V = DAG.getNode(R, DL, VT, V);
  SmallVector<SDValue, 2> Ops = {V};
  if (N->getNumOperands() > 1)
    Ops.push_back(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), DL, N->getValueType(0), Ops);
}

/// round(x), half away from zero (no libm on AGX): t = trunc(x), plus one
/// away from zero when |x - t| >= 0.5, with the sign of x (round(-0.3) and
/// round(-0) are -0). x - t is exact, and so is t + 1 for every x that is not
/// an integer; infinities give x - t = NaN, so t, and NaN stays NaN.
static cl::opt<bool> RoundBias(
    "agx-round-bias",
    cl::desc("round(x) as trunc(x + copysign(the float below 0.5, x))"),
    cl::init(true), cl::Hidden);

SDValue AGXTargetLowering::lowerFROUND(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue X = Op.getOperand(0);
  if (RoundBias && (VT == MVT::f32 || VT == MVT::f16)) {
    // trunc(x + copysign(b, x)) with b the float just below 0.5: a fraction
    // of 0.5 or more carries x + b over the next integer (at exactly 0.5 the
    // sum is a tie, rounded to the even, larger, magnitude), and less does
    // not; at |x| >= 2^23 (2^10 for half) adding b gives x. -0, infinities
    // and NaN stay. A mov_imm, a bfi, an add and a trunc, instead of nine.
    // Checked for every float (flushing and not) and every half.
    EVT IntVT = VT == MVT::f16 ? MVT::i16 : MVT::i32;
    unsigned BW = IntVT.getSizeInBits();
    uint64_t Bias = VT == MVT::f16 ? 0x37ff : 0x3effffff;
    SDValue B = DAG.getNode(AGXISD::BFI_FIELD, DL, IntVT,
                            DAG.getBitcast(IntVT, X),
                            DAG.getConstant(Bias, DL, IntVT),
                            DAG.getConstant(0, DL, IntVT),
                            DAG.getTargetConstant(BW - 1, DL, MVT::i32));
    return DAG.getNode(ISD::FTRUNC, DL, VT,
                       DAG.getNode(ISD::FADD, DL, VT, X,
                                   DAG.getBitcast(VT, B)));
  }
  SDValue T = DAG.getNode(ISD::FTRUNC, DL, VT, X);
  SDValue D =
      DAG.getNode(ISD::FABS, DL, VT, DAG.getNode(ISD::FSUB, DL, VT, X, T));
  SDValue One = DAG.getNode(ISD::FCOPYSIGN, DL, VT,
                            DAG.getConstantFP(1.0, DL, VT), X);
  SDValue Step = DAG.getSelectCC(DL, D, DAG.getConstantFP(0.5, DL, VT), One,
                                 DAG.getConstantFP(0.0, DL, VT), ISD::SETOGE);
  return DAG.getNode(ISD::FCOPYSIGN, DL, VT,
                     DAG.getNode(ISD::FADD, DL, VT, T, Step), X);
}

/// float to bfloat bits, rounded to nearest even: (x + 0x7fff + lsb) >> 16,
/// with NaNs quieted.
/// Whether the float (or its bits) V is a bfloat exactly (its low 16 bits
/// zero): a widened bfloat, a constant, and the sign operations, minimum /
/// maximum, selects and bit operations of such values (the hardware's min /
/// max give an operand, or the NaN 0x7fc00000).
static bool isBF16Exact(SelectionDAG &DAG, SDValue V, unsigned Depth = 0) {
  if (Depth > 4)
    return false;
  switch (V.getOpcode()) {
  case ISD::FP_EXTEND:
    return V.getOperand(0).getValueType().getScalarType() == MVT::bf16;
  case ISD::BF16_TO_FP:
    return true;
  case ISD::ConstantFP: {
    APInt Bits = cast<ConstantFPSDNode>(V)->getValueAPF().bitcastToAPInt();
    return Bits.countr_zero() >= 16;
  }
  case ISD::BITCAST:
    return isBF16Exact(DAG, V.getOperand(0), Depth + 1);
  case ISD::AND:
    return isBF16Exact(DAG, V.getOperand(0), Depth + 1) ||
           isBF16Exact(DAG, V.getOperand(1), Depth + 1) ||
           DAG.computeKnownBits(V).countMinTrailingZeros() >= 16;
  case ISD::OR:
  case ISD::XOR:
    return isBF16Exact(DAG, V.getOperand(0), Depth + 1) &&
           isBF16Exact(DAG, V.getOperand(1), Depth + 1);
  case ISD::FNEG:
  case ISD::FABS:
  // An integer rounding of a bfloat is one: from 2^7 it is the value itself,
  // below it an integer of at most 128 (a NaN's low half stays zero).
  case ISD::FFLOOR:
  case ISD::FCEIL:
  case ISD::FTRUNC:
  case ISD::FROUND:
  case ISD::FROUNDEVEN:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
    return isBF16Exact(DAG, V.getOperand(0), Depth + 1);
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FMINIMUM:
  case ISD::FMAXIMUM:
  case ISD::FMINIMUMNUM:
  case ISD::FMAXIMUMNUM:
    return isBF16Exact(DAG, V.getOperand(0), Depth + 1) &&
           isBF16Exact(DAG, V.getOperand(1), Depth + 1);
  case ISD::SELECT:
    return isBF16Exact(DAG, V.getOperand(1), Depth + 1) &&
           isBF16Exact(DAG, V.getOperand(2), Depth + 1);
  case ISD::SELECT_CC:
    return isBF16Exact(DAG, V.getOperand(2), Depth + 1) &&
           isBF16Exact(DAG, V.getOperand(3), Depth + 1);
  default:
    return V.getValueType().isInteger() &&
           DAG.computeKnownBits(V).countMinTrailingZeros() >= 16;
  }
}

SDValue AGXTargetLowering::lowerFP_TO_BF16(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Src = Op.getOperand(Op->isStrictFPOpcode() ? 1 : 0);
  if (Src.getValueType() != MVT::f32)
    Src = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, Src);
  SDValue X = DAG.getBitcast(MVT::i32, Src);
  auto C = [&](uint32_t V) { return DAG.getConstant(V, DL, MVT::i32); };
  // already a bfloat (min / max of bfloats): its top half
  if (isBF16Exact(DAG, Src))
    return DAG.getNode(ISD::TRUNCATE, DL, Op.getValueType(),
                       DAG.getNode(ISD::SRL, DL, MVT::i32, X, C(16)));
  SDValue Lsb = DAG.getNode(ISD::AND, DL, MVT::i32,
                            DAG.getNode(ISD::SRL, DL, MVT::i32, X, C(16)), C(1));
  SDValue R = DAG.getNode(ISD::ADD, DL, MVT::i32, X,
                          DAG.getNode(ISD::ADD, DL, MVT::i32, Lsb, C(0x7fff)));
  // The f32 arithmetic's NaN is 0x7fc00000, which the rounding keeps a NaN:
  // a result of it needs no NaN test (unless NaNs follow PTX's canonical
  // form, whose 0x7fffffff the rounding would carry out of).
  unsigned SrcOp = Src.getOpcode();
  if (!DAG.getSubtarget<AGXSubtarget>().hasCanonicalNaN() &&
      (SrcOp == ISD::FADD || SrcOp == ISD::FSUB || SrcOp == ISD::FMUL ||
       SrcOp == ISD::FMA || SrcOp == ISD::FMAD))
    return DAG.getNode(ISD::TRUNCATE, DL, Op.getValueType(),
                       DAG.getNode(ISD::SRL, DL, MVT::i32, R, C(16)));
  // A NaN keeps its top half, quieted; the test is a float compare of the
  // value with itself (one compare-select, no magnitude), and the select is
  // of the 32-bit words, so one shift serves both.
  SDValue Quiet = DAG.getNode(ISD::OR, DL, MVT::i32, X, C(0x400000));
  EVT CCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);
  SDValue IsNaN = DAG.getSetCC(DL, CCVT, Src, Src, ISD::SETUO);
  R = DAG.getSelect(DL, MVT::i32, IsNaN, Quiet, R);
  return DAG.getNode(ISD::TRUNCATE, DL, Op.getValueType(),
                     DAG.getNode(ISD::SRL, DL, MVT::i32, R, C(16)));
}

SDValue AGXTargetLowering::lowerFREM(SDValue Op, SelectionDAG &DAG) const {
  // a - trunc(a / b) * b
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  SDValue Q = DAG.getNode(ISD::FDIV, DL, VT, A, B, Op->getFlags());
  SDValue T = DAG.getNode(ISD::FTRUNC, DL, VT, Q);
  return DAG.getNode(ISD::FMA, DL, VT, DAG.getNode(ISD::FNEG, DL, VT, T), B,
                     A);
}

static cl::opt<bool> PreciseHalfTrig(
    "agx-precise-half-trig",
    cl::desc("half sin / cos with an exact reduction to turns"),
    cl::init(true), cl::Hidden);

SDValue AGXTargetLowering::lowerTrig(SDValue Op, SelectionDAG &DAG) const {
  // The hardware evaluates sin(x * pi / 2) for x in [0, 4): reduce the
  // argument to turns, take the fraction and scale to quadrants. cos(x) is
  // sin(x + pi / 2), a quarter turn ahead. With ieee-fp32 (strict floating
  // point, which takes what costs time) the quadrant is added to the
  // quadrants q rather than the turns, whose sum rounds to 2^-24 turns past 1
  // and took cos.approx over PTX's 2^-20.4 on [-2pi, 2pi]: q + 1, or q - 3
  // (exact) past 3. (sin(-0) = -0 there too: AGXStrictFP returns arguments
  // below 2^-12.)
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  if (VT == MVT::f16) {
    // In float. Without afn, the turns of a half x exactly: x / 2pi with
    // 1 / 2pi = c_hi + c_mid, c_hi of 13 bits so that x c_hi (11 bits by 13)
    // is exact and so is its fraction; then x c_mid added by an fma and the
    // fraction taken again. x / 2pi rounded to float was up to 2^-24 t turns
    // off, which put half sin / cos up to 1,517 / 4,800 ulp off for large x
    // (simulator, every half).
    SDValue X = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, Op.getOperand(0));
    SDValue R;
    if (Op->getFlags().hasApproximateFuncs() || !PreciseHalfTrig) {
      R = lowerTrig(DAG.getNode(Op.getOpcode(), DL, MVT::f32, X, Op->getFlags()),
                    DAG);
    } else {
      auto KF = [&](double V) { return DAG.getConstantFP(V, DL, MVT::f32); };
      auto Frac = [&](SDValue T) {
        return DAG.getNode(ISD::FSUB, DL, MVT::f32, T,
                           DAG.getNode(ISD::FFLOOR, DL, MVT::f32, T));
      };
      SDValue T = Frac(DAG.getNode(ISD::FMUL, DL, MVT::f32, X,
                                   KF(0.159149169921875)));
      if (Op.getOpcode() == ISD::FCOS)
        T = DAG.getNode(ISD::FADD, DL, MVT::f32, T, KF(0.25));
      T = Frac(DAG.getNode(ISD::FMA, DL, MVT::f32, X,
                           KF(5.773169959866209e-06), T));
      R = DAG.getNode(AGXISD::SIN_QUADRANTS, DL, MVT::f32,
                      DAG.getNode(ISD::FMUL, DL, MVT::f32, T, KF(4.0)));
    }
    return DAG.getNode(ISD::FP_ROUND, DL, MVT::f16, R,
                       DAG.getTargetConstant(0, DL, MVT::i32));
  }
  auto K = [&](double V) { return DAG.getConstantFP(V, DL, VT); };
  bool Precise = DAG.getSubtarget<AGXSubtarget>().hasIEEEFP32();
  SDValue Turns =
      DAG.getNode(ISD::FMUL, DL, VT, Op.getOperand(0), K(0.5 * M_1_PI));
  if (Op.getOpcode() == ISD::FCOS && !Precise)
    Turns = DAG.getNode(ISD::FADD, DL, VT, Turns, K(0.25));
  SDValue Frac = DAG.getNode(ISD::FSUB, DL, VT, Turns,
                             DAG.getNode(ISD::FFLOOR, DL, VT, Turns));
  SDValue Quadrants = DAG.getNode(ISD::FMUL, DL, VT, Frac, K(4.0));
  if (Op.getOpcode() == ISD::FCOS && Precise) {
    EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
    SDValue Past3 = DAG.getSetCC(DL, CCVT, Quadrants, K(3.0), ISD::SETOGE);
    Quadrants =
        DAG.getSelect(DL, VT, Past3,
                      DAG.getNode(ISD::FSUB, DL, VT, Quadrants, K(3.0)),
                      DAG.getNode(ISD::FADD, DL, VT, Quadrants, K(1.0)));
  }
  return DAG.getNode(AGXISD::SIN_QUADRANTS, DL, VT, Quadrants);
}

SDValue AGXTargetLowering::lowerExpLog(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue X = Op.getOperand(0);
  auto K = [&](double V) { return DAG.getConstantFP(V, DL, VT); };
  switch (Op.getOpcode()) {
  case ISD::FEXP:
    return DAG.getNode(ISD::FEXP2, DL, VT,
                       DAG.getNode(ISD::FMUL, DL, VT, X, K(M_LOG2E)));
  case ISD::FEXP10:
    return DAG.getNode(ISD::FEXP2, DL, VT,
                       DAG.getNode(ISD::FMUL, DL, VT, X, K(3.321928094887362)));
  case ISD::FLOG:
    return DAG.getNode(ISD::FMUL, DL, VT,
                       DAG.getNode(ISD::FLOG2, DL, VT, X), K(M_LN2));
  case ISD::FLOG10:
    return DAG.getNode(ISD::FMUL, DL, VT,
                       DAG.getNode(ISD::FLOG2, DL, VT, X),
                       K(0.301029995663981195));
  case ISD::FPOW:
    return DAG.getNode(
        ISD::FEXP2, DL, VT,
        DAG.getNode(ISD::FMUL, DL, VT, Op.getOperand(1),
                    DAG.getNode(ISD::FLOG2, DL, VT, X)));
  default:
    llvm_unreachable("unexpected opcode");
  }
}

SDValue AGXTargetLowering::lowerINT_TO_FP64(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  bool Signed = Op.getOpcode() == ISD::SINT_TO_FP;
  SDValue X = Op.getOperand(0);
  auto C32 = [&](uint64_t V) { return DAG.getConstant(V, DL, MVT::i32); };

  SDValue Sign;
  if (Signed) {
    // Convert |x| and apply the sign at the end.
    SDValue S = DAG.getNode(ISD::SRA, DL, MVT::i32, split64(DAG, DL, X).second,
                            C32(31));
    SDValue S64 = join64(DAG, DL, S, S);
    X = DAG.getNode(ISD::SUB, DL, MVT::i64,
                    DAG.getNode(ISD::XOR, DL, MVT::i64, X, S64), S64);
    Sign = DAG.getNode(ISD::AND, DL, MVT::i32, S, C32(0x80000000));
  }

  // Normalize so that the top bit is set, keep the high 32 bits with a
  // sticky bit for the rest, convert with round to nearest even and scale by
  // 2^(32 - lz). This rounds correctly since 32 bits exceed the 24 bit
  // significand by more than the two guard bits.
  SDValue LZ = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32,
                           DAG.getNode(ISD::CTLZ, DL, MVT::i64, X));
  SDValue Norm = DAG.getNode(ISD::SHL, DL, MVT::i64, X, LZ);
  auto [NLo, NHi] = split64(DAG, DL, Norm);
  SDValue Sticky = DAG.getSetCC(DL, MVT::i32, NLo, C32(0), ISD::SETNE);
  SDValue H = DAG.getNode(ISD::OR, DL, MVT::i32, NHi, Sticky);
  SDValue F = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, H);
  SDValue ScaleBits = DAG.getNode(
      ISD::SHL, DL, MVT::i32,
      DAG.getNode(ISD::SUB, DL, MVT::i32, C32(127 + 32), LZ), C32(23));
  SDValue R = DAG.getNode(ISD::FMUL, DL, MVT::f32, F,
                          DAG.getBitcast(MVT::f32, ScaleBits));
  if (Signed)
    R = DAG.getBitcast(
        MVT::f32, DAG.getNode(ISD::OR, DL, MVT::i32,
                              DAG.getBitcast(MVT::i32, R), Sign));
  if (VT != MVT::f32)
    R = DAG.getNode(ISD::FP_ROUND, DL, VT, R,
                    DAG.getIntPtrConstant(0, DL, /*isTarget=*/true));
  return R;
}

SDValue AGXTargetLowering::lowerFP_TO_INT64(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  bool Signed = Op.getOpcode() == ISD::FP_TO_SINT;
  SDValue X = Op.getOperand(0);
  if (X.getValueType() != MVT::f32)
    X = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, X);

  SDValue T = DAG.getNode(ISD::FTRUNC, DL, MVT::f32, X);
  SDValue Sign;
  if (Signed) {
    Sign = DAG.getNode(ISD::SRA, DL, MVT::i32, DAG.getBitcast(MVT::i32, T),
                       DAG.getConstant(31, DL, MVT::i32));
    T = DAG.getNode(ISD::FABS, DL, MVT::f32, T);
  }

  // hi = floor(t * 2^-32), lo = t - hi * 2^32, both exact.
  SDValue K0 = DAG.getConstantFP(0x1p-32, DL, MVT::f32);
  SDValue K1 = DAG.getConstantFP(-0x1p+32, DL, MVT::f32);
  SDValue HiF = DAG.getNode(ISD::FFLOOR, DL, MVT::f32,
                            DAG.getNode(ISD::FMUL, DL, MVT::f32, T, K0));
  SDValue LoF = DAG.getNode(ISD::FMA, DL, MVT::f32, HiF, K1, T);
  SDValue Hi = DAG.getNode(ISD::FP_TO_UINT, DL, MVT::i32, HiF);
  SDValue Lo = DAG.getNode(ISD::FP_TO_UINT, DL, MVT::i32, LoF);
  SDValue R = join64(DAG, DL, Lo, Hi);
  if (Signed) {
    SDValue S64 = join64(DAG, DL, Sign, Sign);
    R = DAG.getNode(ISD::SUB, DL, MVT::i64,
                    DAG.getNode(ISD::XOR, DL, MVT::i64, R, S64), S64);
  }
  return R;
}
