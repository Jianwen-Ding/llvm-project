//===-- AGXISelDAGToDAG.cpp - A DAG to DAG instruction selector for AGX ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction selection for AGX: the TableGen generated matcher plus the
// complex patterns for ALU sources and memory addresses.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXISelLowering.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "agx-isel"
#define PASS_NAME "AGX DAG->DAG Pattern Instruction Selection"

namespace {

class AGXDAGToDAGISel : public SelectionDAGISel {
  const AGXSubtarget *Subtarget = nullptr;

public:
  AGXDAGToDAGISel() = delete;
  explicit AGXDAGToDAGISel(AGXTargetMachine &TM) : SelectionDAGISel(TM) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<AGXSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

private:
#include "AGXGenDAGISel.inc"

  void Select(SDNode *N) override;
  bool trySelectShift64(SDNode *N);
  bool trySelectFieldInsert(SDNode *N);

  SDValue getI32Imm(uint64_t Imm, const SDLoc &DL) {
    return CurDAG->getTargetConstant(Imm, DL, MVT::i32);
  }

  // Complex patterns.
  bool SelectIntSrc(SDValue N, SDValue &Src);
  bool SelectFPSrc(SDValue N, SDValue &Src, SDValue &Mods);
  bool SelectExt16Src(SDValue N, SDValue &Src, SDValue &Mods);
  bool SelectExt16Src64(SDValue N, SDValue &Src, SDValue &Mods);
  bool SelectDeviceAddr(SDValue Addr, unsigned ElemLog2, SDValue &Base,
                        SDValue &Index, SDValue &Signed, SDValue &Shift);
  bool SelectDeviceAddr8(SDValue Addr, SDValue &Base, SDValue &Index,
                         SDValue &Signed, SDValue &Shift) {
    return SelectDeviceAddr(Addr, 0, Base, Index, Signed, Shift);
  }
  bool SelectDeviceAddr16(SDValue Addr, SDValue &Base, SDValue &Index,
                          SDValue &Signed, SDValue &Shift) {
    return SelectDeviceAddr(Addr, 1, Base, Index, Signed, Shift);
  }
  bool SelectDeviceAddr32(SDValue Addr, SDValue &Base, SDValue &Index,
                          SDValue &Signed, SDValue &Shift) {
    return SelectDeviceAddr(Addr, 2, Base, Index, Signed, Shift);
  }
  bool SelectAtomicAddr(SDValue Addr, SDValue &Base, SDValue &Index,
                        SDValue &Signed);
  bool SelectDeviceAddr32Via16(SDValue Addr, SDValue &Base, SDValue &Index,
                               SDValue &Signed, SDValue &Shift);
  bool SelectLocalAddr(SDValue Addr, SDValue &Base, SDValue &Index,
                       unsigned Unit);
  bool SelectLocalAddr8(SDValue Addr, SDValue &Base, SDValue &Index) {
    return SelectLocalAddr(Addr, Base, Index, 4);
  }
  bool SelectLocalAddr16(SDValue Addr, SDValue &Base, SDValue &Index) {
    return SelectLocalAddr(Addr, Base, Index, 8);
  }
  bool SelectLocalAddr32(SDValue Addr, SDValue &Base, SDValue &Index) {
    return SelectLocalAddr(Addr, Base, Index, 16);
  }
  bool SelectStackAddr(SDValue Addr, SDValue &Offset);

  void selectVectorLoad(SDNode *N, unsigned NumElts);
  void selectByteVectorLoad(SDNode *N, unsigned NumElts);
  void selectByteVectorStore(SDNode *N, unsigned NumElts);
  void selectVectorStore(SDNode *N, unsigned NumElts);
  void selectTextureSample(SDNode *N);
};

class AGXDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  explicit AGXDAGToDAGISelLegacy(AGXTargetMachine &TM)
      : SelectionDAGISelLegacy(ID, std::make_unique<AGXDAGToDAGISel>(TM)) {}
};

} // namespace

char AGXDAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(AGXDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createAGXISelDag(AGXTargetMachine &TM) {
  return new AGXDAGToDAGISelLegacy(TM);
}

void AGXDAGToDAGISel::Select(SDNode *N) {
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;
  }
  switch (N->getOpcode()) {
  case AGXISD::CALL: {
    SDLoc DL(N);
    // Mesa's push_exec r0l, 2 is if_fcmp r0l, eq, 0.0, 0.0, 2.
    // Reserve the caller's masked lanes before RET changes its counters.
    SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
    SDValue Two = CurDAG->getTargetConstant(2, DL, MVT::i32);
    bool HasGlue = N->getOperand(N->getNumOperands() - 1).getValueType() ==
                   MVT::Glue;
    SmallVector<SDValue, 10> PushOps = {Zero, Zero, Zero, Zero,
                                        Zero, Zero, Two,  N->getOperand(0)};
    if (HasGlue)
      PushOps.push_back(N->getOperand(N->getNumOperands() - 1));
    SDNode *Push = CurDAG->getMachineNode(AGX::IF_FCMP, DL, MVT::Other,
                                          MVT::Glue, PushOps);
    // Target, the argument registers (implicit uses), register mask.
    SmallVector<SDValue, 16> CallOps = {N->getOperand(1)};
    for (unsigned I = 3, E = N->getNumOperands() - HasGlue; I != E; ++I)
      CallOps.push_back(N->getOperand(I));
    CallOps.push_back(N->getOperand(2));
    CallOps.push_back(SDValue(Push, 0));
    CallOps.push_back(SDValue(Push, 1));
    SDNode *Call = CurDAG->getMachineNode(AGX::AGX_CALL, DL, MVT::Other,
                                          MVT::Glue, CallOps);
    SDValue PopOps[] = {Zero, Two, SDValue(Call, 0), SDValue(Call, 1)};
    ReplaceNode(N, CurDAG->getMachineNode(AGX::POP_CALL_EXEC, DL, MVT::Other,
                                          MVT::Glue, PopOps));
    return;
  }
  case AGXISD::RET: {
    SDLoc DL(N);
    bool HasGlue = N->getOperand(N->getNumOperands() - 1).getValueType() ==
                   MVT::Glue;
    SmallVector<SDValue, 16> Ops = {CurDAG->getRegister(AGX::R1, MVT::i32)};
    for (unsigned I = 1, E = N->getNumOperands() - HasGlue; I != E; ++I)
      Ops.push_back(N->getOperand(I));
    Ops.push_back(N->getOperand(0));
    if (HasGlue)
      Ops.push_back(N->getOperand(N->getNumOperands() - 1));
    ReplaceNode(N, CurDAG->getMachineNode(AGX::AGX_RET, DL, MVT::Other, Ops));
    return;
  }
  case AGXISD::LOAD_V2:
  case AGXISD::LOAD_V3:
  case AGXISD::LOAD_V4:
    return selectVectorLoad(N, N->getNumValues() - 1);
  case AGXISD::LOAD_V2_I8:
  case AGXISD::LOAD_V4_I8:
    return selectByteVectorLoad(N, N->getNumValues() - 1);
  case AGXISD::STORE_V2_I8:
  case AGXISD::STORE_V4_I8:
    return selectByteVectorStore(N, N->getNumOperands() - 2);
  case AGXISD::STORE_V2:
  case AGXISD::STORE_V3:
  case AGXISD::STORE_V4:
    return selectVectorStore(N, N->getNumOperands() - 2);
  case AGXISD::TEXTURE_SAMPLE:
    return selectTextureSample(N);
  case ISD::BUILD_PAIR: {
    SDLoc DL(N);
    SDValue Ops[] = {
        CurDAG->getTargetConstant(AGX::GPR64RegClassID, DL, MVT::i32),
        N->getOperand(0), CurDAG->getTargetConstant(AGX::sub0, DL, MVT::i32),
        N->getOperand(1), CurDAG->getTargetConstant(AGX::sub1, DL, MVT::i32)};
    ReplaceNode(N, CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                          MVT::i64, Ops));
    return;
  }
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    // Shifts of extended 32-bit values by up to 4 are selected as iadd by
    // the generated matcher.
    if (N->getOpcode() == ISD::SHL &&
        (N->getOperand(0).getOpcode() == ISD::ZERO_EXTEND ||
         N->getOperand(0).getOpcode() == ISD::SIGN_EXTEND) &&
        isa<ConstantSDNode>(N->getOperand(1)) &&
        N->getConstantOperandVal(1) <= 4)
      break;
    if (trySelectShift64(N))
      return;
    break;
  case ISD::OR:
    if (trySelectFieldInsert(N))
      return;
    break;
  default:
    break;
  }
  SelectCode(N);
}

/// (x & ~(m << s)) | (c << s) with a field m of consecutive ones and a small
/// c: one bfi of c into x (setting a float's exponent field, as the float
/// division does), instead of two mov_imm, an and and an or.
bool AGXDAGToDAGISel::trySelectFieldInsert(SDNode *N) {
  if (N->getValueType(0) != MVT::i32)
    return false;
  for (unsigned I = 0; I < 2; ++I) {
    SDValue And = N->getOperand(I);
    auto *Ins = dyn_cast<ConstantSDNode>(N->getOperand(1 - I));
    if (!Ins || And.getOpcode() != ISD::AND || !And.hasOneUse())
      continue;
    auto *Keep = dyn_cast<ConstantSDNode>(And.getOperand(1));
    if (!Keep)
      continue;
    uint32_t Field = ~uint32_t(Keep->getZExtValue());
    uint32_t C = Ins->getZExtValue();
    unsigned Shift, Width;
    if (!isShiftedMask_32(Field, Shift, Width) || Width == 32 ||
        (C & ~Field) || (C >> Shift) > 0xff)
      return false;
    // Small masks are and-able as immediates; leave those to the matcher.
    if (Keep->getZExtValue() <= 0xff && C <= 0xff)
      return false;
    SDLoc DL(N);
    SDValue Zero = getI32Imm(0, DL);
    ReplaceNode(N, CurDAG->getMachineNode(
                       AGX::BFI_I32, DL, MVT::i32,
                       {Zero, And.getOperand(0), Zero, getI32Imm(C >> Shift, DL),
                        Zero, getI32Imm(Shift, DL), getI32Imm(Width, DL)}));
    return true;
  }
  return false;
}

static cl::opt<bool> Via16Addresses(
    "agx-via16-addresses", cl::init(true), cl::Hidden,
    cl::desc("4- and 8-byte device accesses at a 16-bit-unit index in the "
             "16-bit format, the index folded"));

// The i32 format and the masks of 2, 3 and 4 consecutive values.
static constexpr unsigned FmtI32 = 2;
static unsigned vectorMask(unsigned NumElts) { return (1u << NumElts) - 1; }

static unsigned tupleClass(unsigned NumElts) {
  return NumElts == 2   ? AGX::GPR32x2RegClassID
         : NumElts == 3 ? AGX::GPR96RegClassID
                        : AGX::GPR128RegClassID;
}

static const unsigned SubRegs[] = {AGX::sub0, AGX::sub1, AGX::sub2, AGX::sub3};

/// LOAD_V2/3/4: one device_load / threadgroup_load of the register tuple, and
/// its elements as subregisters.
void AGXDAGToDAGISel::selectVectorLoad(SDNode *N, unsigned NumElts) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  SDValue Chain = N->getOperand(0), Addr = N->getOperand(1);
  SDValue Mask = CurDAG->getTargetConstant(vectorMask(NumElts), DL, MVT::i32);
  SDValue Fmt = CurDAG->getTargetConstant(FmtI32, DL, MVT::i32);
  MachineSDNode *Load;
  if (M->getAddressSpace() == AGX::THREADGROUP_ADDRESS) {
    SDValue Base, Index;
    SelectLocalAddr32(Addr, Base, Index);
    unsigned Opc = NumElts == 2   ? AGX::LOCAL_LOAD_64
                   : NumElts == 3 ? AGX::LOCAL_LOAD_96
                                  : AGX::LOCAL_LOAD_128;
    Load = CurDAG->getMachineNode(Opc, DL, MVT::Untyped, MVT::Other,
                                  {Fmt, Mask, Base, Index, Chain});
  } else {
    SDValue Base, Index, Signed, Shift;
    unsigned Opc = NumElts == 2   ? AGX::DEVICE_LOAD_64
                   : NumElts == 3 ? AGX::DEVICE_LOAD_96
                                  : AGX::DEVICE_LOAD_128;
    if (NumElts == 2 && Via16Addresses &&
        SelectDeviceAddr32Via16(Addr, Base, Index, Signed, Shift)) {
      Opc = AGX::DEVICE_LOAD_64_H16;
      Fmt = CurDAG->getTargetConstant(1, DL, MVT::i32);
      Mask = CurDAG->getTargetConstant(vectorMask(4), DL, MVT::i32);
    } else {
      SelectDeviceAddr32(Addr, Base, Index, Signed, Shift);
    }
    SDValue G = CurDAG->getTargetConstant(0, DL, MVT::i32);
    Load = CurDAG->getMachineNode(
        Opc, DL, MVT::Untyped, MVT::Other,
        {G, Fmt, Mask, Base, Index, Signed, Shift, Chain});
  }
  CurDAG->setNodeMemRefs(Load, {M->getMemOperand()});
  for (unsigned I = 0; I < NumElts; ++I)
    ReplaceUses(SDValue(N, I), CurDAG->getTargetExtractSubreg(
                                   SubRegs[I], DL, MVT::i32, SDValue(Load, 0)));
  ReplaceUses(SDValue(N, NumElts), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(N);
}

/// LOAD_V2_I8 / LOAD_V4_I8: one device_load of the byte format into 2 or 4
/// consecutive 16-bit register halves (a 32-bit register, or a pair), byte k
/// zero-extended in half k.
void AGXDAGToDAGISel::selectByteVectorLoad(SDNode *N, unsigned NumElts) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  SDValue Chain = N->getOperand(0), Addr = N->getOperand(1);
  SDValue Base, Index, Signed, Shift;
  // 16-bit elements (the i16 format; -agx-element-aligned-vectors) or bytes.
  bool Halves = M->getMemoryVT().getScalarSizeInBits() == 16;
  if (Halves)
    SelectDeviceAddr16(Addr, Base, Index, Signed, Shift);
  else
    SelectDeviceAddr8(Addr, Base, Index, Signed, Shift);
  SDValue G = CurDAG->getTargetConstant(0, DL, MVT::i32);
  SDValue Fmt = CurDAG->getTargetConstant(Halves ? 1 : 0, DL, MVT::i32);
  SDValue Mask = CurDAG->getTargetConstant(vectorMask(NumElts), DL, MVT::i32);
  MachineSDNode *Load = CurDAG->getMachineNode(
      NumElts == 2 ? AGX::DEVICE_LOAD_32_H16 : AGX::DEVICE_LOAD_64_H16, DL,
      NumElts == 2 ? MVT::i32 : MVT::Untyped, MVT::Other,
      {G, Fmt, Mask, Base, Index, Signed, Shift, Chain});
  CurDAG->setNodeMemRefs(Load, {M->getMemOperand()});
  for (unsigned I = 0; I < NumElts; ++I) {
    SDValue Word = SDValue(Load, 0);
    if (NumElts == 4)
      Word = CurDAG->getTargetExtractSubreg(SubRegs[I / 2], DL, MVT::i32, Word);
    ReplaceUses(SDValue(N, I),
                CurDAG->getTargetExtractSubreg(I % 2 ? AGX::hi16 : AGX::lo16,
                                               DL, MVT::i16, Word));
  }
  ReplaceUses(SDValue(N, NumElts), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(N);
}

/// STORE_V2_I8 / STORE_V4_I8: the 16-bit values as the halves of a 32-bit
/// register or pair, then one device_store of the byte format (the low byte of
/// half k at byte k).
void AGXDAGToDAGISel::selectByteVectorStore(SDNode *N, unsigned NumElts) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  SDValue Chain = N->getOperand(0), Addr = N->getOperand(NumElts + 1);
  auto Word = [&](unsigned I) {
    SDValue Lo = CurDAG->getTargetConstant(AGX::lo16, DL, MVT::i32);
    SDValue Hi = CurDAG->getTargetConstant(AGX::hi16, DL, MVT::i32);
    SDValue Ops[] = {
        CurDAG->getTargetConstant(AGX::GPR32RegClassID, DL, MVT::i32),
        N->getOperand(1 + I), Lo, N->getOperand(2 + I), Hi};
    return SDValue(CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                          MVT::i32, Ops),
                   0);
  };
  SDValue Val = Word(0);
  if (NumElts == 4) {
    SDValue Ops[] = {
        CurDAG->getTargetConstant(AGX::GPR32x2RegClassID, DL, MVT::i32), Val,
        CurDAG->getTargetConstant(AGX::sub0, DL, MVT::i32), Word(2),
        CurDAG->getTargetConstant(AGX::sub1, DL, MVT::i32)};
    Val = SDValue(CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                         MVT::Untyped, Ops),
                  0);
  }
  SDValue Base, Index, Signed, Shift;
  SelectDeviceAddr8(Addr, Base, Index, Signed, Shift);
  SDValue G = CurDAG->getTargetConstant(0, DL, MVT::i32);
  SDValue Fmt = CurDAG->getTargetConstant(0, DL, MVT::i32); // i8
  SDValue Mask = CurDAG->getTargetConstant(vectorMask(NumElts), DL, MVT::i32);
  MachineSDNode *Store = CurDAG->getMachineNode(
      NumElts == 2 ? AGX::DEVICE_STORE_32_H16 : AGX::DEVICE_STORE_64_H16, DL,
      MVT::Other, {G, Fmt, Mask, Val, Base, Index, Signed, Shift, Chain});
  CurDAG->setNodeMemRefs(Store, {M->getMemOperand()});
  ReplaceNode(N, Store);
}

/// STORE_V2/3/4: the elements as one register tuple, then one device_store /
/// threadgroup_store.
void AGXDAGToDAGISel::selectVectorStore(SDNode *N, unsigned NumElts) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  SDValue Chain = N->getOperand(0), Addr = N->getOperand(NumElts + 1);
  SmallVector<SDValue, 9> Seq{
      CurDAG->getTargetConstant(tupleClass(NumElts), DL, MVT::i32)};
  for (unsigned I = 0; I < NumElts; ++I) {
    Seq.push_back(N->getOperand(1 + I));
    Seq.push_back(CurDAG->getTargetConstant(SubRegs[I], DL, MVT::i32));
  }
  SDValue Val(CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                     MVT::Untyped, Seq),
              0);
  SDValue Mask = CurDAG->getTargetConstant(vectorMask(NumElts), DL, MVT::i32);
  SDValue Fmt = CurDAG->getTargetConstant(FmtI32, DL, MVT::i32);
  MachineSDNode *Store;
  if (M->getAddressSpace() == AGX::THREADGROUP_ADDRESS) {
    SDValue Base, Index;
    SelectLocalAddr32(Addr, Base, Index);
    unsigned Opc = NumElts == 2   ? AGX::LOCAL_STORE_64
                   : NumElts == 3 ? AGX::LOCAL_STORE_96
                                  : AGX::LOCAL_STORE_128;
    Store = CurDAG->getMachineNode(Opc, DL, MVT::Other,
                                   {Fmt, Mask, Val, Base, Index, Chain});
  } else {
    SDValue Base, Index, Signed, Shift;
    unsigned Opc = NumElts == 2   ? AGX::DEVICE_STORE_64
                   : NumElts == 3 ? AGX::DEVICE_STORE_96
                                  : AGX::DEVICE_STORE_128;
    if (NumElts == 2 && Via16Addresses &&
        SelectDeviceAddr32Via16(Addr, Base, Index, Signed, Shift)) {
      Opc = AGX::DEVICE_STORE_64_H16;
      Fmt = CurDAG->getTargetConstant(1, DL, MVT::i32);
      Mask = CurDAG->getTargetConstant(vectorMask(4), DL, MVT::i32);
    } else {
      SelectDeviceAddr32(Addr, Base, Index, Signed, Shift);
    }
    SDValue G = CurDAG->getTargetConstant(0, DL, MVT::i32);
    Store = CurDAG->getMachineNode(
        Opc, DL, MVT::Other,
        {G, Fmt, Mask, Val, Base, Index, Signed, Shift, Chain});
  }
  CurDAG->setNodeMemRefs(Store, {M->getMemOperand()});
  ReplaceNode(N, Store);
}

/// TEXTURE_SAMPLE: one texture_sample of the four channels into a register
/// tuple, and the channels as its subregisters; 2D coordinates as a pair.
void AGXDAGToDAGISel::selectTextureSample(SDNode *N) {
  SDLoc DL(N);
  SDValue Chain = N->getOperand(0), T = N->getOperand(1), S = N->getOperand(2);
  unsigned Dim = N->getConstantOperandVal(5);
  SDValue C = N->getOperand(3);
  unsigned Opc = AGX::TEXTURE_SAMPLE_1D;
  if (Dim == 2) {
    SDValue Seq[] = {
        CurDAG->getTargetConstant(AGX::GPR32x2RegClassID, DL, MVT::i32), C,
        CurDAG->getTargetConstant(AGX::sub0, DL, MVT::i32), N->getOperand(4),
        CurDAG->getTargetConstant(AGX::sub1, DL, MVT::i32)};
    C = SDValue(CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                       MVT::Untyped, Seq),
                0);
    Opc = AGX::TEXTURE_SAMPLE_2D;
  }
  MachineSDNode *Sample = CurDAG->getMachineNode(
      Opc, DL, MVT::Untyped, MVT::Other,
      {CurDAG->getTargetConstant(0, DL, MVT::i32), T, S, C,
       CurDAG->getTargetConstant(Dim, DL, MVT::i32),
       CurDAG->getTargetConstant(vectorMask(4), DL, MVT::i32), Chain});
  for (unsigned I = 0; I < 4; ++I)
    ReplaceUses(SDValue(N, I), CurDAG->getTargetExtractSubreg(
                                   SubRegs[I], DL, MVT::i32, SDValue(Sample, 0)));
  ReplaceUses(SDValue(N, 4), SDValue(Sample, 1));
  CurDAG->RemoveDeadNode(N);
}

/// Integer immediates 0-255 are encoded in the instruction.
bool AGXDAGToDAGISel::SelectIntSrc(SDValue N, SDValue &Src) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    if (C->getZExtValue() <= 0xff) {
      Src = CurDAG->getTargetConstant(C->getZExtValue(), SDLoc(N),
                                      N.getValueType());
      return true;
    }
  }
  Src = N;
  return true;
}

/// A 32-bit integer source that is a 16-bit value sign or zero extended: the
/// 16-bit register (a half of a 32-bit one for x >> 16 and the low 16 bits of
/// x) read with or without .sx, instead of the extension.
bool AGXDAGToDAGISel::SelectExt16Src(SDValue N, SDValue &Src, SDValue &Mods) {
  SDLoc DL(N);
  auto Half = [&](SDValue X, unsigned SubReg) {
    return CurDAG->getTargetExtractSubreg(SubReg, DL, MVT::i16, X);
  };
  unsigned M = 0;
  switch (N.getOpcode()) {
  case ISD::SIGN_EXTEND:
    M = AGX::SRC_MOD_SX;
    [[fallthrough]];
  case ISD::ZERO_EXTEND:
    if (N.getOperand(0).getValueType() != MVT::i16)
      return false;
    Src = N.getOperand(0);
    break;
  case ISD::SIGN_EXTEND_INREG:
    if (cast<VTSDNode>(N.getOperand(1))->getVT() != MVT::i16)
      return false;
    M = AGX::SRC_MOD_SX;
    Src = Half(N.getOperand(0), AGX::lo16);
    break;
  case ISD::AND:
    if (!isa<ConstantSDNode>(N.getOperand(1)) ||
        N.getConstantOperandVal(1) != 0xffff)
      return false;
    Src = Half(N.getOperand(0), AGX::lo16);
    break;
  case ISD::SRA:
    M = AGX::SRC_MOD_SX;
    [[fallthrough]];
  case ISD::SRL:
    if (!isa<ConstantSDNode>(N.getOperand(1)) ||
        N.getConstantOperandVal(1) != 16)
      return false;
    Src = Half(N.getOperand(0), AGX::hi16);
    break;
  default:
    return false;
  }
  Mods = getI32Imm(M, DL);
  return true;
}

/// A 64-bit source that is a 16-bit value sign or zero extended: directly
/// (sext / zext of an i16), or a 32-bit such value extended the same way.
bool AGXDAGToDAGISel::SelectExt16Src64(SDValue N, SDValue &Src, SDValue &Mods) {
  unsigned Opc = N.getOpcode();
  // A pair with a zero high word: the low word zero-extended.
  if (Opc == ISD::BUILD_PAIR && isNullConstant(N.getOperand(1)))
    Opc = ISD::ZERO_EXTEND;
  else if (Opc != ISD::SIGN_EXTEND && Opc != ISD::ZERO_EXTEND)
    return false;
  SDValue X = N.getOperand(0);
  if (X.getValueType() == MVT::i16) {
    Src = X;
    Mods = getI32Imm(Opc == ISD::SIGN_EXTEND ? AGX::SRC_MOD_SX : 0, SDLoc(N));
    return true;
  }
  if (X.getValueType() != MVT::i32 || !SelectExt16Src(X, Src, Mods))
    return false;
  // The inner extension must be the outer one's kind (zext of a sign
  // extension is not a sign extension to 64 bits), except that a zero
  // extension sign-extended is still a zero extension.
  bool InnerSX = cast<ConstantSDNode>(Mods)->getZExtValue() & AGX::SRC_MOD_SX;
  return Opc == ISD::SIGN_EXTEND || !InnerSX;
}

/// Floating point sources fold neg and abs into modifiers, and constants
/// representable as 8-bit floats into the instruction.
bool AGXDAGToDAGISel::SelectFPSrc(SDValue N, SDValue &Src, SDValue &Mods) {
  SDLoc DL(N);
  unsigned M = 0;
  while (true) {
    if (N.getOpcode() == ISD::FNEG) {
      // A negation inside an absolute value changes nothing: fabs(fneg x) is
      // |x|, not -|x| (the modifiers apply abs, then neg). Folding it gave
      // `fcmpsel lt, x.abs.neg, min_normal` for neg.ftz.f16's flush test, so
      // every negated half became a signed zero (llama.cpp's q4_K / q5_K).
      if (!(M & AGX::SRC_MOD_ABS))
        M ^= AGX::SRC_MOD_NEG;
      N = N.getOperand(0);
    } else if (N.getOpcode() == ISD::FABS) {
      // |-x| = |x|, and an outer negation applies after the absolute value.
      M |= AGX::SRC_MOD_ABS;
      N = N.getOperand(0);
    } else {
      break;
    }
  }

  if (auto *C = dyn_cast<ConstantFPSDNode>(N)) {
    APFloat V = C->getValueAPF();
    if (M & AGX::SRC_MOD_ABS)
      V.clearSign();
    if (M & AGX::SRC_MOD_NEG)
      V.changeSign();
    bool LosesInfo = false;
    V.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
              &LosesInfo);
    if (!LosesInfo) {
      if (std::optional<unsigned> Enc = AGX::encodeFPImm8(V.convertToDouble())) {
        Src = CurDAG->getTargetConstant(*Enc, DL, MVT::i32);
        Mods = getI32Imm(0, DL);
        return true;
      }
    }
  }

  Src = N;
  Mods = getI32Imm(M, DL);
  return true;
}

/// 64-bit shifts by a constant: shift both halves, combining them with a
/// funnel shift (extr).
bool AGXDAGToDAGISel::trySelectShift64(SDNode *N) {
  auto *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!C || N->getValueType(0) != MVT::i64)
    return false;
  SDLoc DL(N);
  unsigned Opc = N->getOpcode();
  uint64_t S = C->getZExtValue() & 63;
  SDValue X = N->getOperand(0);
  SDValue Lo = CurDAG->getTargetExtractSubreg(AGX::sub0, DL, MVT::i32, X);
  SDValue Hi = CurDAG->getTargetExtractSubreg(AGX::sub1, DL, MVT::i32, X);
  SDValue Zero = getI32Imm(0, DL);

  // Machine node builders mirroring the 32-bit shift patterns.
  auto Shl = [&](SDValue V, uint64_t A) {
    return SDValue(CurDAG->getMachineNode(AGX::BFI_I32, DL, MVT::i32,
                                          {Zero, Zero, Zero, V, Zero,
                                           getI32Imm(A, DL), Zero}),
                   0);
  };
  auto Lsr = [&](SDValue V, uint64_t A) {
    return SDValue(CurDAG->getMachineNode(AGX::BFEIL_I32, DL, MVT::i32,
                                          {Zero, Zero, Zero, V, Zero,
                                           getI32Imm(A, DL), Zero}),
                   0);
  };
  auto Asr = [&](SDValue V, uint64_t A) {
    return SDValue(CurDAG->getMachineNode(AGX::ASR_I32, DL, MVT::i32,
                                          {Zero, V, Zero, getI32Imm(A, DL)}),
                   0);
  };
  // (B:A) >> A, the low 32 bits.
  auto Extr = [&](SDValue A, SDValue B, uint64_t Amt) {
    return SDValue(CurDAG->getMachineNode(AGX::EXTR_I32, DL, MVT::i32,
                                          {Zero, A, Zero, B, Zero,
                                           getI32Imm(Amt, DL), Zero}),
                   0);
  };
  auto Imm0 = [&]() {
    return SDValue(
        CurDAG->getMachineNode(AGX::MOV_IMM32, DL, MVT::i32, Zero), 0);
  };

  SDValue NewLo, NewHi;
  if (S == 0) {
    ReplaceNode(N, X.getNode());
    return true;
  }
  switch (Opc) {
  case ISD::SHL:
    if (S >= 32) {
      NewLo = Imm0();
      NewHi = Shl(Lo, S - 32);
    } else {
      NewLo = Shl(Lo, S);
      NewHi = Extr(Lo, Hi, 32 - S);
    }
    break;
  case ISD::SRL:
    if (S >= 32) {
      NewLo = Lsr(Hi, S - 32);
      NewHi = Imm0();
    } else {
      NewLo = Extr(Lo, Hi, S);
      NewHi = Lsr(Hi, S);
    }
    break;
  case ISD::SRA:
    if (S >= 32) {
      NewLo = Asr(Hi, S - 32);
      NewHi = Asr(Hi, 31);
    } else {
      NewLo = Extr(Lo, Hi, S);
      NewHi = Asr(Hi, S);
    }
    break;
  default:
    return false;
  }
  SDValue Ops[] = {
      CurDAG->getTargetConstant(AGX::GPR64RegClassID, DL, MVT::i32), NewLo,
      CurDAG->getTargetConstant(AGX::sub0, DL, MVT::i32), NewHi,
      CurDAG->getTargetConstant(AGX::sub1, DL, MVT::i32)};
  ReplaceNode(N, CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                        MVT::i64, Ops));
  return true;
}

/// Device addresses are base + (index << (log2(element size) + shift)), with
/// a 32-bit index that is zero or sign extended, or a signed 16-bit
/// immediate index.
bool AGXDAGToDAGISel::SelectDeviceAddr(SDValue Addr, unsigned ElemLog2,
                                       SDValue &Base, SDValue &Index,
                                       SDValue &Signed, SDValue &Shift) {
  SDLoc DL(Addr);
  Signed = getI32Imm(0, DL);
  Shift = getI32Imm(0, DL);

  if (Addr.getOpcode() == ISD::ADD ||
      (Addr.getOpcode() == ISD::OR && Addr->getFlags().hasDisjoint())) {
    for (unsigned I = 0; I < 2; ++I) {
      SDValue B = Addr.getOperand(I);
      SDValue O = Addr.getOperand(1 - I);

      // Constant byte offset.
      if (auto *C = dyn_cast<ConstantSDNode>(O)) {
        int64_t Off = C->getSExtValue();
        if (Off % (int64_t(1) << ElemLog2) == 0 &&
            isInt<16>(Off >> ElemLog2)) {
          Base = B;
          Index = CurDAG->getSignedTargetConstant(Off >> ElemLog2, DL, MVT::i32);
          return true;
        }
        continue;
      }

      // Scaled, extended 32-bit index.
      unsigned Scale = 0;
      if (O.getOpcode() == ISD::SHL) {
        auto *C = dyn_cast<ConstantSDNode>(O.getOperand(1));
        if (!C)
          continue;
        Scale = C->getZExtValue();
        O = O.getOperand(0);
      } else if (O.getOpcode() == ISD::MUL) {
        auto *C = dyn_cast<ConstantSDNode>(O.getOperand(1));
        if (!C || !isPowerOf2_64(C->getZExtValue()))
          continue;
        Scale = Log2_64(C->getZExtValue());
        O = O.getOperand(0);
      }
      // The shift field is 0 to 2: applegpu notes that 3 may act as 2.
      if (Scale < ElemLog2 || Scale > ElemLog2 + 2)
        continue;
      bool IsSExt = O.getOpcode() == ISD::SIGN_EXTEND;
      if (!IsSExt && O.getOpcode() != ISD::ZERO_EXTEND)
        continue;
      if (O.getOperand(0).getValueType() != MVT::i32)
        continue;
      Base = B;
      Index = O.getOperand(0);
      // The index signedness field is 1 for zero extension.
      Signed = getI32Imm(!IsSExt, DL);
      Shift = getI32Imm(Scale - ElemLog2, DL);
      return true;
    }
  }

  Base = Addr;
  Index = CurDAG->getTargetConstant(0, DL, MVT::i32);
  return true;
}

static cl::opt<bool> AtomicIndices(
    "agx-atomic-indices", cl::init(true), cl::Hidden,
    cl::desc("Device atomics at base + 4 * zext(i32) take the index register"));

/// Device atomics take the base and index registers as loads do (the
/// simulator's coverage recordings on the M2 Max: atomic add, 0, r14, u8_u9,
/// r5, unsigned, r6), the index counting 4 bytes. Only the recorded form, a
/// zero-extended 32-bit register index unshifted; otherwise the full address
/// in the base register (a 64-bit add and, for a uniform base, its copy to
/// registers).
bool AGXDAGToDAGISel::SelectAtomicAddr(SDValue Addr, SDValue &Base,
                                       SDValue &Index, SDValue &Signed) {
  SDLoc DL(Addr);
  SDValue B, I, S, Sh;
  if (AtomicIndices && SelectDeviceAddr(Addr, 2, B, I, S, Sh) && B != Addr &&
      !isa<ConstantSDNode>(I) && cast<ConstantSDNode>(Sh)->isZero() &&
      cast<ConstantSDNode>(S)->getZExtValue() == 1) {
    Base = B;
    Index = I;
    Signed = S;
    return true;
  }
  Base = Addr;
  Index = CurDAG->getTargetConstant(0, DL, MVT::i32);
  Signed = getI32Imm(0, DL);
  return true;
}

/// A 4- or 8-byte access whose index is in 16-bit units (base + 2 * i, a
/// half or bfloat vector at an element index) does not fold into the 32-bit
/// format's index, which would compute the 64-bit address; the 16-bit format
/// (xy / xyzw into the same register halves) takes it. Only where the 32-bit
/// form would not fold.
bool AGXDAGToDAGISel::SelectDeviceAddr32Via16(SDValue Addr, SDValue &Base,
                                              SDValue &Index, SDValue &Signed,
                                              SDValue &Shift) {
  SelectDeviceAddr32(Addr, Base, Index, Signed, Shift);
  if (Base != Addr)
    return false;
  SelectDeviceAddr16(Addr, Base, Index, Signed, Shift);
  return Base != Addr;
}

/// Threadgroup addresses are 16-bit byte offsets in the base register. The
/// index counts four-element vectors, Unit bytes each (the address is base +
/// 4 * element size * index), so constant offsets that are multiples of Unit
/// go in the immediate index.
bool AGXDAGToDAGISel::SelectLocalAddr(SDValue Addr, SDValue &Base,
                                      SDValue &Index, unsigned Unit) {
  SDLoc DL(Addr);
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t Offset = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    SDValue B = Addr.getOperand(0);
    if (Offset > 0 && Offset % Unit == 0 && isUInt<15>(Offset / Unit)) {
      Base = isNullConstant(B)
                 ? CurDAG->getTargetConstant(0, DL, MVT::i16)
                 : CurDAG->getTargetExtractSubreg(AGX::lo16, DL, MVT::i16, B);
      Index = CurDAG->getTargetConstant(Offset / Unit, DL, MVT::i16);
      return true;
    }
    // base + (Offset % Unit) is already computed for another access: reuse
    // it with the rest of the offset in the index.
    if (Offset > int64_t(Unit) && isUInt<15>(Offset / Unit)) {
      SDValue Rem = CurDAG->getConstant(Offset % Unit, DL, MVT::i32);
      if (SDNode *N = CurDAG->getNodeIfExists(
              ISD::ADD, CurDAG->getVTList(MVT::i32), {B, Rem})) {
        Base = CurDAG->getTargetExtractSubreg(AGX::lo16, DL, MVT::i16,
                                              SDValue(N, 0));
        Index = CurDAG->getTargetConstant(Offset / Unit, DL, MVT::i16);
        return true;
      }
    }
  }
  if (isNullConstant(Addr))
    Base = CurDAG->getTargetConstant(0, DL, MVT::i16);
  else
    Base = CurDAG->getTargetExtractSubreg(AGX::lo16, DL, MVT::i16, Addr);
  Index = CurDAG->getTargetConstant(0, DL, MVT::i16);
  return true;
}

/// Stack addresses are byte offsets: a frame index (replaced by its offset
/// after frame lowering), a 16-bit immediate or a register.
bool AGXDAGToDAGISel::SelectStackAddr(SDValue Addr, SDValue &Offset) {
  SDLoc DL(Addr);
  if (auto *FI = dyn_cast<FrameIndexSDNode>(Addr)) {
    Offset = CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
    return true;
  }
  if (auto *C = dyn_cast<ConstantSDNode>(Addr)) {
    if (isInt<16>(C->getSExtValue())) {
      Offset = CurDAG->getTargetConstant(C->getSExtValue(), DL, MVT::i32);
      return true;
    }
  }
  Offset = Addr;
  return true;
}
