//===-- AGXStrictFP.cpp - IEEE binary32 and canonical NaNs in software ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The 32-bit floating point ALU flushes denormal operands and results to zero,
// division and square root are not correctly rounded, and NaN results have the
// hardware's bit patterns. Two opt-in subtarget features implement the
// missing behavior in software:
//
//  * ieee-fp32: IEEE 754 binary32 arithmetic in functions whose denormal mode
//    for float is IEEE. Additions, multiplications and fused multiply-adds
//    keep the hardware instruction and recompute the result with integer
//    arithmetic only where the hardware result can be wrong (denormal
//    operands, or a result close to the denormal range). Comparisons,
//    minimum and maximum compare denormals as integers, and rounding
//    functions and the approximate functions (rcp, rsqrt, log2, exp2) treat
//    denormal operands or results explicitly. Division and square root are
//    correctly rounded: refined hardware estimates are corrected with exact
//    remainders for normal operands and results, and computed with integer
//    arithmetic otherwise.
//
//  * canonical-nan: NaN results of floating point arithmetic are the
//    canonical NaN of PTX, 0x7fffffff (0x7fff for half and bfloat).
//
// Exact results are represented as sign * M * 2^E with a 64-bit integer M.
// convert() rounds such a value to nearest even, producing denormals,
// normals or infinities as IEEE 754 requires; values that are not exact keep
// a sticky bit in the low bit of M.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXSoftF64.h"
#include "AGXSubtarget.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "agx-strict-fp"

namespace {

class AGXStrictFP : public FunctionPass {
public:
  static char ID;
  AGXStrictFP() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Strict FP"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

/// A binary floating point format.
struct Format {
  unsigned Precision; // significand bits, including the implicit one
  int Bias;
  unsigned Width;
  int minLSB() const { return 2 - Bias - int(Precision); }
  uint32_t fracMask() const { return (1u << (Precision - 1)) - 1; }
  uint32_t expMask() const { return uint32_t(2 * Bias + 1); }
  uint32_t infBits() const { return expMask() << (Precision - 1); }
  uint32_t signBit() const { return 1u << (Width - 1); }
};
constexpr Format Binary32 = {24, 127, 32};
constexpr Format Binary16 = {11, 15, 16};

/// Rounding directions, numbered like FLT_ROUNDS and llvm.get.rounding.
enum RoundingCode : uint32_t {
  RoundTowardZero = 0,
  RoundNearestEven = 1,
  RoundUp = 2,
  RoundDown = 3,
};

/// sign * M * 2^E: Sign is i1, M is i64 and E is i32.
struct Term {
  Value *Sign;
  Value *M;
  Value *E;
};

/// Builds the integer implementations at the builder's insertion point.
static cl::opt<bool> ClassKey(
    "agx-ieee-class-key", cl::init(true), cl::Hidden,
    cl::desc("IEEE binary32 denormal and zero tests on 0xfffffe - 2|x| (one "
             "isub with a shifted operand, shared), not on |x| from an and"));

class SoftFloat {
  IRBuilderBase &B;
  Type *I1, *I32, *I64, *F32;

public:
  /// The rounding direction (a RoundingCode, constant or not) of the
  /// results built by convert().
  Value *Rounding;

  explicit SoftFloat(IRBuilderBase &B)
      : B(B), I1(B.getInt1Ty()), I32(B.getInt32Ty()), I64(B.getInt64Ty()),
        F32(B.getFloatTy()), Rounding(B.getInt32(RoundNearestEven)) {}

  Value *roundingIs(RoundingCode C) {
    return B.CreateICmpEQ(Rounding, B.getInt32(C));
  }

  Value *c32(uint32_t V) { return B.getInt32(V); }
  Value *c64(uint64_t V) { return B.getInt64(V); }
  Value *bits(Value *F) { return B.CreateBitCast(F, I32); }
  Value *flt(Value *I) { return B.CreateBitCast(I, F32); }
  Value *fconst(float V) { return ConstantFP::get(F32, V); }

  Value *umin(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::umin, A, X);
  }
  Value *smax(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::smax, A, X);
  }
  /// Bit length of a 64-bit value, as i32 (0 for 0).
  Value *bitLength(Value *M) {
    Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, M, B.getFalse());
    return B.CreateTrunc(B.CreateSub(c64(64), Lz), I32);
  }
  /// M << S and M >> S for 0 <= S; amounts of 64 or more give 0.
  Value *shl(Value *M, Value *S) {
    Value *R = B.CreateShl(M, B.CreateZExt(umin(S, c32(63)), I64));
    return B.CreateSelect(B.CreateICmpUGT(S, c32(63)), c64(0), R);
  }
  Value *lshr(Value *M, Value *S) {
    Value *R = B.CreateLShr(M, B.CreateZExt(umin(S, c32(63)), I64));
    return B.CreateSelect(B.CreateICmpUGT(S, c32(63)), c64(0), R);
  }
  /// M >> S with the bits shifted out ORed into the low bit, for 0 <= S.
  Value *lshrSticky(Value *M, Value *S) {
    Value *Q = lshr(M, S);
    Value *Lost = B.CreateICmpNE(B.CreateShl(Q, B.CreateZExt(umin(S, c32(63)), I64)), M);
    Lost = B.CreateOr(Lost, B.CreateAnd(B.CreateICmpUGT(S, c32(63)),
                                        B.CreateICmpNE(M, c64(0))));
    return B.CreateOr(Q, B.CreateZExt(Lost, I64));
  }

  /// The exponent field is 0: zero or denormal.
  Value *expZero(Value *Bits) {
    return B.CreateICmpEQ(B.CreateAnd(Bits, c32(0x7f800000)), c32(0));
  }
  /// 0xfffffe - 2 |x| (mod 2^32), from the bits of x: one isub with a
  /// shifted operand, which the denormal and zero tests share.
  Value *classKey(Value *Bits) {
    return B.CreateSub(c32(0xfffffe), B.CreateShl(Bits, 1));
  }
  Value *isDenormal(Value *Bits) {
    // 2 |x| in [2, 0xfffffe] (the low bit of 2 |x| is 0)
    if (ClassKey)
      return B.CreateICmpULT(classKey(Bits), c32(0xfffffd));
    Value *Abs = B.CreateAnd(Bits, c32(0x7fffffff));
    return B.CreateICmpULT(B.CreateSub(Abs, c32(1)), c32(0x7fffff));
  }
  Value *isZero(Value *Bits) {
    if (ClassKey)
      return B.CreateICmpEQ(classKey(Bits), c32(0xfffffe));
    return B.CreateICmpEQ(B.CreateAnd(Bits, c32(0x7fffffff)), c32(0));
  }
  Value *isInfOrNaN(Value *Bits) {
    return B.CreateICmpEQ(B.CreateAnd(Bits, c32(0x7f800000)), c32(0x7f800000));
  }
  Value *signBit(Value *Bits) { return B.CreateICmpSLT(Bits, c32(0)); }

  /// A finite value (bits in an i32) as sign * M * 2^E with M <
  /// 2^Precision. With Normalize, the most significant bit of a nonzero M is
  /// bit Precision - 1.
  Term unpack(Value *Bits, bool Normalize, Format Fmt = Binary32) {
    unsigned P = Fmt.Precision;
    Value *Exp = B.CreateAnd(B.CreateLShr(Bits, P - 1), c32(Fmt.expMask()));
    Value *Frac = B.CreateAnd(Bits, c32(Fmt.fracMask()));
    Value *Den = B.CreateICmpEQ(Exp, c32(0));
    Value *M = B.CreateSelect(Den, Frac, B.CreateOr(Frac, c32(1u << (P - 1))));
    Value *E = B.CreateSub(B.CreateSelect(Den, c32(1), Exp),
                           c32(Fmt.Bias + P - 1));
    if (Normalize) {
      Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, M, B.getFalse());
      Value *Sh = B.CreateSelect(B.CreateICmpEQ(M, c32(0)), c32(0),
                                 B.CreateSub(Lz, c32(32 - P)));
      M = B.CreateShl(M, Sh);
      E = B.CreateSub(E, Sh);
    }
    Value *Sign = B.CreateICmpNE(B.CreateAnd(Bits, c32(Fmt.signBit())), c32(0));
    return {Sign, B.CreateZExt(M, I64), E};
  }

  /// Round sign * M * 2^E in the direction Rounding: the bits of the result
  /// in Fmt (in an i32).
  Value *convert(const Term &T, Format Fmt = Binary32) {
    unsigned P = Fmt.Precision;
    Value *L = bitLength(T.M);
    // Shift so that P significant bits remain, or fewer for denormals.
    Value *S = smax(B.CreateSub(L, c32(P)), B.CreateSub(c32(Fmt.minLSB()), T.E));
    Value *Right = B.CreateICmpSGT(S, c32(0));
    Value *K = B.CreateSelect(Right, umin(S, c32(64)), c32(0));
    Value *Q = lshr(T.M, K);
    Value *Rem = B.CreateSub(T.M, shl(Q, K));
    Value *Half = shl(c64(1), B.CreateSub(smax(K, c32(1)), c32(1)));
    Value *Inexact = B.CreateICmpNE(Rem, c64(0));
    Value *Nearest = B.CreateOr(
        B.CreateICmpUGT(Rem, Half),
        B.CreateAnd(B.CreateICmpEQ(Rem, Half),
                    B.CreateICmpNE(B.CreateAnd(Q, c64(1)), c64(0))));
    Value *Up = B.CreateSelect(
        roundingIs(RoundNearestEven), Nearest,
        B.CreateSelect(roundingIs(RoundUp), B.CreateAnd(Inexact, B.CreateNot(T.Sign)),
                       B.CreateSelect(roundingIs(RoundDown),
                                      B.CreateAnd(Inexact, T.Sign), B.getFalse())));
    Up = B.CreateAnd(Up, Right);
    Value *Rounded = B.CreateAdd(Q, B.CreateZExt(Up, I64));
    Value *Left = shl(T.M, B.CreateSelect(Right, c32(0), B.CreateNeg(S)));
    Value *Mant = B.CreateTrunc(B.CreateSelect(Right, Rounded, Left), I32);
    // The exponent field minus one: the implicit bit of Mant adds the one,
    // and a rounding carry into bit P increments the exponent.
    Value *Biased = B.CreateAdd(B.CreateAdd(T.E, S), c32(Fmt.Bias + P - 2));
    Value *Bits = B.CreateAdd(B.CreateShl(Biased, P - 1), Mant);
    Value *Over = B.CreateOr(B.CreateICmpSGT(Biased, c32(2 * Fmt.Bias - 1)),
                             B.CreateICmpUGE(Bits, c32(Fmt.infBits())));
    // Overflows give infinity, or the largest finite value when rounding
    // toward zero or away from the sign's infinity.
    Value *Max = c32(Fmt.infBits() - 1), *Inf = c32(Fmt.infBits());
    Value *OverBits = B.CreateSelect(
        roundingIs(RoundTowardZero), Max,
        B.CreateSelect(roundingIs(RoundUp), B.CreateSelect(T.Sign, Max, Inf),
                       B.CreateSelect(roundingIs(RoundDown),
                                      B.CreateSelect(T.Sign, Inf, Max), Inf)));
    Bits = B.CreateSelect(Over, OverBits, Bits);
    Bits = B.CreateSelect(B.CreateICmpEQ(T.M, c64(0)), c32(0), Bits);
    return B.CreateOr(Bits, B.CreateSelect(T.Sign, c32(Fmt.signBit()), c32(0)));
  }

  Term mul(const Term &X, const Term &Y) {
    return {B.CreateXor(X.Sign, Y.Sign), B.CreateMul(X.M, Y.M),
            B.CreateAdd(X.E, Y.E)};
  }

  /// X + Y, exact or with a sticky low bit.
  Term add(const Term &X, const Term &Y) {
    Value *TopX = B.CreateAdd(bitLength(X.M), X.E);
    Value *TopY = B.CreateAdd(bitLength(Y.M), Y.E);
    Value *Swap = B.CreateICmpSGT(TopY, TopX);
    Value *SB = B.CreateSelect(Swap, Y.Sign, X.Sign);
    Value *MB = B.CreateSelect(Swap, Y.M, X.M);
    Value *EB = B.CreateSelect(Swap, Y.E, X.E);
    Value *SS = B.CreateSelect(Swap, X.Sign, Y.Sign);
    Value *MS = B.CreateSelect(Swap, X.M, Y.M);
    Value *ES = B.CreateSelect(Swap, X.E, Y.E);
    // The larger term with its most significant bit at bit 61.
    Value *Up = B.CreateSub(c32(62), bitLength(MB));
    MB = shl(MB, Up);
    EB = B.CreateSub(EB, Up);
    Value *D = B.CreateSub(EB, ES);
    Value *Aligned = B.CreateSelect(B.CreateICmpSGT(D, c32(0)),
                                    lshrSticky(MS, D),
                                    shl(MS, B.CreateNeg(D)));
    Value *Same = B.CreateICmpEQ(SB, SS);
    Value *Sum = B.CreateSelect(Same, B.CreateAdd(MB, Aligned),
                                B.CreateSub(MB, Aligned));
    Value *Neg = B.CreateICmpSLT(Sum, c64(0));
    Value *M = B.CreateSelect(Neg, B.CreateNeg(Sum), Sum);
    // An exact cancellation is +0, or -0 when rounding down.
    Value *Sign = B.CreateSelect(B.CreateICmpNE(M, c64(0)), B.CreateXor(SB, Neg),
                                 roundingIs(RoundDown));
    // Zero terms: the other term; the sum of two zeros is negative if both
    // are, or (rounding down) if either is.
    Value *XZero = B.CreateICmpEQ(X.M, c64(0));
    Value *YZero = B.CreateICmpEQ(Y.M, c64(0));
    Value *BothZero = B.CreateAnd(XZero, YZero);
    Value *ZeroSign = B.CreateSelect(roundingIs(RoundDown),
                                     B.CreateOr(X.Sign, Y.Sign),
                                     B.CreateAnd(X.Sign, Y.Sign));
    Sign = B.CreateSelect(BothZero, ZeroSign,
                          B.CreateSelect(XZero, Y.Sign,
                                         B.CreateSelect(YZero, X.Sign, Sign)));
    M = B.CreateSelect(XZero, Y.M, B.CreateSelect(YZero, X.M, M));
    Value *E = B.CreateSelect(XZero, Y.E, B.CreateSelect(YZero, X.E, EB));
    return {Sign, M, E};
  }

  /// A denormal operand replaced by a normal value of the same sign, for the
  /// hardware operations on infinities and NaNs.
  Value *nonzeroNormal(Value *X) {
    Value *Bits = bits(X);
    Value *One = B.CreateOr(B.CreateAnd(Bits, c32(0x80000000)), c32(0x3f800000));
    return flt(B.CreateSelect(isDenormal(Bits), One, Bits));
  }

  /// X * 2^64 for |X| < 2^-62, exact (denormals included).
  Value *scaleUp64(Value *X) {
    Value *Bits = bits(X);
    Value *Frac = B.CreateAnd(Bits, c32(0x7fffff));
    // A denormal is Frac * 2^-149, so X * 2^64 = Frac * 2^-85.
    Value *Den = B.CreateFMul(B.CreateUIToFP(Frac, F32), fconst(0x1p-85f));
    Den = flt(B.CreateOr(bits(Den), B.CreateAnd(Bits, c32(0x80000000))));
    return B.CreateSelect(isDenormal(Bits), Den,
                          B.CreateFMul(X, fconst(0x1p64f)));
  }

  Value *fma(Value *X, Value *Y, Value *Z) {
    return B.CreateIntrinsic(Intrinsic::fma, {F32}, {X, Y, Z});
  }
  /// The neighbors of a positive normal value.
  Value *nextUp(Value *X) { return flt(B.CreateAdd(bits(X), c32(1))); }
  Value *nextDown(Value *X) { return flt(B.CreateSub(bits(X), c32(1))); }

  /// Q, the correctly rounded positive magnitude of a result with sign Neg,
  /// rounded in the direction Rounding instead: Err has the sign of the exact
  /// magnitude minus Q.
  Value *adjustMagnitude(Value *Q, Value *Err, Value *Neg) {
    Value *Above = B.CreateFCmpOGT(Err, fconst(0.0f));
    Value *Below = B.CreateFCmpOLT(Err, fconst(0.0f));
    // Rounding up a positive or down a negative result increases the
    // magnitude; toward zero, down a positive or up a negative one decreases
    // it.
    Value *Away = B.CreateOr(B.CreateAnd(roundingIs(RoundUp), B.CreateNot(Neg)),
                             B.CreateAnd(roundingIs(RoundDown), Neg));
    Value *Toward = B.CreateOr(
        roundingIs(RoundTowardZero),
        B.CreateOr(B.CreateAnd(roundingIs(RoundUp), Neg),
                   B.CreateAnd(roundingIs(RoundDown), B.CreateNot(Neg))));
    Value *R = B.CreateSelect(B.CreateAnd(Away, Above), nextUp(Q), Q);
    return B.CreateSelect(B.CreateAnd(Toward, Below), nextDown(Q), R);
  }

  /// The magnitude of S, rounded to nearest even, rounded in the direction
  /// Rounding instead, with the sign of S: Err is the exact result minus S.
  /// (A unit on the bits of S is a unit on its magnitude, the sign kept, and
  /// the error with the sign of S applied is that of the exact magnitude
  /// minus |S|: no magnitude, no sign put back.)
  Value *adjust(Value *S, Value *Err) {
    Value *SB = bits(S);
    Value *Neg = B.CreateICmpSLT(SB, c32(0));
    // Up or down, known: a unit up in value is +1 on the bits of a positive
    // S and -1 on a negative one (the reverse down), where the exact value
    // is beyond S in that direction.
    if (auto *RC = dyn_cast<ConstantInt>(Rounding);
        RC &&
        (RC->getZExtValue() == RoundUp || RC->getZExtValue() == RoundDown)) {
      bool Up = RC->getZExtValue() == RoundUp;
      Value *Unit = B.CreateSelect(Neg, c32(Up ? -1 : 1), c32(Up ? 1 : -1));
      Value *Beyond = Up ? B.CreateFCmpOGT(Err, fconst(0.0f))
                         : B.CreateFCmpOLT(Err, fconst(0.0f));
      return flt(B.CreateAdd(SB, B.CreateSelect(Beyond, Unit, c32(0))));
    }
    Value *ErrM = flt(B.CreateXor(bits(Err), B.CreateAnd(SB, c32(0x80000000))));
    return adjustMagnitude(S, ErrM, Neg);
  }

  /// X + Y (X - Y) rounded in the direction Rounding, using the hardware: the
  /// error of the sum rounded to nearest even is exact (TwoSum) when the
  /// operands are zero or at least 2^-102 (all values are then multiples of
  /// 2^-125, so none is denormal) and below 2^126 (all values are then
  /// below 1.5 * 2^127, so the sum is finite). Returns the result and
  /// whether it is valid.
  std::pair<Value *, Value *> fastAdd(Value *X, Value *Y, bool Sub) {
    if (Sub)
      Y = B.CreateFNeg(Y);
    Value *S = B.CreateFAdd(X, Y);
    Value *BB = B.CreateFSub(S, X);
    Value *Err = B.CreateFAdd(B.CreateFSub(X, B.CreateFSub(S, BB)), B.CreateFSub(Y, BB));
    // Both operands zero or in [2^-102, 2^126), at once: the smaller
    // magnitude less one (a zero wraps to the largest) and the larger one.
    Value *MX = B.CreateAnd(bits(X), c32(0x7fffffff));
    Value *MY = B.CreateAnd(bits(Y), c32(0x7fffffff));
    Value *Small = B.CreateBinaryIntrinsic(
        Intrinsic::umin, B.CreateSub(MX, c32(1)), B.CreateSub(MY, c32(1)));
    Value *Large = B.CreateBinaryIntrinsic(Intrinsic::umax, MX, MY);
    Value *Valid = B.CreateAnd(B.CreateICmpUGE(Small, c32(0x0c800000 - 1)),
                               B.CreateICmpULT(Large, c32(0x7e800000)));
    Value *SM = B.CreateAnd(bits(S), c32(0x7fffffff));
    // An exact zero sum is -0 rounding down, unless both operands are +0.
    Value *BothPlusZero = B.CreateICmpEQ(B.CreateOr(bits(X), bits(Y)), c32(0));
    Value *DownZero = B.CreateAnd(B.CreateICmpEQ(SM, c32(0)), roundingIs(RoundDown));
    Value *Zero = flt(B.CreateSelect(BothPlusZero, c32(0), c32(0x80000000)));
    return {B.CreateSelect(DownZero, Zero, adjust(S, Err)), Valid};
  }

  /// X * Y rounded in the direction Rounding, using the hardware: the error
  /// of the product rounded to nearest even, fma(x, y, -p), is exact when the
  /// operands are normal with exponents adding up to at least -80 (it is a
  /// multiple of 2^-126) and the product is finite.
  std::pair<Value *, Value *> fastMul(Value *X, Value *Y) {
    Value *P = B.CreateFMul(X, Y);
    Value *Err = fma(X, Y, B.CreateFNeg(P));
    Value *EX = B.CreateAnd(B.CreateLShr(bits(X), 23), c32(0xff));
    Value *EY = B.CreateAnd(B.CreateLShr(bits(Y), 23), c32(0xff));
    auto Normal = [&](Value *E) {
      return B.CreateICmpULT(B.CreateSub(E, c32(1)), c32(254));
    };
    Value *PM = B.CreateAnd(bits(P), c32(0x7fffffff));
    Value *Valid = B.CreateAnd(B.CreateAnd(Normal(EX), Normal(EY)),
                               B.CreateICmpUGE(B.CreateAdd(EX, EY), c32(254 - 80)));
    Valid = B.CreateAnd(Valid, B.CreateICmpULT(PM, c32(0x7f800000)));
    // A zero operand and a zero or normal one: the product is an exact zero.
    auto ZeroOrNormal = [&](Value *V, Value *E) {
      return B.CreateOr(isZero(bits(V)), Normal(E));
    };
    Value *ZeroProduct = B.CreateAnd(B.CreateOr(isZero(bits(X)), isZero(bits(Y))),
                                     B.CreateAnd(ZeroOrNormal(X, EX), ZeroOrNormal(Y, EY)));
    return {B.CreateSelect(ZeroProduct, P, adjust(P, Err)), B.CreateOr(Valid, ZeroProduct)};
  }

  /// X * Y + Z rounded in the direction Rounding, using the hardware: the
  /// fma rounded to nearest even, R, and the sign of its exact error, from
  /// Boldo and Muller's ErrFma ("Exact and Approximated Error of the FMA",
  /// IEEE TC 2011): with (U1, U2) = TwoProd(X, Y), (A1, A2) = TwoSum(Z, U2),
  /// (B1, B2) = TwoSum(U1, A1) and G = RN(RN(B1 - R) + B2), the error is
  /// G + A2 exactly, and RN(G + A2) has its sign. Valid (every intermediate
  /// a multiple of 2^-126, so none denormal, and none overflowing) for normal
  /// X and Y with a product of at least 2^-80 and below 2^123, a Z zero or in
  /// [2^-102, 2^123), and a finite nonzero R; an exact zero takes the general
  /// path (its sign).
  std::pair<Value *, Value *> fastFma(Value *X, Value *Y, Value *Z) {
    auto TwoSum = [&](Value *A, Value *Bv) {
      Value *S = B.CreateFAdd(A, Bv);
      Value *BB = B.CreateFSub(S, A);
      Value *E = B.CreateFAdd(B.CreateFSub(A, B.CreateFSub(S, BB)), B.CreateFSub(Bv, BB));
      return std::pair(S, E);
    };
    Value *R = fma(X, Y, Z);
    Value *U1 = B.CreateFMul(X, Y);
    Value *U2 = fma(X, Y, B.CreateFNeg(U1));
    auto [A1, A2] = TwoSum(Z, U2);
    auto [B1, B2] = TwoSum(U1, A1);
    Value *G = B.CreateFAdd(B.CreateFSub(B1, R), B2);
    Value *Err = B.CreateFAdd(G, A2);
    Value *EX = B.CreateAnd(B.CreateLShr(bits(X), 23), c32(0xff));
    Value *EY = B.CreateAnd(B.CreateLShr(bits(Y), 23), c32(0xff));
    auto Normal = [&](Value *E) {
      return B.CreateICmpULT(B.CreateSub(E, c32(1)), c32(254));
    };
    // biased exponents adding up to 254 - 80 .. 254 + 121 (the product below 2^123)
    Value *ES = B.CreateAdd(EX, EY);
    Value *Valid = B.CreateAnd(B.CreateAnd(Normal(EX), Normal(EY)),
                               B.CreateICmpULT(B.CreateSub(ES, c32(254 - 80)), c32(80 + 121 + 1)));
    Value *ZM = B.CreateAnd(bits(Z), c32(0x7fffffff));
    // Z zero, or in [2^-102, 2^123)
    Value *ZOk = B.CreateOr(B.CreateICmpEQ(ZM, c32(0)),
                            B.CreateICmpULT(B.CreateSub(ZM, c32(0x0c800000)), c32(0x7d000000 - 0x0c800000)));
    Value *RM = B.CreateAnd(bits(R), c32(0x7fffffff));
    Value *ROk = B.CreateAnd(B.CreateICmpNE(RM, c32(0)), B.CreateICmpULT(RM, c32(0x7f800000)));
    Valid = B.CreateAnd(Valid, B.CreateAnd(ZOk, ROk));
    return {adjust(R, Err), Valid};
  }

  /// Division of normal operands with a normal result, using the hardware:
  /// the mantissas (in [1, 2)) are divided with a refined reciprocal, and
  /// the quotient is corrected until the exact remainder shows that it is
  /// the correctly rounded one. Returns the result and whether it is valid.
  std::pair<Value *, Value *> fastDiv(Value *A, Value *D) {
    Value *AB = bits(A), *DB = bits(D);
    Value *EA = B.CreateAnd(B.CreateLShr(AB, 23), c32(0xff));
    Value *ED = B.CreateAnd(B.CreateLShr(DB, 23), c32(0xff));
    auto Normal = [&](Value *E) {
      return B.CreateICmpULT(B.CreateSub(E, c32(1)), c32(254));
    };
    Value *Mant = c32(0x7fffff), *One = c32(0x3f800000);
    Value *A1 = flt(B.CreateOr(B.CreateAnd(AB, Mant), One));
    Value *D1 = flt(B.CreateOr(B.CreateAnd(DB, Mant), One));
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {F32}, {D1});
    Value *NegD = B.CreateFNeg(D1);
    Value *Y = fma(fma(NegD, Y0, fconst(1.0f)), Y0, Y0);
    Value *Q0 = B.CreateFMul(A1, Y);
    Value *Q = fma(fma(NegD, Q0, A1), Y, Q0);
    for (int I = 0; I < 2; ++I) {
      // a/b lies in (q - gap below / 2, q + gap above / 2] around the
      // correctly rounded q; the remainder r = a - b * q is exact.
      Value *Up = nextUp(Q), *Down = nextDown(Q);
      Value *R = fma(NegD, Q, A1);
      Value *HalfUp = B.CreateFMul(D1, B.CreateFMul(B.CreateFSub(Up, Q), fconst(0.5f)));
      Value *HalfDown = B.CreateFMul(D1, B.CreateFMul(B.CreateFSub(Q, Down), fconst(0.5f)));
      Value *Odd = B.CreateTrunc(bits(Q), I1);
      Value *GoUp = B.CreateOr(B.CreateFCmpOGT(R, HalfUp),
                               B.CreateAnd(B.CreateFCmpOEQ(R, HalfUp), Odd));
      Value *NegHalfDown = B.CreateFNeg(HalfDown);
      Value *GoDown = B.CreateOr(B.CreateFCmpOLT(R, NegHalfDown),
                                 B.CreateAnd(B.CreateFCmpOEQ(R, NegHalfDown), Odd));
      Q = B.CreateSelect(GoUp, Up, B.CreateSelect(GoDown, Down, Q));
    }
    // Directed rounding: the remainder gives the direction of the error of
    // the magnitude.
    Value *Neg = B.CreateICmpSLT(B.CreateXor(AB, DB), c32(0));
    Q = adjustMagnitude(Q, fma(NegD, Q, A1), Neg);
    Value *QB = bits(Q);
    Value *Shift = B.CreateSub(EA, ED);
    Value *Exp = B.CreateAdd(B.CreateAnd(B.CreateLShr(QB, 23), c32(0xff)), Shift);
    Value *Valid = B.CreateAnd(B.CreateAnd(Normal(EA), Normal(ED)), Normal(Exp));
    Value *Sign = B.CreateAnd(B.CreateXor(AB, DB), c32(0x80000000));
    Value *R = B.CreateOr(B.CreateAdd(QB, B.CreateShl(Shift, 23)), Sign);
    return {flt(R), Valid};
  }

  /// The target's division (round to nearest, flushing; tagged exact, so it
  /// is not lowered again), and whether it is the IEEE result: normal
  /// operands and a normal result.
  std::pair<Value *, Value *> hwDiv(Value *A, Value *D) {
    auto *Q = cast<Instruction>(B.CreateFDiv(A, D));
    Q->setMetadata("agx.exact", MDNode::get(Q->getContext(), {}));
    auto Normal = [&](Value *V) {
      Value *E = B.CreateAnd(B.CreateLShr(bits(V), 23), c32(0xff));
      return B.CreateICmpULT(B.CreateSub(E, c32(1)), c32(254));
    };
    return {Q, B.CreateAnd(B.CreateAnd(Normal(A), Normal(D)), Normal(Q))};
  }

  /// The target's square root (round to nearest; tagged exact), valid for a
  /// positive normal operand.
  std::pair<Value *, Value *> hwSqrt(Value *X) {
    auto *S = cast<Instruction>(B.CreateUnaryIntrinsic(Intrinsic::sqrt, X));
    S->setMetadata("agx.exact", MDNode::get(S->getContext(), {}));
    Value *Valid = B.CreateICmpULT(B.CreateSub(bits(X), c32(0x00800000)), c32(0x7f000000));
    return {S, Valid};
  }

  /// Square root of a positive normal value, using the hardware: the root of
  /// the mantissa (in [1, 4)) is refined from rsqrt and corrected until the
  /// midpoint test of Tuckerman shows that it is correctly rounded.
  std::pair<Value *, Value *> fastSqrt(Value *X) {
    Value *XB = bits(X);
    Value *Valid = B.CreateICmpULT(B.CreateSub(XB, c32(0x00800000)),
                                   c32(0x7f000000));
    Value *K = B.CreateSub(B.CreateAnd(B.CreateLShr(XB, 23), c32(0xff)), c32(127));
    Value *Odd = B.CreateAnd(K, c32(1));
    Value *Exp = B.CreateSelect(B.CreateICmpNE(Odd, c32(0)), c32(0x40000000),
                                c32(0x3f800000));
    Value *M = flt(B.CreateOr(B.CreateAnd(XB, c32(0x7fffff)), Exp));
    Value *NegM = B.CreateFNeg(M);
    Value *Y = B.CreateIntrinsic(Intrinsic::agx_rsqrt, {F32}, {M});
    Value *S0 = B.CreateFMul(M, Y);
    Value *S = fma(fma(B.CreateFNeg(S0), S0, M), B.CreateFMul(Y, fconst(0.5f)), S0);
    for (int I = 0; I < 2; ++I) {
      // s is correctly rounded iff s * down(s) < m <= s * up(s) (there are no
      // ties for square roots).
      Value *Up = nextUp(S), *Down = nextDown(S);
      Value *Low = B.CreateFCmpOGE(fma(S, Down, NegM), fconst(0.0f));
      Value *High = B.CreateFCmpOLT(fma(S, Up, NegM), fconst(0.0f));
      S = B.CreateSelect(High, Up, B.CreateSelect(Low, Down, S));
    }
    S = adjustMagnitude(S, fma(B.CreateFNeg(S), S, M), B.getFalse());
    Value *Half = B.CreateAShr(B.CreateSub(K, Odd), 1);
    Value *R = B.CreateAdd(bits(S), B.CreateShl(Half, 23));
    return {flt(R), Valid};
  }

  /// IEEE division of finite or special operands, correctly rounded.
  Value *div(Value *A, Value *D) {
    Value *AB = bits(A), *DB = bits(D);
    Term TA = unpack(AB, true), TD = unpack(DB, true);
    // Q = floor(MA * 2^26 / MD), with MA, MD in [2^23, 2^24): 26 or 27 bits.
    // A floating point estimate is exact to a few units; two integer
    // corrections make it exact.
    Value *X = B.CreateShl(TA.M, 26);
    Value *FD = B.CreateUIToFP(TD.M, F32);
    Value *Inv = B.CreateIntrinsic(Intrinsic::agx_rcp, {F32}, {FD});
    Value *Est = B.CreateFMul(B.CreateFMul(B.CreateUIToFP(TA.M, F32), Inv),
                              fconst(0x1p26f));
    Value *Q = B.CreateZExt(B.CreateFPToUI(Est, I32), I64);
    Value *R = B.CreateSub(X, B.CreateMul(Q, TD.M));
    Value *C = B.CreateFPToSI(
        B.CreateFMul(B.CreateSIToFP(R, F32), Inv), I32);
    Q = B.CreateAdd(Q, B.CreateSExt(C, I64));
    R = B.CreateSub(R, B.CreateMul(B.CreateSExt(C, I64), TD.M));
    for (int I = 0; I < 2; ++I) {
      Value *Low = B.CreateICmpSLT(R, c64(0));
      Q = B.CreateSelect(Low, B.CreateSub(Q, c64(1)), Q);
      R = B.CreateSelect(Low, B.CreateAdd(R, TD.M), R);
      Value *High = B.CreateICmpSGE(R, TD.M);
      Q = B.CreateSelect(High, B.CreateAdd(Q, c64(1)), Q);
      R = B.CreateSelect(High, B.CreateSub(R, TD.M), R);
    }
    Value *M = B.CreateOr(B.CreateShl(Q, 1),
                          B.CreateZExt(B.CreateICmpNE(R, c64(0)), I64));
    Value *E = B.CreateSub(B.CreateSub(TA.E, TD.E), c32(27));
    Value *Sign = B.CreateXor(TA.Sign, TD.Sign);
    Value *Bits = convert({Sign, M, E});

    // Zeros, infinities and NaNs.
    Value *SignBits = B.CreateShl(B.CreateZExt(Sign, I32), 31);
    Value *AZero = isZero(AB), *DZero = isZero(DB);
    Value *AInf = B.CreateICmpEQ(B.CreateAnd(AB, c32(0x7fffffff)), c32(0x7f800000));
    Value *DInf = B.CreateICmpEQ(B.CreateAnd(DB, c32(0x7fffffff)), c32(0x7f800000));
    Value *NaN = B.CreateOr(B.CreateFCmpUNO(A, D),
                            B.CreateOr(B.CreateAnd(AZero, DZero),
                                       B.CreateAnd(AInf, DInf)));
    Value *Inf = B.CreateOr(AInf, DZero);
    Value *Zero = B.CreateOr(AZero, DInf);
    Bits = B.CreateSelect(Zero, SignBits, Bits);
    Bits = B.CreateSelect(Inf, B.CreateOr(SignBits, c32(0x7f800000)), Bits);
    Bits = B.CreateSelect(NaN, c32(0x7fc00000), Bits);
    return flt(Bits);
  }

  /// IEEE square root, correctly rounded.
  Value *sqrt(Value *X) {
    Value *XB = bits(X);
    Term T = unpack(XB, true);
    // An even exponent: M in [2^23, 2^25).
    Value *Odd = B.CreateTrunc(T.E, I1);
    Value *M = B.CreateSelect(Odd, B.CreateShl(T.M, 1), T.M);
    Value *E = B.CreateSelect(Odd, B.CreateSub(T.E, c32(1)), T.E);
    // S = floor(sqrt(M * 2^28)), 26 or 27 bits.
    Value *W = B.CreateShl(M, 28);
    Value *FM = B.CreateUIToFP(M, F32);
    Value *Root = B.CreateFMul(
        FM, B.CreateIntrinsic(Intrinsic::agx_rsqrt, {F32}, {FM}));
    Value *S = B.CreateZExt(
        B.CreateFPToUI(B.CreateFMul(Root, fconst(0x1p14f)), I32), I64);
    Value *R = B.CreateSub(W, B.CreateMul(S, S));
    Value *Half = B.CreateIntrinsic(
        Intrinsic::agx_rcp, {F32},
        {B.CreateFMul(B.CreateUIToFP(S, F32), fconst(2.0f))});
    Value *C = B.CreateFPToSI(B.CreateFMul(B.CreateSIToFP(R, F32), Half), I32);
    S = B.CreateAdd(S, B.CreateSExt(C, I64));
    R = B.CreateSub(W, B.CreateMul(S, S));
    for (int I = 0; I < 2; ++I) {
      Value *Low = B.CreateICmpSLT(R, c64(0));
      R = B.CreateSelect(Low, B.CreateAdd(R, B.CreateSub(B.CreateShl(S, 1),
                                                         c64(1))),
                         R);
      S = B.CreateSelect(Low, B.CreateSub(S, c64(1)), S);
      Value *High = B.CreateICmpSGT(R, B.CreateShl(S, 1));
      R = B.CreateSelect(High, B.CreateSub(R, B.CreateAdd(B.CreateShl(S, 1),
                                                          c64(1))),
                         R);
      S = B.CreateSelect(High, B.CreateAdd(S, c64(1)), S);
    }
    Value *Root2 = B.CreateOr(B.CreateShl(S, 1),
                              B.CreateZExt(B.CreateICmpNE(R, c64(0)), I64));
    Value *RE = B.CreateSub(B.CreateAShr(B.CreateSub(E, c32(28)), 1), c32(1));
    Value *Bits = convert({B.getFalse(), Root2, RE});
    // sqrt(+-0) = +-0, sqrt(+inf) = +inf, NaN for negative values and NaNs.
    Bits = B.CreateSelect(B.CreateICmpEQ(XB, c32(0x7f800000)), XB, Bits);
    Bits = B.CreateSelect(
        B.CreateOr(B.CreateFCmpUNO(X, X),
                   B.CreateAnd(signBit(XB), B.CreateNot(isZero(XB)))),
        c32(0x7fc00000), Bits);
    Bits = B.CreateSelect(isZero(XB), XB, Bits);
    return flt(Bits);
  }

  /// Binary32 X + Y (or X - Y), rounded in the direction Rounding.
  Value *roundedAdd(Value *X, Value *Y, bool Sub) {
    Value *XB = bits(X), *YB = bits(Y);
    Term TX = unpack(XB, false), TY = unpack(YB, false);
    if (Sub)
      TY.Sign = B.CreateNot(TY.Sign);
    Value *Exact = flt(convert(add(TX, TY)));
    // Infinities and NaNs: denormal operands do not change the result.
    Value *HW = Sub ? B.CreateFSub(X, Y) : B.CreateFAdd(X, Y);
    return B.CreateSelect(B.CreateOr(isInfOrNaN(XB), isInfOrNaN(YB)), HW, Exact);
  }

  /// Binary32 X * Y, rounded in the direction Rounding.
  Value *roundedMul(Value *X, Value *Y) {
    Value *XB = bits(X), *YB = bits(Y);
    Value *Exact = flt(convert(mul(unpack(XB, false), unpack(YB, false))));
    Value *HW = B.CreateFMul(nonzeroNormal(X), nonzeroNormal(Y));
    return B.CreateSelect(B.CreateOr(isInfOrNaN(XB), isInfOrNaN(YB)), HW, Exact);
  }

  /// Binary32 X * Y + Z with one rounding in the direction Rounding.
  Value *roundedFma(Value *X, Value *Y, Value *Z) {
    Value *XB = bits(X), *YB = bits(Y), *ZB = bits(Z);
    Term P = mul(unpack(XB, false), unpack(YB, false));
    Value *Exact = flt(convert(add(P, unpack(ZB, false))));
    Value *HW = fma(nonzeroNormal(X), nonzeroNormal(Y), nonzeroNormal(Z));
    Value *Special = B.CreateOr(isInfOrNaN(XB),
                                B.CreateOr(isInfOrNaN(YB), isInfOrNaN(ZB)));
    return B.CreateSelect(Special, HW, Exact);
  }

  /// An integer converted to Fmt (float or half), rounded in the direction
  /// Rounding.
  Value *roundedFromInt(Value *V, bool Signed, Type *Ty) {
    Format Fmt = Ty->isHalfTy() ? Binary16 : Binary32;
    Value *W = Signed ? B.CreateSExtOrTrunc(V, I64) : B.CreateZExtOrTrunc(V, I64);
    Value *Neg = Signed ? B.CreateICmpSLT(W, c64(0)) : B.getFalse();
    Value *M = B.CreateSelect(Neg, B.CreateNeg(W), W);
    Value *Bits = convert({Neg, M, c32(0)}, Fmt);
    return toFloat(Bits, Ty);
  }

  /// A float converted to half, rounded in the direction Rounding.
  Value *roundedToHalf(Value *X) {
    Value *XB = bits(X);
    Value *HW = B.CreateFPTrunc(X, B.getHalfTy());
    Value *NegX = B.CreateICmpSLT(XB, c32(0));
    // A known direction: the hardware's conversion to nearest, one unit
    // toward the direction where it went past X (its half is a normal float
    // or zero, so the comparison is exact; that covers zeros, the overflow
    // to infinity and infinities, NaNs compare false). A float denormal,
    // which the hardware flushes, takes the integer conversion.
    if (auto *RC = dyn_cast<ConstantInt>(Rounding)) {
      uint64_t Dir = RC->getZExtValue();
      if (Dir == RoundNearestEven || Dir == RoundTowardZero || Dir == RoundUp ||
          Dir == RoundDown) {
        Value *R = HW;
        if (Dir != RoundNearestEven) {
          Value *E = B.CreateFPExt(HW, X->getType());
          Value *HB = B.CreateBitCast(HW, B.getInt16Ty());
          Value *Up = B.CreateAdd(HB, ConstantInt::get(B.getInt16Ty(), 1));
          Value *Down = B.CreateSub(HB, ConstantInt::get(B.getInt16Ty(), 1));
          // a unit up or down in magnitude (the sign bit is kept)
          Value *Adj;
          if (Dir == RoundTowardZero) {
            Value *Past =
                B.CreateFCmpOGT(B.CreateUnaryIntrinsic(Intrinsic::fabs, E),
                                B.CreateUnaryIntrinsic(Intrinsic::fabs, X));
            Adj = B.CreateSelect(Past, Down, HB);
          } else {
            bool UpDir = Dir == RoundUp;
            Value *Past = UpDir ? B.CreateFCmpOLT(E, X) : B.CreateFCmpOGT(E, X);
            // toward +inf: a positive grows, a negative shrinks (and the
            // reverse toward -inf)
            Value *Grow = UpDir ? B.CreateNot(NegX) : NegX;
            Adj = B.CreateSelect(Past, B.CreateSelect(Grow, Up, Down), HB);
          }
          R = B.CreateBitCast(Adj, B.getHalfTy());
        }
        // a float denormal (which the hardware flushes) is below half the
        // smallest half denormal: a zero of its sign, or that denormal
        // rounding away from zero
        Value *Mag = B.CreateAnd(XB, c32(0x7fffffff));
        Value *Denormal = B.CreateAnd(B.CreateICmpNE(Mag, c32(0)),
                                      B.CreateICmpULT(Mag, c32(0x800000)));
        Value *Away = Dir == RoundUp     ? B.CreateNot(NegX)
                      : Dir == RoundDown ? NegX
                                         : B.getFalse();
        Value *Tiny = B.CreateOr(
            B.CreateSelect(NegX, ConstantInt::get(B.getInt16Ty(), 0x8000),
                           ConstantInt::get(B.getInt16Ty(), 0)),
            B.CreateZExt(Away, B.getInt16Ty()));
        return B.CreateSelect(Denormal, B.CreateBitCast(Tiny, B.getHalfTy()),
                              R);
      }
    }
    Value *Exact = toFloat(convert(unpack(XB, false), Binary16), B.getHalfTy());
    return B.CreateSelect(isInfOrNaN(XB), HW, Exact);
  }

  /// Bits of Fmt in an i32 as a value of type Ty.
  Value *toFloat(Value *Bits, Type *Ty) {
    if (Ty->isHalfTy())
      return B.CreateBitCast(B.CreateTrunc(Bits, B.getInt16Ty()), Ty);
    return flt(Bits);
  }

  /// Round to an integral value in the direction Rounding.
  Value *roundedIntegral(Value *X) {
    auto Call = [&](Intrinsic::ID ID) { return B.CreateUnaryIntrinsic(ID, X); };
    Value *R = B.CreateSelect(roundingIs(RoundDown), Call(Intrinsic::floor),
                              Call(Intrinsic::roundeven));
    R = B.CreateSelect(roundingIs(RoundUp), Call(Intrinsic::ceil), R);
    return B.CreateSelect(roundingIs(RoundTowardZero), Call(Intrinsic::trunc), R);
  }

  /// flush(X) = |X| below the smallest normal ? +-0 : X, for float or half.
  Value *flushDenormal(Value *X) {
    Type *Ty = X->getType();
    APFloat Min = APFloat::getSmallestNormalized(Ty->getFltSemantics());
    Value *Small = B.CreateFCmpOLT(B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
                                   ConstantFP::get(Ty, Min));
    Value *Zero = B.CreateBinaryIntrinsic(Intrinsic::copysign,
                                          ConstantFP::get(Ty, 0.0), X);
    return B.CreateSelect(Small, Zero, X);
  }

  /// A comparison operand as a signed integer, for zeros and denormals.
  Value *denormalOrder(Value *Bits) {
    Value *Mag = B.CreateAnd(Bits, c32(0x7fffff));
    return B.CreateSelect(signBit(Bits), B.CreateNeg(Mag), Mag);
  }

  /// fcmp that orders denormals.
  Value *compare(CmpInst::Predicate P, Value *X, Value *Y, Value *HW) {
    Value *XB = bits(X), *YB = bits(Y);
    Value *Both = B.CreateAnd(expZero(XB), expZero(YB));
    Value *VX = denormalOrder(XB), *VY = denormalOrder(YB);
    Value *Int;
    switch (P) {
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_UEQ:
      Int = B.CreateICmpEQ(VX, VY);
      break;
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_UNE:
      Int = B.CreateICmpNE(VX, VY);
      break;
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_ULT:
      Int = B.CreateICmpSLT(VX, VY);
      break;
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ULE:
      Int = B.CreateICmpSLE(VX, VY);
      break;
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_UGT:
      Int = B.CreateICmpSGT(VX, VY);
      break;
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_UGE:
      Int = B.CreateICmpSGE(VX, VY);
      break;
    default:
      return HW; // ord, uno, true, false: denormals change nothing.
    }
    return B.CreateSelect(Both, Int, HW);
  }

  /// X < Y, with denormals.
  Value *less(Value *X, Value *Y) {
    return compare(CmpInst::FCMP_OLT, X, Y, B.CreateFCmpOLT(X, Y));
  }

  /// minimumnum / maximumnum (and minnum / maxnum): a NaN operand is ignored,
  /// -0 < +0. With Propagate (minimum / maximum), NaN operands give NaN.
  Value *minMax(Value *X, Value *Y, bool Min, bool Propagate) {
    Value *XB = bits(X), *YB = bits(Y);
    Value *Lt = less(X, Y), *Gt = less(Y, X);
    Value *Pick = Min ? B.CreateSelect(Lt, XB, B.CreateSelect(Gt, YB, B.CreateOr(XB, YB)))
                      : B.CreateSelect(Gt, XB, B.CreateSelect(Lt, YB, B.CreateAnd(XB, YB)));
    Value *XNaN = B.CreateFCmpUNO(X, X), *YNaN = B.CreateFCmpUNO(Y, Y);
    if (Propagate) {
      Pick = B.CreateSelect(YNaN, YB, Pick);
      Pick = B.CreateSelect(XNaN, XB, Pick);
    } else {
      Pick = B.CreateSelect(YNaN, XB, Pick);
      Pick = B.CreateSelect(XNaN, YB, Pick);
    }
    return flt(Pick);
  }
};

} // namespace

char AGXStrictFP::ID = 0;

INITIALIZE_PASS_BEGIN(AGXStrictFP, DEBUG_TYPE, "AGX Strict FP", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXStrictFP, DEBUG_TYPE, "AGX Strict FP", false, false)

FunctionPass *llvm::createAGXStrictFPPass() { return new AGXStrictFP(); }

static bool isF32(const Value *V) { return V->getType()->isFloatTy(); }

static Intrinsic::ID getIntrinsicID(const Instruction &I) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&I))
    return II->getIntrinsicID();
  return Intrinsic::not_intrinsic;
}

/// Floating point operations whose NaN results are canonicalized.
static bool producesArithmeticNaN(const Instruction &I) {
  if (!I.getType()->isFPOrFPVectorTy())
    return false;
  // (nnan: a front end's promise that no NaN comes of it)
  if (isa<FPMathOperator>(I) && I.hasNoNaNs())
    return false;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
    return true;
  default:
    break;
  }
  switch (getIntrinsicID(I)) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::agx_exp2:
  case Intrinsic::agx_log2:
  case Intrinsic::agx_sin:
  case Intrinsic::agx_cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::round:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::canonicalize:
  case Intrinsic::agx_rcp:
  case Intrinsic::agx_rsqrt:
    return true;
  default:
    return false;
  }
}

static cl::opt<bool> CanonicalNaNAtUses(
    "agx-canonical-nan-at-uses", cl::init(true), cl::Hidden,
    cl::desc("canonical-nan: no canonical NaN for a result whose every use "
             "canonicalizes its own NaN results or only tests for NaN"));

/// No use of V (directly or through selects and phis) sees the bits of a NaN
/// V: each is an operation of Canon, whose NaN results are canonicalized
/// whatever the NaN operands' bits (and which gives a NaN for a NaN operand,
/// or ignores it, as IEEE operations do), or a comparison.
static bool nanPayloadUnobserved(const Instruction &V,
                                 const SmallPtrSetImpl<const Instruction *> &Canon,
                                 DenseMap<const Instruction *, bool> &Memo,
                                 unsigned Depth) {
  if (auto It = Memo.find(&V); It != Memo.end())
    return It->second;
  // (a cycle of phis is taken as observed, also while it is being visited)
  Memo[&V] = false;
  if (Depth > 8 || V.use_empty())
    return false;
  bool OK = all_of(V.uses(), [&](const Use &U) {
    auto *UI = dyn_cast<Instruction>(U.getUser());
    if (!UI)
      return false;
    if (isa<FCmpInst>(UI))
      return true;
    if (Canon.contains(UI) && UI->getType()->isFloatingPointTy())
      return true;
    if ((isa<PHINode>(UI) ||
         (isa<SelectInst>(UI) && U.getOperandNo() != 0)) &&
        UI->getType() == V.getType())
      return nanPayloadUnobserved(*UI, Canon, Memo, Depth + 1);
    return false;
  });
  Memo[&V] = OK;
  return OK;
}

/// Replace the NaN results of I by the canonical NaN.
static void canonicalizeNaN(Instruction &I) {
  Type *Ty = I.getType();
  const fltSemantics &Sem = Ty->getScalarType()->getFltSemantics();
  unsigned Width = APFloat::getSizeInBits(Sem);
  APFloat NaN(Sem, APInt::getSignedMaxValue(Width));
  IRBuilder<> B(I.getNextNode());
  SmallVector<Use *, 4> Uses;
  for (Use &U : I.uses())
    Uses.push_back(&U);
  Value *R = B.CreateSelect(B.CreateFCmpUNO(&I, &I), ConstantFP::get(Ty, NaN),
                            &I);
  for (Use *U : Uses)
    U->set(R);
}

/// A canonical NaN made on the bits of a float V (a front end does it so where
/// float comparisons may be taken for fast math), either
/// select((bits(V) & Abs) > bits(Inf), Abs, bits(V)) or, as InstCombine makes
/// it, select(bits(fabs(V)) > bits(Inf), NaN, V): returns V, or null. On AGX
/// a float comparison sees NaNs, and one fcmpsel replaces the and and icmpsel.
static Value *matchBitsCanonicalNaN(SelectInst &S) {
  using namespace PatternMatch;
  Value *X, *V;
  const APInt *Abs = nullptr, *Inf, *NaN = nullptr;
  if (match(&S, m_Select(m_SpecificICmp(ICmpInst::ICMP_UGT,
                                        m_And(m_Value(X), m_APInt(Abs)),
                                        m_APInt(Inf)),
                         m_APInt(NaN), m_Deferred(X)))) {
    auto *BC = dyn_cast<BitCastInst>(X);
    if (!BC)
      return nullptr;
    V = BC->getOperand(0);
  } else if (!match(&S, m_Select(m_SpecificICmp(
                                     ICmpInst::ICMP_UGT,
                                     m_BitCast(m_FAbs(m_Value(V))),
                                     m_APInt(Inf)),
                                 m_Constant(), m_Deferred(V)))) {
    return nullptr;
  }
  Type *FTy = V->getType();
  Type *ITy = S.getCondition()->getType();
  if (!FTy->getScalarType()->isFloatingPointTy() ||
      FTy->isVectorTy() != ITy->isVectorTy() ||
      (FTy->isVectorTy() && cast<VectorType>(FTy)->getElementCount() !=
                                cast<VectorType>(ITy)->getElementCount()))
    return nullptr;
  const fltSemantics &Sem = FTy->getScalarType()->getFltSemantics();
  unsigned W = APFloat::getSizeInBits(Sem);
  APInt Max = APInt::getSignedMaxValue(W);
  if (Inf->getBitWidth() != W ||
      *Inf != APFloat::getInf(Sem).bitcastToAPInt())
    return nullptr;
  if (Abs) {
    if (*Abs != Max || *NaN != Max)
      return nullptr;
  } else {
    auto *T = cast<Constant>(S.getTrueValue());
    auto *C = dyn_cast_or_null<ConstantFP>(
        T->getType()->isVectorTy() ? T->getSplatValue() : T);
    if (!C || C->getValueAPF().bitcastToAPInt() != Max)
      return nullptr;
  }
  return V;
}

/// Replace I by Fast where Slow is false, and by the value computed by Build
/// in a conditional block otherwise. Cond is built right after I.
/// Batched IEEE lowering (lowerIEEEGroups): in Collect mode the two helpers
/// below leave the fast result in place and record the condition under which
/// it is wrong; in Exact mode they replace the instruction by its exact
/// software result. Normal: a conditional slow path per operation.
enum class SlowPathMode { Normal, Collect, Exact };
static SlowPathMode SPMode = SlowPathMode::Normal;
static SmallVector<Value *, 16> SPInvalid;

/// Hardware results of grouped IEEE operations (lowerIEEEGroup), with the
/// block in which their run's check covers them: there they are never
/// denormal (the hardware flushes its results, and the check fails where the
/// right result is denormal), so operations in that block need not test them.
static DenseMap<const Value *, const BasicBlock *> GroupFast;

static cl::opt<bool> GroupNormal(
    "agx-ieee-group-normal", cl::init(true), cl::Hidden,
    cl::desc("IEEE runs checked together: no denormal tests of the run's "
             "results in the run"));

/// V is not denormal where an operation in block At uses it.
static bool knownNotDenormal(Value *V, const BasicBlock *At) {
  for (unsigned D = 0; D < 4; ++D) {
    if (auto *C = dyn_cast<ConstantFP>(V))
      return !C->getValueAPF().isDenormal();
    if (auto It = GroupFast.find(V); It != GroupFast.end())
      return It->second == At;
    Value *A;
    if (match(V, PatternMatch::m_FNeg(PatternMatch::m_Value(A))) ||
        match(V, PatternMatch::m_FAbs(PatternMatch::m_Value(A)))) {
      V = A;
      continue;
    }
    // (the canonical NaN of a value)
    auto *S = dyn_cast<SelectInst>(V);
    auto *C = S ? dyn_cast<FCmpInst>(S->getCondition()) : nullptr;
    auto *NaN = S ? dyn_cast<ConstantFP>(S->getTrueValue()) : nullptr;
    if (!C || !NaN || !NaN->isNaN() ||
        C->getPredicate() != CmpInst::FCMP_UNO ||
        C->getOperand(0) != S->getFalseValue() ||
        C->getOperand(1) != S->getFalseValue())
      return false;
    V = S->getFalseValue();
  }
  return false;
}

/// Whether a value is never denormal by what computes it (IEEE semantics, the
/// value the lowering keeps): integer conversions, half extensions, square
/// roots, reciprocal square roots and rounding to integers give none, and
/// moves, selects and phis keep the property of what they choose from. A phi
/// cycle is assumed not denormal while it is visited (its values come from
/// its other incoming values through those moves).
static bool neverDenormal(const Value *V, SmallPtrSetImpl<const Value *> &Visiting,
                          unsigned Depth = 0) {
  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->getValueAPF().isDenormal();
  if (isa<UndefValue>(V))
    return false;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || Depth > 6 || !I->getType()->isFloatTy())
    return false;
  switch (I->getOpcode()) {
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    return true;
  case Instruction::FPExt:
    return I->getOperand(0)->getType()->isHalfTy();
  case Instruction::FNeg:
    return neverDenormal(I->getOperand(0), Visiting, Depth + 1);
  case Instruction::Select:
    return neverDenormal(I->getOperand(1), Visiting, Depth + 1) &&
           neverDenormal(I->getOperand(2), Visiting, Depth + 1);
  case Instruction::PHI: {
    if (!Visiting.insert(I).second)
      return true;
    bool R = all_of(cast<PHINode>(I)->incoming_values(), [&](const Value *In) {
      return neverDenormal(In, Visiting, Depth + 1);
    });
    Visiting.erase(I);
    return R;
  }
  default:
    break;
  }
  switch (getIntrinsicID(*I)) {
  case Intrinsic::sqrt:
  case Intrinsic::agx_rsqrt:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::round:
    return true;
  case Intrinsic::fabs:
    return neverDenormal(I->getOperand(0), Visiting, Depth + 1);
  case Intrinsic::copysign:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
    return neverDenormal(I->getOperand(0), Visiting, Depth + 1) &&
           (getIntrinsicID(*I) == Intrinsic::copysign ||
            neverDenormal(I->getOperand(1), Visiting, Depth + 1));
  default:
    return false;
  }
}

static cl::opt<bool> OperandRanges(
    "agx-ieee-operand-ranges", cl::init(true), cl::Hidden,
    cl::desc("IEEE binary32: no denormal tests of operands that are never "
             "denormal by what computes them (conversions from integers and "
             "half, square roots, roundings)"));

/// The operands of each operation that are never denormal (bit K for operand
/// K), from the IR before any lowering: a lowering replaces an operation by
/// one of the same value, so the bits stay right for its users.
static DenseMap<const Instruction *, unsigned> OperandNotDenormal;

static void recordOperandRanges(ArrayRef<Instruction *> Ops) {
  OperandNotDenormal.clear();
  if (!OperandRanges)
    return;
  for (Instruction *I : Ops) {
    unsigned Mask = 0;
    for (unsigned K = 0, E = std::min(I->getNumOperands(), 8u); K < E; ++K) {
      SmallPtrSet<const Value *, 8> Visiting;
      if (I->getOperand(K)->getType()->isFloatTy() &&
          neverDenormal(I->getOperand(K), Visiting))
        Mask |= 1u << K;
    }
    if (Mask)
      OperandNotDenormal[I] = Mask;
  }
}

/// Operand K of I is not denormal where I uses it.
static bool operandNotDenormal(const Instruction &I, unsigned K) {
  if (auto It = OperandNotDenormal.find(&I);
      It != OperandNotDenormal.end() && (It->second >> K & 1))
    return true;
  return knownNotDenormal(I.getOperand(K), I.getParent());
}

static cl::opt<bool> GroupChain(
    "agx-ieee-group-chain", cl::init(true), cl::Hidden,
    cl::desc("IEEE runs checked together: each multiplication's and fma's "
             "condition as selects that fall back to the run's condition so "
             "far (fcmpsel / icmpsel chains), not or-ed after"));

/// In a run's collection (lowerIEEEGroup): the condition of the run so far,
/// which a chained condition includes.
static Value *SPPrev = nullptr;
static bool SPChained = false;
static Value *chainPrev() {
  if (SPMode != SlowPathMode::Collect || !GroupChain || !SPPrev)
    return nullptr;
  SPChained = true;
  return SPPrev;
}

/// Prev, or Tiny with neither factor zero: as selects falling back to Prev.
static Value *chainTiny(IRBuilderBase &B, Value *Tiny, Value *ZX, Value *ZY,
                        Value *Prev) {
  Value *NZ = B.CreateSelect(ZY, Prev, B.getTrue());
  NZ = B.CreateSelect(ZX, Prev, NZ);
  return B.CreateSelect(Tiny, NZ, Prev);
}

/// Operand K of I is zero: a float comparison where it is not denormal (which
/// the hardware compares as zero), else on its bits.
static Value *zeroTest(const Instruction &I, unsigned K, SoftFloat &SF,
                       IRBuilderBase &B) {
  Value *V = I.getOperand(K);
  if (operandNotDenormal(I, K))
    return B.CreateFCmpOEQ(V, ConstantFP::get(V->getType(), 0.0));
  return SF.isZero(SF.bits(V));
}

template <typename CondFn, typename SlowFn>
static void lowerWithSlowPath(Instruction &I, CondFn BuildCond,
                              SlowFn BuildSlow) {
  SmallVector<Use *, 4> Uses;
  for (Use &U : I.uses())
    Uses.push_back(&U);
  Instruction *Next = I.getNextNode();
  IRBuilder<> B(Next);
  SoftFloat SF(B);
  if (SPMode == SlowPathMode::Exact) {
    Value *Slow = BuildSlow(SF, B);
    I.replaceAllUsesWith(Slow); // (value handles follow)
    I.eraseFromParent();
    return;
  }
  Value *Cond = BuildCond(SF, B);
  if (SPMode == SlowPathMode::Collect) {
    SPInvalid.push_back(Cond);
    return;
  }
  Instruction *ThenTerm = SplitBlockAndInsertIfThen(
      Cond, Next->getIterator(), false,
      MDBuilder(I.getContext()).createUnlikelyBranchWeights());
  B.SetInsertPoint(ThenTerm);
  Value *Slow = BuildSlow(SF, B);
  BasicBlock *Then = ThenTerm->getParent();
  BasicBlock *Head = Then->getSinglePredecessor();
  B.SetInsertPoint(Next->getParent(), Next->getParent()->begin());
  PHINode *Phi = B.CreatePHI(I.getType(), 2);
  Phi->addIncoming(&I, Head);
  Phi->addIncoming(Slow, Then);
  for (Use *U : Uses)
    U->set(Phi);
}

/// Replace I by the result of BuildFast, or of BuildSlow (in a conditional
/// block) where BuildFast says that its result is not valid.
template <typename BuilderT = IRBuilder<>, typename FastFn, typename SlowFn>
static void replaceWithSlowPath(Instruction &I, FastFn BuildFast,
                                SlowFn BuildSlow) {
  Instruction *Next = I.getNextNode();
  BuilderT B(Next);
  SoftFloat SF(B);
  if (SPMode == SlowPathMode::Exact) {
    Value *Slow = BuildSlow(SF, B);
    Slow->takeName(&I);
    I.replaceAllUsesWith(Slow);
    I.eraseFromParent();
    return;
  }
  auto [Fast, Valid] = BuildFast(SF, B);
  if (SPMode == SlowPathMode::Collect) {
    SPInvalid.push_back(B.CreateNot(Valid));
    Fast->takeName(&I);
    I.replaceAllUsesWith(Fast);
    I.eraseFromParent();
    return;
  }
  Instruction *ThenTerm =
      SplitBlockAndInsertIfThen(B.CreateNot(Valid), Next->getIterator(), false,
                                MDBuilder(I.getContext()).createUnlikelyBranchWeights());
  B.SetInsertPoint(ThenTerm);
  Value *Slow = BuildSlow(SF, B);
  BasicBlock *Then = ThenTerm->getParent();
  BasicBlock *Head = Then->getSinglePredecessor();
  B.SetInsertPoint(Next->getParent(), Next->getParent()->begin());
  PHINode *Phi = B.CreatePHI(I.getType(), 2);
  Phi->addIncoming(Fast, Head);
  Phi->addIncoming(Slow, Then);
  Phi->takeName(&I);
  I.replaceAllUsesWith(Phi);
  I.eraseFromParent();
}

/// Replace I by the value computed by Build right after it.
template <typename BuildFn> static void replace(Instruction &I, BuildFn Build) {
  IRBuilder<> B(I.getNextNode());
  SoftFloat SF(B);
  Value *R = Build(SF, B);
  R->takeName(&I);
  I.replaceAllUsesWith(R);
  I.eraseFromParent();
}

static cl::opt<bool> OneLog2Expansion(
    "agx-ieee-one-log2", cl::init(true), cl::Hidden,
    cl::desc("IEEE log2 / rcp / rsqrt: one evaluation of the operand scaled by "
             "2^64 where it is denormal (not of both and a select)"));

static cl::opt<bool> Exp2FastScale(
    "agx-ieee-exp2-fast-scale", cl::init(true), cl::Hidden,
    cl::desc("IEEE exp2: the exponent add where the result is normal, the "
             "integer scaling behind an unlikely branch for denormal results"));

static bool lowerIEEE(Instruction &I) {
  Intrinsic::ID ID = getIntrinsicID(I);
  bool Approx = isa<FPMathOperator>(I) &&
                (I.hasApproxFunc() || I.hasAllowReciprocal());

  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub: {
    if (!isF32(&I))
      return false;
    Value *X = I.getOperand(0), *Y = I.getOperand(1);
    bool Sub = I.getOpcode() == Instruction::FSub;
    // (never with an operand of at least 2^-101: a constant one, say)
    auto Large = [](Value *V) {
      auto *C = dyn_cast<ConstantFP>(V);
      return C && !(abs(C->getValueAPF()) < APFloat(0x1p-101f));
    };
    if (Large(X) || Large(Y))
      return false;
    // The hardware result is right unless both operands are below 2^-101:
    // otherwise the result is above 2^-102, so a denormal operand is below
    // half an ulp of it, and a cancellation of normal operands cannot reach
    // the denormal range.
    lowerWithSlowPath(
        I,
        [&](SoftFloat &SF, IRBuilderBase &B) {
          Value *Limit = SF.fconst(0x1p-101f);
          auto Small = [&](Value *V) {
            return B.CreateFCmpOLT(
                B.CreateUnaryIntrinsic(Intrinsic::fabs, V), Limit);
          };
          return B.CreateAnd(Small(X), Small(Y));
        },
        [&](SoftFloat &SF, IRBuilderBase &B) {
          Term TX = SF.unpack(SF.bits(X), false);
          Term TY = SF.unpack(SF.bits(Y), false);
          if (Sub)
            TY.Sign = B.CreateNot(TY.Sign);
          return SF.flt(SF.convert(SF.add(TX, TY)));
        });
    return true;
  }
  case Instruction::FMul: {
    if (!isF32(&I))
      return false;
    Value *X = I.getOperand(0), *Y = I.getOperand(1);
    lowerWithSlowPath(
        I,
        [&](SoftFloat &SF, IRBuilderBase &B) {
          // A product at most 2^-125 (or NaN) of nonzero operands.
          Value *Tiny = B.CreateFCmpULE(
              B.CreateUnaryIntrinsic(Intrinsic::fabs, &I),
              SF.fconst(0x1p-125f));
          Value *ZX = zeroTest(I, 0, SF, B), *ZY = zeroTest(I, 1, SF, B);
          if (Value *P = chainPrev())
            return chainTiny(B, Tiny, ZX, ZY, P);
          return B.CreateAnd(Tiny, B.CreateNot(B.CreateOr(ZX, ZY)));
        },
        [&](SoftFloat &SF, IRBuilderBase &B) {
          Value *XB = SF.bits(X), *YB = SF.bits(Y);
          Value *Exact = SF.flt(SF.convert(
              SF.mul(SF.unpack(XB, false), SF.unpack(YB, false))));
          Value *Special = B.CreateFMul(SF.nonzeroNormal(X), SF.nonzeroNormal(Y));
          return B.CreateSelect(B.CreateOr(SF.isInfOrNaN(XB), SF.isInfOrNaN(YB)),
                                Special, Exact);
        });
    return true;
  }
  case Instruction::FDiv: {
    if (!isF32(&I) || Approx)
      return false;
    Value *X = I.getOperand(0), *Y = I.getOperand(1);
    // Round to nearest: the fast path is the target's own division (without
    // the features: correctly rounded, denormal operands and results flushed;
    // AGXUsage.rst "Strict floating point"), which is the IEEE result exactly
    // when both operands and the result are normal: a quotient below 2^-126
    // comes back as zero there (invalid: the slow path), and one that rounds
    // to 2^-126 is 2^-126 in both. 19 instructions against fastDiv's ~55.
    replaceWithSlowPath(
        I, [&](SoftFloat &SF, IRBuilderBase &B) { return SF.hwDiv(X, Y); },
        [&](SoftFloat &SF, IRBuilderBase &) { return SF.div(X, Y); });
    return true;
  }
  case Instruction::FCmp: {
    if (!isF32(I.getOperand(0)))
      return false;
    auto *Cmp = cast<FCmpInst>(&I);
    Value *X = Cmp->getOperand(0), *Y = Cmp->getOperand(1);
    CmpInst::Predicate P = Cmp->getPredicate();
    // (the hardware orders zeros and normal numbers)
    if (operandNotDenormal(I, 0) && operandNotDenormal(I, 1))
      return false;
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    Value *R = SF.compare(P, X, Y, &I);
    if (R == &I)
      return false;
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  default:
    break;
  }

  if (!isF32(&I))
    return false;
  auto *II = dyn_cast<IntrinsicInst>(&I);
  switch (ID) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd: {
    Value *X = II->getArgOperand(0), *Y = II->getArgOperand(1),
          *Z = II->getArgOperand(2);
    lowerWithSlowPath(
        I,
        [&](SoftFloat &SF, IRBuilderBase &B) {
          Value *XB = SF.bits(X), *YB = SF.bits(Y), *ZB = SF.bits(Z);
          Value *Den = nullptr;
          unsigned K = 0;
          for (Value *VB : {XB, YB, ZB}) {
            if (operandNotDenormal(I, K++))
              continue;
            Value *D = SF.isDenormal(VB);
            Den = Den ? B.CreateOr(Den, D) : D;
          }
          if (!Den)
            Den = B.getFalse();
          Value *Tiny = B.CreateFCmpULE(
              B.CreateUnaryIntrinsic(Intrinsic::fabs, &I), SF.fconst(0x1p-125f));
          Value *ZX = zeroTest(I, 0, SF, B), *ZY = zeroTest(I, 1, SF, B);
          if (Value *P = chainPrev())
            return B.CreateSelect(Den, B.getTrue(),
                                  chainTiny(B, Tiny, ZX, ZY, P));
          Tiny = B.CreateAnd(Tiny, B.CreateNot(B.CreateOr(ZX, ZY)));
          return B.CreateOr(Den, Tiny);
        },
        [&](SoftFloat &SF, IRBuilderBase &B) {
          Value *XB = SF.bits(X), *YB = SF.bits(Y), *ZB = SF.bits(Z);
          Term P = SF.mul(SF.unpack(XB, false), SF.unpack(YB, false));
          Value *Exact = SF.flt(SF.convert(SF.add(P, SF.unpack(ZB, false))));
          Value *Special = B.CreateIntrinsic(
              Intrinsic::fma, {B.getFloatTy()},
              {SF.nonzeroNormal(X), SF.nonzeroNormal(Y), SF.nonzeroNormal(Z)});
          Value *AnySpecial = B.CreateOr(
              SF.isInfOrNaN(XB), B.CreateOr(SF.isInfOrNaN(YB), SF.isInfOrNaN(ZB)));
          return B.CreateSelect(AnySpecial, Special, Exact);
        });
    return true;
  }
  case Intrinsic::sqrt: {
    if (Approx)
      return false;
    Value *X = II->getArgOperand(0);
    // Round to nearest: the target's own square root (correctly rounded for
    // every positive normal operand, whose root is normal) as the fast path.
    replaceWithSlowPath(
        I, [&](SoftFloat &SF, IRBuilderBase &) { return SF.hwSqrt(X); },
        [&](SoftFloat &SF, IRBuilderBase &) { return SF.sqrt(X); });
    return true;
  }
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum: {
    Value *X = II->getArgOperand(0), *Y = II->getArgOperand(1);
    bool Min = ID == Intrinsic::minnum || ID == Intrinsic::minimumnum ||
               ID == Intrinsic::minimum;
    bool Propagate = ID == Intrinsic::minimum || ID == Intrinsic::maximum;
    replace(I, [&](SoftFloat &SF, IRBuilderBase &) {
      return SF.minMax(X, Y, Min, Propagate);
    });
    return true;
  }
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::round: {
    // Denormals round to zero, except away from zero for floor and ceil.
    Value *X = II->getArgOperand(0);
    if (operandNotDenormal(I, 0))
      return false;
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    Value *XB = SF.bits(X);
    Value *Sign = B.CreateAnd(XB, SF.c32(0x80000000));
    Value *Fix = Sign;
    if (ID == Intrinsic::floor)
      Fix = B.CreateSelect(SF.signBit(XB), SF.c32(0xbf800000), SF.c32(0));
    if (ID == Intrinsic::ceil)
      Fix = B.CreateSelect(SF.signBit(XB), SF.c32(0x80000000), SF.c32(0x3f800000));
    Value *R = B.CreateSelect(SF.isDenormal(XB), SF.flt(Fix), &I);
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  case Intrinsic::agx_rcp:
  case Intrinsic::agx_rsqrt:
  case Intrinsic::log2:
  case Intrinsic::agx_log2: {
    // Denormal operands are scaled by 2^64 first.
    Value *X = II->getArgOperand(0);
    if (operandNotDenormal(I, 0)) {
      if (!(ID == Intrinsic::log2 || ID == Intrinsic::agx_log2))
        return false;
      replace(I, [&](SoftFloat &, IRBuilderBase &B) {
        return AGX::expandMath(B, Intrinsic::log2, {X});
      });
      return true;
    }
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    // (log2 of the strict mode: the precise expansion, within the rounding of the
    // exact result where the hardware's approximation may be off by one)
    bool Log2 = ID == Intrinsic::log2 || ID == Intrinsic::agx_log2;
    Value *Den = SF.isDenormal(SF.bits(X));
    Value *R;
    if (OneLog2Expansion) {
      // one evaluation, of the operand scaled where it is denormal
      Value *Op = B.CreateSelect(Den, SF.scaleUp64(X), X);
      Value *V = Log2 ? AGX::expandMath(B, Intrinsic::log2, {Op})
                      : B.CreateIntrinsic(ID, {B.getFloatTy()}, {Op});
      Value *Fixed;
      if (ID == Intrinsic::agx_rcp)
        Fixed = B.CreateFMul(V, SF.fconst(0x1p64f));
      else if (ID == Intrinsic::agx_rsqrt)
        Fixed = B.CreateFMul(V, SF.fconst(0x1p32f));
      else
        Fixed = B.CreateFSub(V, SF.fconst(64.0f));
      R = B.CreateSelect(Den, Fixed, V);
    } else {
    Value *Scaled = Log2 ? AGX::expandMath(B, Intrinsic::log2, {SF.scaleUp64(X)})
                         : B.CreateIntrinsic(ID, {B.getFloatTy()}, {SF.scaleUp64(X)});
    Value *Normal = Log2 ? AGX::expandMath(B, Intrinsic::log2, {X}) : static_cast<Value *>(&I);
    Value *Fixed;
    if (ID == Intrinsic::agx_rcp)
      Fixed = B.CreateFMul(Scaled, SF.fconst(0x1p64f));
    else if (ID == Intrinsic::agx_rsqrt)
      Fixed = B.CreateFMul(Scaled, SF.fconst(0x1p32f));
    else
      Fixed = B.CreateFSub(Scaled, SF.fconst(64.0f));
    R = B.CreateSelect(Den, Fixed, Normal);
    }
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  case Intrinsic::exp2:
  case Intrinsic::agx_exp2: {
    // 2^x = 2^f * 2^n with n = floor(x): the hardware computes 2^f in
    // [1, 2), and the scaling by 2^n is exact (rounded once for denormal
    // results), so integral x give exact powers of 2.
    Value *X = II->getArgOperand(0);
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    // Below -152 the result rounds to 0 anyway; clamp so that n fits.
    Value *XC = B.CreateBinaryIntrinsic(Intrinsic::maxnum, X, SF.fconst(-160.0f));
    XC = B.CreateBinaryIntrinsic(Intrinsic::minnum, XC, SF.fconst(160.0f));
    Value *N = B.CreateUnaryIntrinsic(Intrinsic::floor, XC);
    Value *Frac = B.CreateFSub(XC, N);
    Value *P = B.CreateIntrinsic(ID, {B.getFloatTy()}, {Frac});
    Value *NI = B.CreateFPToSI(N, B.getInt32Ty());
    Value *R;
    if (Exp2FastScale) {
      // n >= -126: 2^f in [1, 2] times 2^n is normal (or infinite past 2^127),
      // and the scaling is the exponent field's: one add. Below, the result
      // is denormal and rounds once: the integer scaling, rarely.
      Value *Fast = SF.flt(B.CreateAdd(SF.bits(P), B.CreateShl(NI, 23)));
      Value *Normal = B.CreateICmpSGE(NI, B.getInt32(-126));
      Instruction *Term0 = SplitBlockAndInsertIfThen(
          B.CreateNot(Normal), &*B.GetInsertPoint(), false,
          MDBuilder(I.getContext()).createUnlikelyBranchWeights());
      BasicBlock *Then = Term0->getParent(), *Head = Then->getSinglePredecessor();
      IRBuilder<> TB(Term0);
      SoftFloat TSF(TB);
      Term T = TSF.unpack(TSF.bits(P), false);
      T.E = TB.CreateAdd(T.E, NI);
      Value *Slow = TSF.flt(TSF.convert(T));
      BasicBlock *Join = Term0->getSuccessor(0);
      B.SetInsertPoint(Join, Join->getFirstInsertionPt());
      PHINode *Phi = B.CreatePHI(B.getFloatTy(), 2);
      Phi->addIncoming(Fast, Head);
      Phi->addIncoming(Slow, Then);
      R = Phi;
    } else {
      Term T = SF.unpack(SF.bits(P), false);
      T.E = B.CreateAdd(T.E, NI);
      R = SF.flt(SF.convert(T));
    }
    // NaN and infinities (2^-inf = 0) as the hardware gives them.
    Value *Special = B.CreateFCmpUEQ(B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
                                     SF.fconst(INFINITY));
    R = B.CreateSelect(Special, &I, R);
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::agx_sin:
  case Intrinsic::agx_cos: {
    // Below 2^-12, sin x and cos x round to x and 1 (the hardware flushes
    // denormal operands and loses the sign of zero).
    Value *X = II->getArgOperand(0);
    SmallVector<Use *, 4> Uses;
    for (Use &U : I.uses())
      Uses.push_back(&U);
    IRBuilder<> B(I.getNextNode());
    SoftFloat SF(B);
    Value *Small = B.CreateFCmpOLT(B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
                                   SF.fconst(0x1p-12f));
    bool Sin = ID == Intrinsic::sin || ID == Intrinsic::agx_sin;
    Value *R = B.CreateSelect(Small, Sin ? X : SF.fconst(1.0f), &I);
    for (Use *U : Uses)
      U->set(R);
    return true;
  }
  default:
    return false;
  }
}

/// Instructions built by the software implementations themselves, which
/// need no further lowering.
static constexpr const char *ExactTag = "agx.exact";
/// Instructions whose denormals follow IEEE although their function flushes
/// them (their operands and results were flushed explicitly if needed).
static constexpr const char *IEEETag = "agx.ieee";
/// Binary32 operations that keep denormals in any function and with or
/// without ieee-fp32 (AGXUsage.rst, "Floating point"): an IEEE operation the
/// front end needs in otherwise flushing code.
static constexpr const char *KeepDenormalsTag = "agx.keep-denormals";

/// Division and square root, which the hardware only approximates.
// With ieee-fp32, only the correctly rounded division and square root, every
// function's float denormals flushed: the part of the IEEE binary32 cost that
// is not denormals (agx-bench, EXPERIMENTS.md).
static cl::opt<bool> IEEEFP32NoDenormals(
    "agx-ieee-fp32-no-denormals", cl::Hidden, cl::init(false),
    cl::desc("With ieee-fp32, flush float denormals in every function (only "
             "division and square root correctly rounded)"));

static bool isCorrectlyRounded(const Instruction &I) {
  if (!isF32(&I) || (isa<FPMathOperator>(I) &&
                     (I.hasApproxFunc() || I.hasAllowReciprocal())))
    return false;
  return I.getOpcode() == Instruction::FDiv ||
         getIntrinsicID(I) == Intrinsic::sqrt;
}

/// A correctly rounded operation in a flushing function: the IEEE lowering
/// on flushed operands, with a flushed result.
static bool lowerFlushing(Instruction &I) {
  IRBuilder<> B(&I);
  SoftFloat SF(B);
  for (Use &Op : I.operands())
    if (Op->getType()->isFloatTy())
      Op.set(SF.flushDenormal(Op));
  SmallVector<Use *, 4> Uses;
  for (Use &U : I.uses())
    Uses.push_back(&U);
  B.SetInsertPoint(I.getNextNode());
  Value *Result = SF.flushDenormal(&I);
  for (Use *U : Uses)
    U->set(Result);
  lowerIEEE(I);
  return true;
}

/// bfloat arithmetic as float arithmetic on the extended operands, tagged
/// for the IEEE lowering (bfloat operations keep denormals, which the float
/// hardware flushes), with the result rounded back. Float has more than
/// twice the precision of bfloat, so rounding twice gives the correctly
/// rounded result. Returns false if I is not a bfloat operation.
static cl::opt<bool> BF16FMA(
    "agx-bf16-fma", cl::init(true), cl::Hidden,
    cl::desc("bfloat fma rounded once without ieee-fp32 too (not a float fma "
             "rounded again to bfloat)"));

static cl::opt<bool> BF16FastFMA(
    "agx-strict-bf16-fast-fma", cl::init(true), cl::Hidden,
    cl::desc("Strict bfloat fma in float where that is exact (the software "
             "fma otherwise)"));

/// bfloat fma, rounded once: in float where that is exact (BF16FastFMA), the
/// exact software fma behind an unlikely branch otherwise.
static void lowerBF16FMA(Instruction &I, bool CanonicalNaN) {
  // In float where that is exact: the product of two normal bfloats is
  // exact in float (8-bit significands) while it is normal; with it and
  // the addend zero or in [2^-102, 2^126), TwoSum gives the sum's exact
  // error (no value of it is denormal or overflows), the sum rounded to
  // odd (the odd neighbour where it is inexact) then rounded to nearest
  // bfloat rounds once, float keeping 16 bits more. Otherwise (bfloat
  // denormals, which the float hardware flushes, tiny or huge products,
  // infinities, NaNs) the exact software fma.
  Value *X0 = I.getOperand(0), *Y0 = I.getOperand(1), *Z0 = I.getOperand(2);
  replaceWithSlowPath(
      I,
      [&](SoftFloat &, IRBuilderBase &B) -> std::pair<Value *, Value *> {
        Type *F32 = B.getFloatTy(), *I32 = B.getInt32Ty();
        auto C = [&](uint32_t V) { return B.getInt32(V); };
        auto Bits = [&](Value *V) { return B.CreateBitCast(V, I32); };
        Value *X = B.CreateFPExt(X0, F32), *Y = B.CreateFPExt(Y0, F32),
              *Z = B.CreateFPExt(Z0, F32);
        auto Mag = [&](Value *V) {
          return B.CreateAnd(Bits(V), C(0x7fffffff));
        };
        // zero or in [Lo, Hi) (on the bits: a flushed denormal is not zero)
        auto In = [&](Value *M, uint32_t Lo, uint32_t Hi) {
          return B.CreateOr(
              B.CreateICmpEQ(M, C(0)),
              B.CreateICmpULT(B.CreateSub(M, C(Lo)), C(Hi - Lo)));
        };
        Value *MX = Mag(X), *MY = Mag(Y);
        Value *P = B.CreateFMul(X, Y);
        Value *ZeroOperand =
            B.CreateOr(B.CreateICmpEQ(MX, C(0)), B.CreateICmpEQ(MY, C(0)));
        Value *Valid = B.CreateAnd(In(MX, 0x00800000, 0x7f800000),
                                   In(MY, 0x00800000, 0x7f800000));
        // a nonzero product of nonzero operands in range (a zero product
        // of nonzero ones flushed)
        Value *MP = Mag(P);
        Value *POk =
            B.CreateSelect(ZeroOperand, B.getTrue(),
                           B.CreateICmpULT(B.CreateSub(MP, C(0x0c800000)),
                                           C(0x7e800000 - 0x0c800000)));
        Valid = B.CreateAnd(B.CreateAnd(Valid, POk),
                            In(Mag(Z), 0x0c800000, 0x7e800000));
        Value *S = B.CreateFAdd(P, Z);
        Value *BV = B.CreateFSub(S, P);
        Value *Err = B.CreateFAdd(B.CreateFSub(P, B.CreateFSub(S, BV)),
                                  B.CreateFSub(Z, BV));
        // to odd: one unit toward the exact value where S is inexact and
        // even (a unit on the bits is one on the magnitude, the sign kept)
        Value *SB = Bits(S);
        Value *Inexact = B.CreateFCmpUNE(Err, ConstantFP::get(F32, 0.0));
        Value *Even = B.CreateICmpEQ(B.CreateAnd(SB, C(1)), C(0));
        Value *Away =
            B.CreateICmpEQ(B.CreateICmpSLT(SB, C(0)),
                           B.CreateFCmpOLT(Err, ConstantFP::get(F32, 0.0)));
        Value *Odd =
            B.CreateSelect(B.CreateAnd(Inexact, Even),
                           B.CreateSelect(Away, B.CreateAdd(SB, C(1)),
                                          B.CreateSub(SB, C(1))),
                           SB);
        // to nearest even bfloat
        Value *Lsb = B.CreateAnd(B.CreateLShr(Odd, C(16)), C(1));
        Value *R = B.CreateLShr(B.CreateAdd(Odd, B.CreateAdd(Lsb, C(0x7fff))),
                                C(16));
        Value *Fast =
            B.CreateBitCast(B.CreateTrunc(R, B.getInt16Ty()), I.getType());
        return {Fast, Valid};
      },
      [&](SoftFloat &, IRBuilderBase &B) {
        return AGX::buildSoftFMA(B, X0, Y0, Z0, B.getInt32(RoundNearestEven),
                                 CanonicalNaN);
      });
}

static bool promoteBFloat(Instruction &I, bool CanonicalNaN) {
  bool IsCmp = isa<FCmpInst>(I);
  Type *Ty = IsCmp ? I.getOperand(0)->getType() : I.getType();
  if (!Ty->isBFloatTy())
    return false;
  Intrinsic::ID ID = getIntrinsicID(I);
  switch (ID) {
  case Intrinsic::not_intrinsic:
    if (!isa<BinaryOperator>(I) && !IsCmp)
      return false;
    if (isa<BinaryOperator>(I) && I.getOpcode() == Instruction::FRem)
      return false;
    break;
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::round:
  case Intrinsic::canonicalize:
    break;
  default:
    return false;
  }
  IRBuilder<> B(&I);
  if ((ID == Intrinsic::fma || ID == Intrinsic::fmuladd) && BF16FastFMA) {
    lowerBF16FMA(I, CanonicalNaN);
    return true;
  }
  if (ID == Intrinsic::fma || ID == Intrinsic::fmuladd) {
    // A float fma would round twice: compute it exactly.
    Value *R = AGX::buildSoftFMA(B, I.getOperand(0), I.getOperand(1),
                                 I.getOperand(2), B.getInt32(RoundNearestEven),
                                 CanonicalNaN);
    R->takeName(&I);
    I.replaceAllUsesWith(R);
    I.eraseFromParent();
    return true;
  }
  Type *F32 = B.getFloatTy();
  SmallVector<Value *, 3> Ops;
  unsigned N = isa<CallInst>(I) ? cast<CallInst>(I).arg_size() : I.getNumOperands();
  for (unsigned K = 0; K < N; ++K)
    Ops.push_back(B.CreateFPExt(I.getOperand(K), F32));
  Value *New;
  if (IsCmp)
    New = B.CreateFCmp(cast<FCmpInst>(I).getPredicate(), Ops[0], Ops[1]);
  else if (auto *BO = dyn_cast<BinaryOperator>(&I))
    New = B.CreateBinOp(BO->getOpcode(), Ops[0], Ops[1]);
  else
    New = B.CreateIntrinsic(ID, {F32}, Ops);
  auto *NI = cast<Instruction>(New);
  if (isa<FPMathOperator>(I))
    NI->copyFastMathFlags(&I);
  NI->setMetadata(IEEETag, MDNode::get(I.getContext(), {}));
  Value *R = IsCmp ? New : B.CreateFPTrunc(New, I.getType());
  R->takeName(&I);
  I.replaceAllUsesWith(R);
  I.eraseFromParent();
  return true;
}


static cl::opt<unsigned> IEEEGroupLimit(
    "agx-ieee-group-limit", cl::init(1), cl::Hidden,
    cl::desc("IEEE binary32 operations of a basic block checked together: the "
             "hardware results of a run of them, one unlikely branch to an "
             "exact recomputation of the whole run (1: one branch each)"));

/// Operations whose IEEE lowering is a fast hardware result with a validity
/// check (lowerWithSlowPath / replaceWithSlowPath): they can share one check.
static bool groupableIEEE(const Instruction &I) {
  if (!isF32(&I))
    return false;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
    return true;
  default:
    break;
  }
  Intrinsic::ID ID = getIntrinsicID(I);
  return ID == Intrinsic::fma || ID == Intrinsic::fmuladd;
}

/// Operations whose own IEEE lowering has a slow path or a long expansion:
/// a copy in a run's exact path would cost more than the run saves.
static bool slowIEEE(const Instruction &I) {
  if (I.getOpcode() == Instruction::FDiv || I.getOpcode() == Instruction::FRem)
    return true;
  switch (getIntrinsicID(I)) {
  case Intrinsic::sqrt:
  case Intrinsic::exp2:
  case Intrinsic::agx_exp2:
  case Intrinsic::log2:
  case Intrinsic::agx_log2:
    return true;
  default:
    return false;
  }
}

static bool pureForGroup(const Instruction &I) {
  // (a convergent operation, a SIMD-group shuffle say, must not be copied into
  // the divergent exact path)
  if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->isConvergent())
    return false;
  return !I.mayHaveSideEffects() && !I.mayReadOrWriteMemory() &&
         !isa<PHINode>(I) && !I.isTerminator() && !isa<AllocaInst>(I);
}

static cl::opt<bool> GroupCheckedExact(
    "agx-ieee-group-checked-exact", cl::init(true), cl::Hidden,
    cl::desc("IEEE runs checked together: the exact path is the run with "
             "each operation's own check and slow path (as without runs), not "
             "every operation computed in software (whose independent "
             "computations the scheduler interleaves, raising the register "
             "count)"));

/// Lowers the IEEE operations Members of one basic block (in order) with one
/// check: Region holds them and the pure instructions between them that
/// depend on them (in order), which the exact path recomputes. The copies of
/// the region's other operations of Worklist (an IEEE comparison, say) are
/// added to NewWork, which lowers them like the originals.
static void lowerIEEEGroup(ArrayRef<Instruction *> Region,
                           const SmallPtrSetImpl<Instruction *> &Members,
                           const SmallPtrSetImpl<Instruction *> &Work,
                           SmallVectorImpl<Instruction *> &NewWork) {
  Instruction *Last = Region.back();
  Instruction *Next = Last->getNextNode();
  BasicBlock *BB = Last->getParent();
  // the exact path's copies, made before anything changes
  ValueToValueMapTy VM;
  SmallVector<Instruction *, 32> Clones;
  for (Instruction *I : Region) {
    Instruction *C = I->clone();
    VM[I] = C;
    Clones.push_back(C);
    if (auto It = OperandNotDenormal.find(I); It != OperandNotDenormal.end())
      OperandNotDenormal[C] = It->second;
  }
  for (Instruction *C : Clones)
    RemapInstruction(C, VM, RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
  // fast results, and the conditions under which they are wrong (or-ed as
  // they come, so that each condition dies at once)
  SmallVector<WeakTrackingVH, 32> FastVals;
  for (Instruction *I : Region)
    FastVals.push_back(I);
  SPMode = SlowPathMode::Collect;
  SPInvalid.clear();
  Value *Bad = nullptr;
  for (Instruction *I : Region) {
    if (!Members.contains(I))
      continue;
    SPPrev = Bad;
    SPChained = false;
    lowerIEEE(*I);
    SPPrev = nullptr;
    // (the members' hardware results, never denormal, are right where the
    // run's check passes: their operand checks in the run can go)
    if (GroupNormal)
      GroupFast[I] = BB;
    for (Value *C : SPInvalid) {
      if (!Bad || SPChained) {
        // (a chained condition includes the run's so far)
        Bad = C;
        continue;
      }
      auto *CI = dyn_cast<Instruction>(C);
      IRBuilder<> OB(CI ? CI->getNextNode() : Next);
      Bad = OB.CreateOr(Bad, C);
    }
    SPInvalid.clear();
  }
  SPMode = SlowPathMode::Normal;
  if (!Bad) {
    for (Instruction *C : Clones)
      C->deleteValue();
    return;
  }
  if (isa<Constant>(Bad)) {
    // (a constant condition: or-ed into the next one's builder above)
    IRBuilder<> B(Next);
    Bad = B.CreateOr(Bad, B.getFalse());
  }
  Instruction *ThenTerm = SplitBlockAndInsertIfThen(
      Bad, Next->getIterator(), false,
      MDBuilder(Next->getContext()).createUnlikelyBranchWeights());
  BasicBlock *Then = ThenTerm->getParent();
  BasicBlock *Head = Then->getSinglePredecessor();
  BasicBlock *Join = Next->getParent();
  SmallVector<WeakTrackingVH, 32> ExactVals;
  for (Instruction *C : Clones) {
    C->insertBefore(ThenTerm->getIterator());
    ExactVals.push_back(C);
  }
  if (!GroupCheckedExact) {
    SPMode = SlowPathMode::Exact;
    for (auto [I, C] : zip(Region, Clones))
      if (Members.contains(I))
        lowerIEEE(*C);
    SPMode = SlowPathMode::Normal;
  }
  for (unsigned K = 0; K < Region.size(); ++K)
    if ((GroupCheckedExact && Members.contains(Region[K])) ||
        (!Members.contains(Region[K]) && Work.contains(Region[K])))
      if (Value *E = ExactVals[K])
        if (auto *EI = dyn_cast<Instruction>(E))
          NewWork.push_back(EI);
  IRBuilder<> B(Join, Join->begin());
  for (unsigned K = 0; K < Region.size(); ++K) {
    Value *F = FastVals[K], *E = ExactVals[K];
    if (!F || !E || F->getType()->isVoidTy())
      continue;
    PHINode *Phi = B.CreatePHI(F->getType(), 2);
    Phi->addIncoming(F, Head);
    Phi->addIncoming(E, Then);
    F->replaceUsesWithIf(Phi, [&](Use &U) {
      auto *UI = dyn_cast<Instruction>(U.getUser());
      return UI && UI != Phi && UI->getParent() != Head &&
             UI->getParent() != Then;
    });
    if (Phi->use_empty())
      Phi->eraseFromParent();
  }
}

static cl::opt<unsigned> IEEEGroupLiveIns(
    "agx-ieee-group-live-ins", cl::init(4), cl::Hidden,
    cl::desc("Most operands of a run of IEEE binary32 operations checked "
             "together that would otherwise die inside it (the exact path "
             "keeps them live to its end)"));

/// Groups the IEEE operations of each basic block of Ops (in program order)
/// for lowerIEEEGroup; returns the operations left for the per-operation
/// lowering. Copies that need their own lowering are added to NewWork.
static bool lowerIEEEGroups(Function &F, SmallVectorImpl<Instruction *> &Ops,
                            const SmallPtrSetImpl<Instruction *> &Work,
                            SmallVectorImpl<Instruction *> &NewWork) {
  SmallPtrSet<Instruction *, 32> Candidate;
  for (Instruction *I : Ops)
    if (groupableIEEE(*I))
      Candidate.insert(I);
  if (IEEEGroupLimit <= 1 || Candidate.size() < 2)
    return false;
  bool Changed = false;
  SmallVector<SmallVector<Instruction *, 32>, 8> Regions;
  SmallVector<SmallPtrSet<Instruction *, 32>, 8> MemberSets;
  for (BasicBlock &BB : F) {
    DenseMap<const Instruction *, unsigned> Pos;
    unsigned N = 0;
    for (Instruction &I : BB)
      Pos[&I] = N++;
    SmallVector<Instruction *, 32> Region;
    SmallPtrSet<Instruction *, 32> Dep, Mem;
    SmallSetVector<Instruction *, 16> Inputs;
    auto Close = [&]() {
      // trailing dependents after the last member are not part of it
      while (!Region.empty() && !Mem.contains(Region.back()))
        Region.pop_back();
      if (Mem.size() >= 2) {
        Regions.push_back(Region);
        MemberSets.push_back(Mem);
      }
      Region.clear();
      Dep.clear();
      Mem.clear();
      Inputs.clear();
    };
    // Operands from outside the run that die in it up to position End: the
    // exact path keeps them live to the run's end.
    auto Extended = [&](unsigned End) {
      unsigned K = 0;
      for (Instruction *O : Inputs) {
        bool Dies = all_of(O->users(), [&](User *U) {
          auto *UI = cast<Instruction>(U);
          return UI->getParent() == &BB && !isa<PHINode>(UI) &&
                 Pos.lookup(UI) <= End;
        });
        K += Dies;
      }
      return K;
    };
    auto AddInputs = [&](Instruction &I, SmallSetVector<Instruction *, 16> &To) {
      for (Use &U : I.operands())
        if (auto *OI = dyn_cast<Instruction>(U.get()); OI && !Dep.contains(OI))
          To.insert(OI);
    };
    for (Instruction &I : BB) {
      bool UsesDep = any_of(I.operands(), [&](Use &U) {
        auto *OI = dyn_cast<Instruction>(U.get());
        return OI && Dep.contains(OI);
      });
      if (Candidate.contains(&I)) {
        if (Mem.size() >= IEEEGroupLimit)
          Close();
        else if (!Mem.empty() && IEEEGroupLiveIns) {
          SmallSetVector<Instruction *, 16> Saved = Inputs;
          AddInputs(I, Inputs);
          unsigned K = Extended(Pos[&I]);
          Inputs = Saved;
          if (K > IEEEGroupLiveIns)
            Close();
        }
        AddInputs(I, Inputs);
        Region.push_back(&I);
        Dep.insert(&I);
        Mem.insert(&I);
      } else if (UsesDep) {
        if (pureForGroup(I) && !(Work.contains(&I) && slowIEEE(I))) {
          AddInputs(I, Inputs);
          Region.push_back(&I);
          Dep.insert(&I);
        } else {
          Close();
        }
      }
      if (I.isTerminator())
        Close();
    }
    Close();
  }
  for (auto [R, M] : zip(Regions, MemberSets)) {
    lowerIEEEGroup(R, M, Work, NewWork);
    for (Instruction *I : M)
      Candidate.erase(I);
    Changed = true;
  }
  // the grouped ones are done
  SmallPtrSet<Instruction *, 32> Done;
  for (auto &M : MemberSets)
    for (Instruction *I : M)
      Done.insert(I);
  erase_if(Ops, [&](Instruction *I) { return Done.contains(I); });
  return Changed;
}


bool AGXStrictFP::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  bool FunctionIEEE = F.getDenormalMode(APFloat::IEEEsingle()).Output ==
                          DenormalMode::IEEE &&
                      !IEEEFP32NoDenormals;
  bool CanonicalNaN = ST.hasCanonicalNaN();
  bool KeepDenormals = any_of(instructions(F), [](const Instruction &I) {
    return I.hasMetadata(KeepDenormalsTag);
  });
  bool Changed = false;
  // Canonical NaNs made on the bits: nothing where the value is canonicalized
  // here anyway, else a float comparison.
  {
    SmallPtrSet<Value *, 32> WillCanonicalize;
    if (CanonicalNaN)
      for (Instruction &I : instructions(F))
        if (!I.hasMetadata(ExactTag) && producesArithmeticNaN(I))
          WillCanonicalize.insert(&I);
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      auto *S = dyn_cast<SelectInst>(&I);
      Value *V = S ? matchBitsCanonicalNaN(*S) : nullptr;
      // (doubles have no float comparison here: AGXSoftF64 has already run)
      if (!V || V->getType()->getScalarType()->isDoubleTy())
        continue;
      // (the value unchanged, as bits or as the float)
      Value *R = S->getFalseValue();
      if (!WillCanonicalize.contains(V)) {
        IRBuilder<> B(S);
        const fltSemantics &Sem =
            V->getType()->getScalarType()->getFltSemantics();
        APFloat NaN(Sem, APInt::getSignedMaxValue(APFloat::getSizeInBits(Sem)));
        R = B.CreateBitCast(
            B.CreateSelect(B.CreateFCmpUNO(V, V),
                           ConstantFP::get(V->getType(), NaN), V),
            S->getType());
      }
      S->replaceAllUsesWith(R);
      R->takeName(S);
      Instruction *Cmp = dyn_cast<Instruction>(S->getCondition());
      S->eraseFromParent();
      if (Cmp)
        RecursivelyDeleteTriviallyDeadInstructions(Cmp);
      Changed = true;
    }
  }

  // bfloat fma rounded once in any mode (the float fma and a rounding to
  // bfloat would round twice); fmuladd, which may round the product, as the
  // bfloat multiplication and addition (each a float operation rounded once
  // to bfloat: the product is exact in float, and float keeps more than
  // twice bfloat's precision for the sum).
  if (!ST.hasIEEEFP32() && BF16FMA) {
    SmallVector<Instruction *, 4> FMAs;
    for (Instruction &I : instructions(F))
      if (I.getType()->isBFloatTy() &&
          (getIntrinsicID(I) == Intrinsic::fma ||
           getIntrinsicID(I) == Intrinsic::fmuladd))
        FMAs.push_back(&I);
    for (Instruction *I : FMAs) {
      if (getIntrinsicID(*I) == Intrinsic::fmuladd) {
        IRBuilder<> B(I);
        auto *Mul = cast<Instruction>(
            B.CreateFMul(I->getOperand(0), I->getOperand(1)));
        Mul->copyFastMathFlags(I);
        auto *Add = cast<Instruction>(B.CreateFAdd(Mul, I->getOperand(2)));
        Add->copyFastMathFlags(I);
        Add->takeName(I);
        I->replaceAllUsesWith(Add);
        I->eraseFromParent();
      } else {
        lowerBF16FMA(*I, CanonicalNaN);
      }
      Changed = true;
    }
  }

  if (!ST.hasIEEEFP32() && !CanonicalNaN && !KeepDenormals)
    return Changed;

  if (ST.hasIEEEFP32()) {
    SmallVector<Instruction *, 8> BFloat;
    for (Instruction &I : instructions(F))
      BFloat.push_back(&I);
    for (Instruction *I : BFloat)
      Changed |= promoteBFloat(*I, CanonicalNaN);
  }

  SmallVector<Instruction *, 32> Worklist;
  for (Instruction &I : instructions(F))
    if (!I.hasMetadata(ExactTag))
      Worklist.push_back(&I);

  recordOperandRanges(Worklist);
  // NaN canonicalization first: the IEEE lowering then replaces the
  // canonicalized value.
  if (CanonicalNaN) {
    SmallPtrSet<const Instruction *, 32> Canon;
    for (Instruction *I : Worklist)
      if (producesArithmeticNaN(*I))
        Canon.insert(I);
    SmallVector<Instruction *, 32> Skip;
    if (CanonicalNaNAtUses) {
      DenseMap<const Instruction *, bool> Memo;
      for (Instruction *I : Worklist)
        if (Canon.contains(I) && I->getType()->isFloatingPointTy() &&
            nanPayloadUnobserved(*I, Canon, Memo, 0))
          Skip.push_back(I);
    }
    SmallPtrSet<Instruction *, 32> SkipSet(Skip.begin(), Skip.end());
    for (Instruction *I : Worklist)
      if (Canon.contains(I) && !SkipSet.contains(I)) {
        canonicalizeNaN(*I);
        Changed = true;
      }
  }
  {
    // IEEE operations that can share one check (lowerIEEEGroups)
    SmallVector<Instruction *, 32> IEEEOps;
    for (Instruction *I : Worklist)
      if (!I->hasMetadata("air.flush.harmless") &&
          (I->hasMetadata(KeepDenormalsTag) ||
           (ST.hasIEEEFP32() && (FunctionIEEE || I->hasMetadata(IEEETag)))))
        IEEEOps.push_back(I);
    SmallVector<Instruction *, 32> Left(IEEEOps);
    SmallPtrSet<Instruction *, 32> Work(Worklist.begin(), Worklist.end());
    SmallVector<Instruction *, 16> NewWork;
    GroupFast.clear();
    if (lowerIEEEGroups(F, Left, Work, NewWork)) {
      Changed = true;
      SmallPtrSet<Instruction *, 32> Keep(Left.begin(), Left.end());
      SmallPtrSet<Instruction *, 32> IEEESet(IEEEOps.begin(), IEEEOps.end());
      erase_if(Worklist, [&](Instruction *I) {
        return IEEESet.contains(I) && !Keep.contains(I);
      });
      append_range(Worklist, NewWork);
    }
  }
  for (Instruction *I : Worklist) {
    // (a front end's operation whose result a flush does not change)
    if (I->hasMetadata("air.flush.harmless"))
      continue;
    if (I->hasMetadata(KeepDenormalsTag) ||
        (ST.hasIEEEFP32() && (FunctionIEEE || I->hasMetadata(IEEETag))))
      Changed |= lowerIEEE(*I);
    else if (ST.hasIEEEFP32() && isCorrectlyRounded(*I))
      Changed |= lowerFlushing(*I);
  }
  GroupFast.clear();
  OperandNotDenormal.clear();
  return Changed;
}

//===----------------------------------------------------------------------===//
// AGXLowerFPEnv
//===----------------------------------------------------------------------===//

namespace {

/// Models the floating point mode register of llvm.agx.set.fpmode and
/// llvm.agx.get.fpmode with a value per program point, and lowers the
/// constrained floating point intrinsics: those that round to nearest even
/// become ordinary operations, the others are computed in software with the
/// rounding direction of their metadata or of the mode. Operations whose
/// denormal mode differs from their function's flush their operands and
/// results explicitly, and are tagged agx.ieee.
class AGXLowerFPEnv : public FunctionPass {
public:
  static char ID;
  AGXLowerFPEnv() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Floating Point Environment";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

/// A builder that tags the floating point operations it creates.
struct TaggingBuilder : IRBuilder<ConstantFolder, IRBuilderCallbackInserter> {
  explicit TaggingBuilder(Instruction *IP)
      : IRBuilder(IP->getContext(), ConstantFolder(),
                  IRBuilderCallbackInserter([](Instruction *I) {
                    if (isa<FPMathOperator>(I) || isa<CallBase>(I) ||
                        isa<FPTruncInst>(I))
                      I->setMetadata(ExactTag, MDNode::get(I->getContext(), {}));
                  })) {
    SetInsertPoint(IP);
  }
};

} // namespace

char AGXLowerFPEnv::ID = 0;

INITIALIZE_PASS_BEGIN(AGXLowerFPEnv, "agx-lower-fp-env",
                      "AGX Lower Floating Point Environment", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXLowerFPEnv, "agx-lower-fp-env",
                    "AGX Lower Floating Point Environment", false, false)

FunctionPass *llvm::createAGXLowerFPEnvPass() { return new AGXLowerFPEnv(); }

static constexpr uint32_t ModeHalfShift = 2;
static constexpr uint32_t ModeFloatFlush = 0x10, ModeHalfFlush = 0x20;

static bool flushes(const Function &F, const fltSemantics &Sem) {
  DenormalMode M = F.getDenormalMode(Sem);
  return M.Output != DenormalMode::IEEE && M.Output != DenormalMode::Dynamic;
}

static uint32_t roundingCode(RoundingMode RM) {
  switch (RM) {
  case RoundingMode::TowardZero:
    return RoundTowardZero;
  case RoundingMode::TowardPositive:
    return RoundUp;
  case RoundingMode::TowardNegative:
    return RoundDown;
  default:
    return RoundNearestEven;
  }
}

/// The constrained intrinsic as the ordinary operation.
static Value *lowerToPlain(ConstrainedFPIntrinsic *CI, IRBuilderBase &B) {
  Function &F = *CI->getFunction();
  SmallVector<Value *, 4> Args(CI->args().begin(),
                               CI->args().begin() + CI->getNonMetadataArgCount());
  switch (CI->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:                                                   \
    if (Instruction::isBinaryOp(Instruction::NAME))                            \
      return B.CreateBinOp(Instruction::BinaryOps(Instruction::NAME), Args[0], \
                           Args[1]);                                           \
    return B.CreateCast(Instruction::CastOps(Instruction::NAME), Args[0],      \
                        CI->getType());
#define CMP_INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC, DAGN)              \
  case Intrinsic::INTRINSIC:                                                   \
    return B.CreateFCmp(cast<ConstrainedFPCmpIntrinsic>(CI)->getPredicate(),   \
                        Args[0], Args[1]);
#define FUNCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                           \
  case Intrinsic::INTRINSIC: {                                                 \
    SmallVector<Type *, 4> ArgTys;                                             \
    for (Value *A : Args)                                                      \
      ArgTys.push_back(A->getType());                                          \
    SmallVector<Type *, 2> OverloadTys;                                        \
    FunctionType *FT = FunctionType::get(CI->getType(), ArgTys, false);        \
    if (!Intrinsic::isSignatureValid(Intrinsic::NAME, FT, OverloadTys))        \
      return nullptr;                                                          \
    return B.CreateCall(Intrinsic::getOrInsertDeclaration(                     \
                            F.getParent(), Intrinsic::NAME, OverloadTys),      \
                        Args);                                                 \
  }
#include "llvm/IR/ConstrainedOps.def"
  default:
    return nullptr;
  }
}

/// The constrained intrinsic computed in software with the rounding
/// direction Rounding (an i32 RoundingCode), or null if it does not depend on
/// the rounding direction.
static Value *lowerRounded(ConstrainedFPIntrinsic *CI, Value *Rounding,
                           bool CanonicalNaN) {
  Type *Ty = CI->getType();
  Value *R = nullptr;
  auto Arg = [&](unsigned I) { return CI->getArgOperand(I); };
  bool Float = Ty->isFloatTy();
  {
    TaggingBuilder B(CI);
    SoftFloat SF(B);
    SF.Rounding = Rounding;
    switch (CI->getIntrinsicID()) {
    case Intrinsic::experimental_constrained_fadd:
    case Intrinsic::experimental_constrained_fsub:
    case Intrinsic::experimental_constrained_fmul:
      // with a fast path below
      break;
    case Intrinsic::experimental_constrained_fma:
    case Intrinsic::experimental_constrained_fmuladd:
      // with a fast path below
      break;
    case Intrinsic::experimental_constrained_sitofp:
    case Intrinsic::experimental_constrained_uitofp:
      if (Float || Ty->isHalfTy())
        R = SF.roundedFromInt(Arg(0), CI->getIntrinsicID() ==
                                          Intrinsic::experimental_constrained_sitofp,
                              Ty);
      break;
    case Intrinsic::experimental_constrained_fptrunc:
      if (Ty->isHalfTy() && Arg(0)->getType()->isFloatTy())
        R = SF.roundedToHalf(Arg(0));
      break;
    case Intrinsic::experimental_constrained_rint:
    case Intrinsic::experimental_constrained_nearbyint:
      R = SF.roundedIntegral(Arg(0));
      break;
    default:
      break;
    }
  }
  Intrinsic::ID ID = CI->getIntrinsicID();
  bool AddOrMul = Float && (ID == Intrinsic::experimental_constrained_fadd ||
                            ID == Intrinsic::experimental_constrained_fsub ||
                            ID == Intrinsic::experimental_constrained_fmul);
  bool Fma = Float && (ID == Intrinsic::experimental_constrained_fma ||
                       ID == Intrinsic::experimental_constrained_fmuladd);
  bool DivOrSqrt = Float && (ID == Intrinsic::experimental_constrained_fdiv ||
                             ID == Intrinsic::experimental_constrained_sqrt);
  if (!R && !DivOrSqrt && !AddOrMul && !Fma)
    return nullptr;
  if (Fma) {
    // Placeholder replaced below by the hardware fma corrected with the sign
    // of its exact error, or the result of integer arithmetic.
    Value *X = Arg(0), *Y = Arg(1), *Z = Arg(2);
    IRBuilder<> B(CI);
    Instruction *Stub = cast<Instruction>(B.CreateFreeze(X));
    replaceWithSlowPath<TaggingBuilder>(
        *Stub,
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return SF.fastFma(X, Y, Z);
        },
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return SF.roundedFma(X, Y, Z);
        });
    R = &*CI->getParent()->begin();
  }
  if (AddOrMul) {
    // Placeholder replaced below by the hardware result corrected with its
    // exact error, or the result of integer arithmetic.
    Value *X = Arg(0), *Y = Arg(1);
    bool Sub = ID == Intrinsic::experimental_constrained_fsub;
    bool Mul = ID == Intrinsic::experimental_constrained_fmul;
    IRBuilder<> B(CI);
    Instruction *Stub = cast<Instruction>(B.CreateFreeze(X));
    replaceWithSlowPath<TaggingBuilder>(
        *Stub,
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return Mul ? SF.fastMul(X, Y) : SF.fastAdd(X, Y, Sub);
        },
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return Mul ? SF.roundedMul(X, Y) : SF.roundedAdd(X, Y, Sub);
        });
    R = &*CI->getParent()->begin();
  }
  if (DivOrSqrt) {
    // Placeholder replaced below by the correctly rounded division or root.
    Value *X = Arg(0);
    Value *Y = CI->getIntrinsicID() == Intrinsic::experimental_constrained_fdiv
                   ? Arg(1)
                   : nullptr;
    IRBuilder<> B(CI);
    Instruction *Stub = cast<Instruction>(B.CreateFreeze(X));
    replaceWithSlowPath<TaggingBuilder>(
        *Stub,
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return Y ? SF.fastDiv(X, Y) : SF.fastSqrt(X);
        },
        [&](SoftFloat &SF, IRBuilderBase &) {
          SF.Rounding = Rounding;
          return Y ? SF.div(X, Y) : SF.sqrt(X);
        });
    // The phi that replaced the stub is the first instruction of the block
    // that now holds CI.
    R = &*CI->getParent()->begin();
  }
  if (CanonicalNaN) {
    IRBuilder<> B(CI);
    APFloat NaN(Ty->getFltSemantics(),
                APInt::getSignedMaxValue(Ty->getPrimitiveSizeInBits()));
    R = B.CreateSelect(B.CreateFCmpUNO(R, R), ConstantFP::get(Ty, NaN), R);
  }
  return R;
}

/// A vector constrained intrinsic as one scalar call per lane: the scalar
/// calls, or none if CI is not a vector operation.
static SmallVector<ConstrainedFPIntrinsic *, 4>
scalarizeConstrained(ConstrainedFPIntrinsic *CI) {
  SmallVector<ConstrainedFPIntrinsic *, 4> Lanes;
  auto *VT = dyn_cast<FixedVectorType>(CI->getType());
  if (!VT && CI->arg_size())
    VT = dyn_cast<FixedVectorType>(CI->getArgOperand(0)->getType());
  if (!VT)
    return Lanes;
  IRBuilder<> B(CI);
  Type *ResTy = CI->getType()->getScalarType();
  SmallVector<Type *, 4> ArgTys;
  for (Value *A : CI->args())
    ArgTys.push_back(A->getType()->getScalarType());
  SmallVector<Type *, 2> OverloadTys;
  FunctionType *FT = FunctionType::get(ResTy, ArgTys, false);
  if (!Intrinsic::isSignatureValid(CI->getIntrinsicID(), FT, OverloadTys))
    return Lanes;
  Function *Decl = Intrinsic::getOrInsertDeclaration(
      CI->getModule(), CI->getIntrinsicID(), OverloadTys);
  Value *R = PoisonValue::get(CI->getType());
  for (unsigned L = 0, E = VT->getNumElements(); L < E; ++L) {
    SmallVector<Value *, 4> Args;
    for (Value *A : CI->args())
      Args.push_back(A->getType()->isVectorTy() ? B.CreateExtractElement(A, L) : A);
    auto *Lane = cast<ConstrainedFPIntrinsic>(B.CreateCall(Decl, Args));
    Lane->setAttributes(CI->getAttributes());
    Lanes.push_back(Lane);
    R = CI->getType()->isVectorTy() ? B.CreateInsertElement(R, Lane, L) : Lane;
  }
  CI->replaceAllUsesWith(R);
  CI->eraseFromParent();
  return Lanes;
}

/// V with phi P taken to be C: a constant, or null if V depends on other
/// values (only the integer arithmetic of mode computations is followed).
static Constant *evaluateAssuming(Value *V, PHINode *P, Constant *C,
                                  const DataLayout &DL, unsigned Depth = 0) {
  if (V == P)
    return C;
  if (auto *K = dyn_cast<Constant>(V))
    return K;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || Depth > 32)
    return nullptr;
  if (auto *Fr = dyn_cast<FreezeInst>(I))
    return evaluateAssuming(Fr->getOperand(0), P, C, DL, Depth + 1);
  auto *BO = dyn_cast<BinaryOperator>(I);
  if (!BO)
    return nullptr;
  Constant *L = evaluateAssuming(BO->getOperand(0), P, C, DL, Depth + 1);
  Constant *R = L ? evaluateAssuming(BO->getOperand(1), P, C, DL, Depth + 1)
                  : nullptr;
  return R ? ConstantFoldBinaryOpOperands(BO->getOpcode(), L, R, DL) : nullptr;
}

static cl::opt<bool> FoldModePhis(
    "agx-fold-mode-phis", cl::init(true), cl::Hidden,
    cl::desc("Fold the FP mode phis of loops that set the mode and restore "
             "it to the constant they always hold"));

/// The constant that integer phi P always holds: the one constant among its
/// incoming values that every other incoming value gives back when P holds
/// it (by induction over the executions of P).
static Constant *constantModePhi(PHINode &P) {
  if (!FoldModePhis || !P.getType()->isIntegerTy())
    return nullptr;
  Constant *C = nullptr;
  for (Value *In : P.incoming_values())
    if (auto *K = dyn_cast<Constant>(In)) {
      if (C && C != K)
        return nullptr;
      C = K;
    }
  if (!C || isa<UndefValue>(C) || isa<ConstantExpr>(C))
    return nullptr;
  const DataLayout &DL = P.getDataLayout();
  for (Value *In : P.incoming_values())
    if (!isa<Constant>(In) && evaluateAssuming(In, &P, C, DL) != C)
      return nullptr;
  return C;
}

/// Whether the sign bit of V (a float or its bits) is clear: constants,
/// bitcasts, selects of such values, masks without the sign bit, fabs, and
/// maximumnum with a non-negative operand (it returns the other operand of
/// a NaN and orders -0 below +0). A front end's emulation of ftz and of NaN
/// results is selects of bit patterns, which ValueTracking does not follow.
static bool signBitClear(const Value *V, unsigned Depth = 0) {
  if (Depth > 8)
    return false;
  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->isNegative();
  if (auto *C = dyn_cast<ConstantInt>(V))
    return !C->isNegative();
  if (auto *BC = dyn_cast<BitCastInst>(V))
    return signBitClear(BC->getOperand(0), Depth + 1);
  if (auto *S = dyn_cast<SelectInst>(V))
    return signBitClear(S->getTrueValue(), Depth + 1) &&
           signBitClear(S->getFalseValue(), Depth + 1);
  if (auto *BO = dyn_cast<BinaryOperator>(V);
      BO && BO->getOpcode() == Instruction::And)
    return signBitClear(BO->getOperand(0), Depth + 1) ||
           signBitClear(BO->getOperand(1), Depth + 1);
  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    if (II->getIntrinsicID() == Intrinsic::fabs)
      return true;
    if (II->getIntrinsicID() == Intrinsic::maximumnum)
      return signBitClear(II->getArgOperand(0), Depth + 1) ||
             signBitClear(II->getArgOperand(1), Depth + 1);
  }
  return false;
}

/// CUDA's roundf: trunc(x + copysign(0.5, x)) with the addition rounded
/// toward zero (add.rz.ftz.f32, then cvt.rzi), or trunc(x + 0.5) once the
/// sign of x is known. That is round(x), half away from zero, for every x:
/// the rounding toward zero never carries x + 0.5 past the next integer,
/// from 2^23 on x is an integer that the addition gives back, and zeros,
/// flushed denormals, infinities and NaN agree, except that -0 + 0.5 gives
/// +0 (where the sign of x is not known to be clear, round(x) + 0). Applies
/// when the truncations of the addition are its only users.
static Value *lowerRoundf(ConstrainedFPIntrinsic &CI, Value *Rounding,
                          IRBuilderBase &B) {
  auto *RC = dyn_cast<ConstantInt>(Rounding);
  if (!RC || RC->getZExtValue() != RoundTowardZero ||
      CI.getIntrinsicID() != Intrinsic::experimental_constrained_fadd ||
      !CI.getType()->isFloatTy() || CI.use_empty())
    return nullptr;
  for (User *U : CI.users()) {
    auto *II = dyn_cast<IntrinsicInst>(U);
    if (!II || (II->getIntrinsicID() != Intrinsic::trunc &&
                II->getIntrinsicID() !=
                    Intrinsic::experimental_constrained_trunc))
      return nullptr;
  }
  auto IsHalf = [](const Value *V) {
    auto *C = dyn_cast<ConstantFP>(V);
    return C && (C->isExactlyValue(0.5) || C->isExactlyValue(-0.5));
  };
  for (unsigned I : {0u, 1u}) {
    Value *X = CI.getArgOperand(I), *H = CI.getArgOperand(1 - I);
    auto *C = dyn_cast<IntrinsicInst>(H);
    bool Copysign = C && C->getIntrinsicID() == Intrinsic::copysign &&
                    C->getArgOperand(1) == X && IsHalf(C->getArgOperand(0));
    auto *K = dyn_cast<ConstantFP>(H);
    bool PlusHalf = K && K->isExactlyValue(0.5);
    if (!Copysign && !PlusHalf)
      continue;
    // copysign(0.5, -0) + -0 is -0.5, which truncates to -0 as round(-0)
    if (Copysign || signBitClear(X))
      return B.CreateUnaryIntrinsic(Intrinsic::round, X);
    if (cannotBeOrderedLessThanZero(X, SimplifyQuery(CI.getDataLayout())))
      return B.CreateFAdd(B.CreateUnaryIntrinsic(Intrinsic::round, X),
                          ConstantFP::get(X->getType(), 0.0));
  }
  return nullptr;
}

bool AGXLowerFPEnv::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);

  SmallVector<IntrinsicInst *, 8> ModeOps;
  SmallVector<ConstrainedFPIntrinsic *, 16> Constrained;
  SmallVector<ConstrainedFPIntrinsic *, 4> Vector;
  for (Instruction &I : instructions(F)) {
    if (auto *CI = dyn_cast<ConstrainedFPIntrinsic>(&I))
      if (CI->getType()->isVectorTy() ||
          (CI->arg_size() && CI->getArgOperand(0)->getType()->isVectorTy()))
        Vector.push_back(CI);
      else
        Constrained.push_back(CI);
    else if (auto *II = dyn_cast<IntrinsicInst>(&I);
             II && (II->getIntrinsicID() == Intrinsic::agx_set_fpmode ||
                    II->getIntrinsicID() == Intrinsic::agx_get_fpmode))
      ModeOps.push_back(II);
  }
  // Vector operations are rounded lane by lane.
  for (ConstrainedFPIntrinsic *CI : Vector) {
    SmallVector<ConstrainedFPIntrinsic *, 4> Lanes = scalarizeConstrained(CI);
    if (Lanes.empty())
      Constrained.push_back(CI);
    else
      Constrained.append(Lanes.begin(), Lanes.end());
  }
  if (ModeOps.empty() && Constrained.empty())
    return false;

  // The mode as a variable, promoted to SSA values.
  bool FloatFlush = flushes(F, APFloat::IEEEsingle());
  bool HalfFlush = flushes(F, APFloat::IEEEhalf());
  uint32_t Initial = RoundNearestEven | (RoundNearestEven << ModeHalfShift) |
                     (FloatFlush ? ModeFloatFlush : 0) |
                     (HalfFlush ? ModeHalfFlush : 0);
  IRBuilder<> Entry(&*F.getEntryBlock().getFirstInsertionPt());
  AllocaInst *Mode = Entry.CreateAlloca(Entry.getInt32Ty());
  Entry.CreateStore(Entry.getInt32(Initial), Mode);
  for (IntrinsicInst *II : ModeOps) {
    IRBuilder<> B(II);
    if (II->getIntrinsicID() == Intrinsic::agx_set_fpmode) {
      B.CreateStore(II->getArgOperand(0), Mode);
    } else {
      Value *V = B.CreateLoad(B.getInt32Ty(), Mode);
      II->replaceAllUsesWith(V);
    }
    II->eraseFromParent();
  }
  // The mode at each constrained intrinsic, kept through the promotion by a
  // freeze.
  DenseMap<ConstrainedFPIntrinsic *, FreezeInst *> ModeAt;
  for (ConstrainedFPIntrinsic *CI : Constrained) {
    IRBuilder<> B(CI);
    ModeAt[CI] = cast<FreezeInst>(B.CreateFreeze(B.CreateLoad(B.getInt32Ty(), Mode)));
  }
  DominatorTree DT(F);
  PromoteMemToReg({Mode}, DT);
  // The mode is usually known: fold the integer arithmetic on it, so that the
  // operations below see constant rounding directions and denormal modes.
  // A loop that sets the mode and restores it (PTX's add.rz of roundf
  // followed by round-to-nearest code) gives its header a phi of the
  // incoming mode and a function of the phi itself; it is the constant when
  // the function gives that constant back.
  const DataLayout &DL = F.getDataLayout();
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (BasicBlock &BB : F)
      for (PHINode &P : make_early_inc_range(BB.phis()))
        if (Constant *C = constantModePhi(P)) {
          P.replaceAllUsesWith(C);
          P.eraseFromParent();
          Changed = true;
        }
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      if (!I.getType()->isIntegerTy() || isa<CallBase>(I) || isa<FreezeInst>(I) ||
          I.mayHaveSideEffects())
        continue;
      if (Constant *C = ConstantFoldInstruction(&I, DL)) {
        I.replaceAllUsesWith(C);
        I.eraseFromParent();
        Changed = true;
      }
    }
  }

  for (ConstrainedFPIntrinsic *CI : Constrained) {
    FreezeInst *Fr = ModeAt[CI];
    Value *Env = Fr->getOperand(0);
    IRBuilder<> B(CI);
    // The rounding of the result's format and the denormal mode of the
    // operands' format.
    Type *ResTy = CI->getType()->getScalarType();
    Type *OpTy = CI->getArgOperand(0)->getType()->getScalarType();
    bool ResHalf = !ResTy->isFloatTy();
    Value *Rounding;
    if (std::optional<RoundingMode> RM = CI->getRoundingMode();
        RM && *RM != RoundingMode::Dynamic)
      Rounding = B.getInt32(roundingCode(*RM));
    else
      Rounding = B.CreateAnd(ResHalf ? B.CreateLShr(Env, ModeHalfShift) : Env, 3);
    if (Value *R = lowerRoundf(*CI, Rounding, B)) {
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      continue;
    }
    bool HasFPOperand = OpTy->isFloatingPointTy();
    Type *FlushTy = HasFPOperand ? OpTy : ResTy;
    bool FlushFloat = FlushTy->isFloatTy();
    Value *Flush = B.CreateICmpNE(
        B.CreateAnd(Env, FlushFloat ? ModeFloatFlush : ModeHalfFlush),
        B.getInt32(0));
    bool Default = FlushFloat ? FloatFlush : HalfFlush;
    bool KnownDefault = isa<ConstantInt>(Flush) &&
                        cast<ConstantInt>(Flush)->isOne() == Default;
    bool Mismatch = !KnownDefault &&
                    (FlushTy->isFloatTy() || FlushTy->isHalfTy()) &&
                    !CI->getType()->isVectorTy();
    SoftFloat SF(B);
    bool Nearest = isa<ConstantInt>(Rounding) &&
                   cast<ConstantInt>(Rounding)->getZExtValue() == RoundNearestEven;
    bool Double = ResTy->isDoubleTy() || OpTy->isDoubleTy();
    // Directed rounding is computed in software, and the half precision ALU
    // keeps denormals (AGXCodeGenPrepare flushes plain half operations, which
    // these become only here): in a function that flushes them, flush the
    // operands and the result explicitly.
    if (!Mismatch && KnownDefault && Default && !Double &&
        (FlushTy->isHalfTy() || (FlushTy->isFloatTy() && !Nearest)) &&
        !CI->getType()->isVectorTy()) {
      Mismatch = true;
      Flush = B.getTrue();
    }
    // A denormal mode that differs from the function's: flush the operands
    // explicitly (and the result below).
    if (Mismatch)
      for (unsigned I = 0, E = CI->getNonMetadataArgCount(); I < E; ++I)
        if (CI->getArgOperand(I)->getType() == FlushTy)
          CI->setArgOperand(I, B.CreateSelect(Flush,
                                              SF.flushDenormal(CI->getArgOperand(I)),
                                              CI->getArgOperand(I)));

    Value *R = Nearest || Double
                   ? nullptr
                   : lowerRounded(CI, Rounding, ST.hasCanonicalNaN());
    if (!R) {
      B.SetInsertPoint(CI);
      R = lowerToPlain(CI, B);
      // Double precision is computed in software in any rounding direction.
      if (Double && !Nearest)
        if (auto *RI = dyn_cast_or_null<Instruction>(R))
          if (Value *D = AGX::lowerDoubleArithmetic(*RI, Rounding,
                                                    ST.hasCanonicalNaN()))
            R = D;
    }
    if (!R)
      continue;
    if (Mismatch) {
      if (auto *RI = dyn_cast<Instruction>(R))
        RI->setMetadata(IEEETag, MDNode::get(F.getContext(), {}));
      if (R->getType() == FlushTy) {
        B.SetInsertPoint(CI);
        R = B.CreateSelect(Flush, SF.flushDenormal(R), R);
      }
    }
    R->takeName(CI);
    CI->replaceAllUsesWith(R);
    CI->eraseFromParent();
  }
  for (auto &[CI, Fr] : ModeAt) {
    Fr->replaceAllUsesWith(Fr->getOperand(0));
    Fr->eraseFromParent();
  }
  return true;
}
