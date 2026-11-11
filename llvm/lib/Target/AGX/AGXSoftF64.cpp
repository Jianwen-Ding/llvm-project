//===-- AGXSoftF64.cpp - Double precision in software for AGX -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX has no double precision hardware. This pass replaces double precision
// arithmetic with integer arithmetic on the bits of the values: each operand
// becomes an exact sign * M * 2^E with a 128-bit M, the exact result (with a
// sticky bit where bits are dropped) is computed with integer operations, and
// a single rounding produces the bits of the result. The results are those of
// IEEE 754 with denormals, in all four rounding directions. Values that are
// only moved (loads, stores, phis, selects, arguments) stay double, and the
// type legalizer turns them into i64.
//
//===----------------------------------------------------------------------===//

#include "AGXSoftF64.h"
#include "AGX.h"
#include "AGXSubtarget.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "agx-soft-f64"

static cl::opt<bool> AGXSoftF64FastPaths(
    "agx-soft-f64-fast-paths", cl::init(true), cl::Hidden,
    cl::desc("Double precision arithmetic rounded to nearest takes a short "
             "path for normal operands and results"));

// A short path is code beside the general one: a function with many double
// operations (strict mode's exact texture sampler: ~2000 in one kernel) would
// take 40% more memory to compile (544 -> 764 MB); it keeps the general code
// alone.
static cl::opt<unsigned> AGXSoftF64FastPathsMax(
    "agx-soft-f64-fast-paths-max", cl::init(256), cl::Hidden,
    cl::desc("Most double precision operations per function that get a "
             "short path"));


namespace {

/// A binary interchange format.
struct Fmt {
  unsigned P; // significand bits, including the implicit one
  int Bias;
  unsigned W;
  int minLSB() const { return 2 - Bias - int(P); }
  uint64_t expAll() const { return 2 * Bias + 1; }
  uint64_t fracMask() const { return (uint64_t(1) << (P - 1)) - 1; }
  uint64_t infBits() const { return expAll() << (P - 1); }
  uint64_t signBit() const { return uint64_t(1) << (W - 1); }
  uint64_t maxFinite() const { return infBits() - 1; }
  uint64_t quietBit() const { return uint64_t(1) << (P - 2); }
};
constexpr Fmt F64 = {53, 1023, 64};
constexpr Fmt F32 = {24, 127, 32};
constexpr Fmt F16 = {11, 15, 16};
constexpr Fmt BF16 = {8, 127, 16};

enum RoundingCode : uint32_t {
  RoundTowardZero = 0,
  RoundNearestEven = 1,
  RoundUp = 2,
  RoundDown = 3,
};

/// sign * M * 2^E, with Sign i1, M i64 (or i128 for products) and E i32.
struct Term {
  Value *Sign;
  Value *M;
  Value *E;
};

class SoftDouble {
  IRBuilderBase &B;
  Type *I1, *I32, *I64, *I128;
  Value *Rounding;
  bool CanonicalNaN;

public:
  SoftDouble(IRBuilderBase &B, Value *Rounding, bool CanonicalNaN)
      : B(B), I1(B.getInt1Ty()), I32(B.getInt32Ty()), I64(B.getInt64Ty()),
        I128(B.getInt128Ty()), Rounding(Rounding), CanonicalNaN(CanonicalNaN) {}

  Value *c32(int64_t V) { return ConstantInt::getSigned(I32, V); }
  Value *c64(uint64_t V) { return B.getInt64(V); }
  Value *c128(uint64_t V) { return ConstantInt::get(I128, V); }
  Value *cw(Value *Like, uint64_t V) { return ConstantInt::get(Like->getType(), V); }
  static unsigned width(Value *M) { return M->getType()->getIntegerBitWidth(); }
  Value *roundingIs(RoundingCode C) {
    return B.CreateICmpEQ(Rounding, B.getInt32(C));
  }
  Value *select(Value *C, Value *T, Value *F) { return B.CreateSelect(C, T, F); }
  Value *smax(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::smax, A, X);
  }
  Value *umin64(Value *A, uint64_t X) {
    return B.CreateBinaryIntrinsic(Intrinsic::umin, A, cw(A, X));
  }
  Value *umin(Value *A, Value *X) {
    return B.CreateBinaryIntrinsic(Intrinsic::umin, A, X);
  }

  /// Bit length of M, as i32.
  Value *bitLength(Value *M) {
    Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, M, B.getFalse());
    return B.CreateTrunc(B.CreateSub(cw(M, width(M)), Lz), I32);
  }
  /// M << S and M >> S for an i32 S >= 0; amounts of the width of M or more
  /// give 0.
  Value *shl(Value *M, Value *S) {
    unsigned W = width(M);
    Value *R = B.CreateShl(M, B.CreateZExt(umin(S, c32(W - 1)), M->getType()));
    return select(B.CreateICmpUGT(S, c32(W - 1)), cw(M, 0), R);
  }
  Value *lshr(Value *M, Value *S) {
    unsigned W = width(M);
    Value *R = B.CreateLShr(M, B.CreateZExt(umin(S, c32(W - 1)), M->getType()));
    return select(B.CreateICmpUGT(S, c32(W - 1)), cw(M, 0), R);
  }
  /// Whether M has a bit set below bit S (an i32, any value).
  Value *lostBelow(Value *M, Value *S) {
    Value *Tz = B.CreateTrunc(B.CreateBinaryIntrinsic(Intrinsic::cttz, M, B.getFalse()), I32);
    return B.CreateAnd(B.CreateICmpSLT(Tz, S), B.CreateICmpNE(M, cw(M, 0)));
  }
  /// M >> S with the bits shifted out ORed into the lowest bit.
  Value *lshrSticky(Value *M, Value *S) {
    return B.CreateOr(lshr(M, S), B.CreateZExt(lostBelow(M, S), M->getType()));
  }
  /// M >> S for an M with its top bit clear, the shift amount clamped (an
  /// unsigned S of the width or more gives 0).
  Value *lshrTopClear(Value *M, Value *S) {
    return B.CreateLShr(M, B.CreateZExt(umin(S, c32(width(M) - 1)), M->getType()));
  }
  /// A term with a 128-bit M as one with a 64-bit M (and a sticky bit).
  Term compress(Term T) {
    Value *S = smax(B.CreateSub(bitLength(T.M), c32(63)), c32(0));
    return {T.Sign, B.CreateTrunc(lshrSticky(T.M, S), I64), B.CreateAdd(T.E, S)};
  }
  Term widen(Term T) { return {T.Sign, B.CreateZExt(T.M, I128), T.E}; }

  // Classification of bits (an i64 holding a value of format F).
  Value *magnitude(Value *Bits, Fmt F) {
    return B.CreateAnd(Bits, c64(F.signBit() - 1));
  }
  Value *signOf(Value *Bits, Fmt F) {
    return B.CreateICmpNE(B.CreateAnd(Bits, c64(F.signBit())), c64(0));
  }
  Value *isNaN(Value *Bits, Fmt F) {
    return B.CreateICmpUGT(magnitude(Bits, F), c64(F.infBits()));
  }
  Value *isInf(Value *Bits, Fmt F) {
    return B.CreateICmpEQ(magnitude(Bits, F), c64(F.infBits()));
  }
  Value *isZero(Value *Bits, Fmt F) {
    return B.CreateICmpEQ(magnitude(Bits, F), c64(0));
  }
  Value *isInfOrNaN(Value *Bits, Fmt F) {
    return B.CreateICmpUGE(magnitude(Bits, F), c64(F.infBits()));
  }

  /// The NaN result of an operation with the given operands of format In,
  /// in format Out: the first NaN operand quieted, or the default NaN if
  /// there is none. With canonical NaNs the default NaN is PTX's canonical
  /// one, and only double precision operations keep payloads (PTX ISA
  /// 9.7.3): conversions from or to other formats give the canonical NaN.
  Value *nanResult(ArrayRef<Value *> Ops, Fmt In, Fmt Out) {
    if (CanonicalNaN && (In.P != F64.P || Out.P != F64.P))
      return c64(Out.signBit() - 1);
    Value *R = CanonicalNaN ? c64(Out.signBit() - 1) : c64(Out.infBits() | Out.quietBit());
    for (Value *X : reverse(Ops)) {
      if (In.P == Out.P) {
        // the same format: a NaN quieted is itself with the quiet bit set
        R = select(isNaN(X, In), B.CreateOr(X, c64(Out.quietBit())), R);
        continue;
      }
      Value *Payload = B.CreateAnd(X, c64(In.fracMask()));
      Payload = In.P > Out.P ? B.CreateLShr(Payload, In.P - Out.P)
                             : B.CreateShl(Payload, Out.P - In.P);
      Value *Q = B.CreateOr(B.CreateOr(Payload, c64(Out.infBits() | Out.quietBit())),
                            select(signOf(X, In), c64(Out.signBit()), c64(0)));
      R = select(isNaN(X, In), Q, R);
    }
    return R;
  }

  Term unpack(Value *Bits, Fmt F) {
    Value *Exp = B.CreateAnd(B.CreateLShr(Bits, F.P - 1), c64(F.expAll()));
    Value *Frac = B.CreateAnd(Bits, c64(F.fracMask()));
    Value *Denormal = B.CreateICmpEQ(Exp, c64(0));
    Value *M = select(Denormal, Frac, B.CreateOr(Frac, c64(F.fracMask() + 1)));
    Value *E = B.CreateTrunc(select(Denormal, c64(1), Exp), I32);
    E = B.CreateSub(E, c32(F.Bias + int(F.P) - 1));
    return {signOf(Bits, F), M, E};
  }

  /// T rounded to format F in the rounding direction, as bits in an i64.
  /// T.M may hold a sticky bit below the bits that matter.
  Value *round(Term T, Fmt F) {
    Value *L = bitLength(T.M);
    Value *Shift = smax(B.CreateSub(L, c32(F.P)), B.CreateSub(c32(F.minLSB()), T.E));
    Value *Right = B.CreateICmpSGT(Shift, c32(0));
    Value *RS = smax(Shift, c32(0));
    Value *Q = select(Right, lshr(T.M, RS),
                      shl(T.M, B.CreateNeg(B.CreateBinaryIntrinsic(Intrinsic::smin, Shift, c32(0)))));
    // The guard bit is bit RS - 1 of M, and the sticky bits those below it.
    Value *GuardS = B.CreateSub(RS, c32(1));
    Value *Guard = B.CreateTrunc(lshr(T.M, GuardS), I1);
    Value *Sticky = lostBelow(T.M, GuardS);
    Guard = B.CreateAnd(Guard, Right);
    Value *Odd = B.CreateTrunc(Q, I1);
    Value *Inexact = B.CreateOr(Guard, Sticky);
    Value *IncNearest = B.CreateAnd(Guard, B.CreateOr(Sticky, Odd));
    Value *Inc = select(roundingIs(RoundNearestEven), IncNearest,
                        select(roundingIs(RoundUp), B.CreateAnd(Inexact, B.CreateNot(T.Sign)),
                               select(roundingIs(RoundDown), B.CreateAnd(Inexact, T.Sign),
                                      B.getFalse())));
    Q = B.CreateAdd(Q, B.CreateZExt(Inc, Q->getType()));
    // bits = k << (P - 1) + Q, with k the biased exponent of the lowest bit
    // of Q; a carry out of Q increments the exponent.
    Value *K = B.CreateSub(B.CreateAdd(T.E, Shift), c32(F.minLSB()));
    K = umin(K, c32(F.expAll() + 1));
    Value *Bits = B.CreateAdd(B.CreateShl(B.CreateZExt(K, I64), F.P - 1),
                              B.CreateZExtOrTrunc(Q, I64));
    Bits = select(B.CreateICmpEQ(T.M, cw(T.M, 0)), c64(0), Bits);
    Value *Overflow = B.CreateICmpUGE(Bits, c64(F.infBits()));
    // Overflows round to infinity or to the largest finite value.
    Value *ToInf = select(roundingIs(RoundNearestEven), B.getTrue(),
                          select(roundingIs(RoundUp), B.CreateNot(T.Sign),
                                 select(roundingIs(RoundDown), T.Sign, B.getFalse())));
    Bits = select(Overflow, select(ToInf, c64(F.infBits()), c64(F.maxFinite())), Bits);
    return B.CreateOr(Bits, select(T.Sign, c64(F.signBit()), c64(0)));
  }

  /// The exact sum of A and B (with M of at most 10 bits less than its
  /// width), with a sticky bit.
  Term addTerms(Term X, Term Y) {
    unsigned Top = width(X.M) - 2;
    auto Normalize = [&](Term T) {
      // At most Top places: a plain shift.
      Value *S = B.CreateSub(c32(Top), bitLength(T.M));
      return Term{T.Sign, B.CreateShl(T.M, B.CreateZExt(S, T.M->getType())), B.CreateSub(T.E, S)};
    };
    X = Normalize(X);
    Y = Normalize(Y);
    // Order by magnitude; zeros are the smallest.
    Value *XZero = B.CreateICmpEQ(X.M, cw(X.M, 0));
    Value *YZero = B.CreateICmpEQ(Y.M, cw(Y.M, 0));
    Value *YBigger = B.CreateOr(
        B.CreateICmpSGT(Y.E, X.E),
        B.CreateAnd(B.CreateICmpEQ(Y.E, X.E), B.CreateICmpUGT(Y.M, X.M)));
    YBigger = B.CreateAnd(B.CreateNot(YZero), B.CreateOr(XZero, YBigger));
    Term Big{select(YBigger, Y.Sign, X.Sign), select(YBigger, Y.M, X.M),
             select(YBigger, Y.E, X.E)};
    Term Small{select(YBigger, X.Sign, Y.Sign), select(YBigger, X.M, Y.M),
               select(YBigger, X.E, Y.E)};
    Value *D = B.CreateSub(Big.E, Small.E);
    Value *SM = B.CreateOr(lshrTopClear(Small.M, D),
                           B.CreateZExt(lostBelow(Small.M, D), Small.M->getType()));
    Value *Same = B.CreateICmpEQ(Big.Sign, Small.Sign);
    Value *M = select(Same, B.CreateAdd(Big.M, SM), B.CreateSub(Big.M, SM));
    return {Big.Sign, M, Big.E};
  }

  /// The sign of an exact zero sum of operands with signs SX and SY that
  /// are both zero (BothZero) or cancel.
  Value *zeroSumSign(Value *BothZero, Value *SX, Value *SY) {
    Value *Down = roundingIs(RoundDown);
    return select(BothZero, select(Down, B.CreateOr(SX, SY), B.CreateAnd(SX, SY)), Down);
  }

  Value *withZeroSign(Term T, Value *ZeroSign, Fmt F = F64) {
    // round() gives the sign of T to a zero.
    return round({select(B.CreateICmpEQ(T.M, cw(T.M, 0)), ZeroSign, T.Sign), T.M, T.E}, F);
  }

  /// The exact sum of the doubles X and Y (bits), with a sticky bit: the
  /// larger magnitude is the one with the larger bits, so no normalization
  /// is needed to order them; both significands go 10 places up (the sum of
  /// two of 63 bits fits) and the smaller one is aligned with a sticky bit.
  Term addF64(Value *X, Value *Y) {
    Value *YBigger = B.CreateICmpUGT(magnitude(Y, F64), magnitude(X, F64));
    Value *Big = select(YBigger, Y, X), *Small = select(YBigger, X, Y);
    Term TB = unpack(Big, F64), TS = unpack(Small, F64);
    Value *BM = B.CreateShl(TB.M, 10), *SM = B.CreateShl(TS.M, 10);
    Value *D = B.CreateSub(TB.E, TS.E);
    SM = B.CreateOr(lshrTopClear(SM, D), B.CreateZExt(lostBelow(SM, D), I64));
    Value *Same = B.CreateICmpEQ(TB.Sign, TS.Sign);
    Value *M = select(Same, B.CreateAdd(BM, SM), B.CreateSub(BM, SM));
    return {TB.Sign, M, B.CreateSub(TB.E, c32(10))};
  }

  Value *add(Value *X, Value *Y, bool Sub) {
    if (Sub)
      Y = B.CreateXor(Y, c64(F64.signBit()));
    Term TX = unpack(X, F64), TY = unpack(Y, F64);
    Term S = addF64(X, Y);
    Value *R = withZeroSign(
        S, zeroSumSign(B.CreateAnd(isZero(X, F64), isZero(Y, F64)), TX.Sign, TY.Sign));
    // Infinities: the infinite operand, or NaN for opposite infinities.
    Value *XInf = isInf(X, F64), *YInf = isInf(Y, F64);
    Value *Special = select(XInf, X, Y);
    Value *NaN = B.CreateOr(B.CreateOr(isNaN(X, F64), isNaN(Y, F64)),
                            B.CreateAnd(B.CreateAnd(XInf, YInf), B.CreateICmpNE(X, Y)));
    R = select(B.CreateOr(XInf, YInf), Special, R);
    Value *OrigY = Sub ? B.CreateXor(Y, c64(F64.signBit())) : Y;
    return select(NaN, nanResult({X, OrigY}, F64, F64), R);
  }

  /// Whether a double is zero or normal (finite, not denormal).
  Value *isNormalOrZero(Value *Bits) {
    return B.CreateOr(isNormal(Bits), isZero(Bits, F64));
  }
  /// Whether a double's exponent field is that of a normal number.
  /// (in 32 bits: the field is in the high word)
  Value *isNormal(Value *Bits) {
    Value *Hi = B.CreateTrunc(B.CreateLShr(Bits, 32), I32);
    Value *Exp = B.CreateAnd(B.CreateLShr(Hi, 20), c32(F64.expAll()));
    return B.CreateICmpULT(B.CreateSub(Exp, c32(1)), c32(F64.expAll() - 1));
  }
  /// The bits of a normal double rounded to nearest even from a 64-bit N
  /// with its top bit set (and a sticky bit in its low bits), of the value
  /// N 2^(E - 1086) with E the biased exponent it would have unrounded, and
  /// whether that is a normal result: the exponent field after the rounding
  /// carry within 1..2046.
  std::pair<Value *, Value *> roundNormalRNE(Value *N, Value *E) {
    Value *Sig = B.CreateLShr(N, 11);
    Value *Guard = B.CreateTrunc(B.CreateLShr(N, 10), I1);
    Value *Rest = B.CreateICmpNE(B.CreateAnd(N, c64(0x3ff)), c64(0));
    Value *Inc = B.CreateAnd(Guard, B.CreateOr(Rest, B.CreateTrunc(Sig, I1)));
    Sig = B.CreateAdd(Sig, B.CreateZExt(Inc, I64));
    // (E - 1) << 52 plus the significand with its implicit bit (a carry to
    // 2^53 adds one more to the exponent)
    Value *E64 = B.CreateSExt(E, I64);
    Value *Bits = B.CreateAdd(B.CreateShl(B.CreateSub(E64, c64(1)), 52), Sig);
    Value *Field = B.CreateLShr(B.CreateTrunc(B.CreateLShr(Bits, 32), I32), 20);
    Value *Valid = B.CreateAnd(
        B.CreateICmpSGE(E, c32(1)),
        B.CreateICmpULT(B.CreateSub(Field, c32(1)), c32(F64.expAll() - 1)));
    return {Bits, Valid};
  }
  /// X * Y rounded to nearest even where both are normal and so is the
  /// result, and whether they are: the significands' product has its top bit
  /// at bit 104 or 105, so its top 64 bits (with a sticky bit for the rest)
  /// need one conditional shift by one, not a normalization.
  std::pair<Value *, Value *> mulFast(Value *X, Value *Y) {
    Value *Implicit = c64(F64.fracMask() + 1);
    Value *MX = B.CreateOr(B.CreateAnd(X, c64(F64.fracMask())), Implicit);
    Value *MY = B.CreateOr(B.CreateAnd(Y, c64(F64.fracMask())), Implicit);
    Value *EX = B.CreateTrunc(B.CreateAnd(B.CreateLShr(X, 52), c64(F64.expAll())), I32);
    Value *EY = B.CreateTrunc(B.CreateAnd(B.CreateLShr(Y, 52), c64(F64.expAll())), I32);
    Value *P = B.CreateMul(B.CreateZExt(MX, I128), B.CreateZExt(MY, I128));
    Value *H = B.CreateTrunc(B.CreateLShr(P, 42), I64);
    Value *Low = B.CreateICmpNE(
        B.CreateAnd(B.CreateTrunc(P, I64), c64((uint64_t(1) << 42) - 1)), c64(0));
    Value *Top = B.CreateICmpSLT(H, c64(0)); // product >= 2^105
    Value *N = B.CreateOr(select(Top, H, B.CreateShl(H, 1)), B.CreateZExt(Low, I64));
    Value *E = B.CreateAdd(B.CreateSub(B.CreateAdd(EX, EY), c32(F64.Bias)),
                           B.CreateZExt(Top, I32));
    auto [Bits, Valid] = roundNormalRNE(N, E);
    Value *SignBits = B.CreateAnd(B.CreateXor(X, Y), c64(F64.signBit()));
    Bits = B.CreateOr(Bits, SignBits);
    // a zero operand: a zero with the sign of the product
    Value *AnyZero = B.CreateOr(isZero(X, F64), isZero(Y, F64));
    Bits = select(AnyZero, SignBits, Bits);
    Valid = B.CreateOr(AnyZero, Valid);
    return {Bits, B.CreateAnd(Valid, B.CreateAnd(isNormalOrZero(X), isNormalOrZero(Y)))};
  }
  /// X + Y rounded to nearest even where both are normal and so is the
  /// result (not zero), and whether they are. The significands go 10 places
  /// up, the smaller is aligned with a sticky bit, and the sum or difference
  /// is normalized with a count of leading zeros: a cancellation of more than
  /// two places needs exponents at most one apart, when nothing was shifted
  /// out, so the shifted sticky bit stays below the guard bit.
  std::pair<Value *, Value *> addFast(Value *X, Value *Y) {
    Value *YBigger = B.CreateICmpUGT(magnitude(Y, F64), magnitude(X, F64));
    Value *Big = select(YBigger, Y, X), *Small = select(YBigger, X, Y);
    Value *EB = B.CreateTrunc(B.CreateAnd(B.CreateLShr(Big, 52), c64(F64.expAll())), I32);
    Value *ES = B.CreateTrunc(B.CreateAnd(B.CreateLShr(Small, 52), c64(F64.expAll())), I32);
    // (a zero's significand 0: the implicit bit only with a nonzero
    // exponent, so that a zero operand adds nothing; denormals take the slow
    // path)
    auto Implicit = [&](Value *E) {
      return B.CreateShl(B.CreateZExt(B.CreateICmpNE(E, c32(0)), I64), 52);
    };
    Value *MB = B.CreateShl(B.CreateOr(B.CreateAnd(Big, c64(F64.fracMask())), Implicit(EB)), 10);
    Value *MS = B.CreateShl(B.CreateOr(B.CreateAnd(Small, c64(F64.fracMask())), Implicit(ES)), 10);
    Value *D = B.CreateSub(EB, ES);
    MS = B.CreateOr(lshrTopClear(MS, D), B.CreateZExt(lostBelow(MS, D), I64));
    Value *Same = B.CreateICmpSGE(B.CreateXor(X, Y), c64(0));
    Value *S = select(Same, B.CreateAdd(MB, MS), B.CreateSub(MB, MS));
    // (a zero S gives 64, clamped so that the shift is defined: the branch
    // on Valid must not see poison)
    Value *Lz = umin64(B.CreateBinaryIntrinsic(Intrinsic::ctlz, S, B.getFalse()), 63);
    Value *N = B.CreateShl(S, Lz);
    Value *E = B.CreateSub(B.CreateAdd(EB, c32(1)), B.CreateTrunc(Lz, I32));
    auto [Bits, Valid] = roundNormalRNE(N, E);
    Bits = B.CreateOr(Bits, B.CreateAnd(Big, c64(F64.signBit())));
    // a zero sum (an exact cancellation, or two zeros): +0, or -0 where both
    // are -0, the signs ANDed (a zero Small and a nonzero Big: Big, exactly)
    Value *SZero = B.CreateICmpEQ(S, c64(0));
    Bits = select(SZero, B.CreateAnd(B.CreateAnd(Big, Small), c64(F64.signBit())), Bits);
    Valid = B.CreateOr(SZero, Valid);
    return {Bits, B.CreateAnd(Valid, B.CreateAnd(isNormalOrZero(X), isNormalOrZero(Y)))};
  }

  /// X * Y + Z rounded to nearest even where all three are normal and so is
  /// the result (not zero), and whether they are. The product (top bit at bit
  /// 104 or 105) goes 20 places up and Z's significand 72, so both have their
  /// top bits at bit 124 or 125 and at least 20 zero bits below; the smaller
  /// is aligned with a sticky bit (shifts of up to 20 places lose nothing), the
  /// magnitude of the sum or difference is normalized once and its top 64 bits
  /// rounded. A cancellation of more than two places needs the alignment to be
  /// at most 20 places, when nothing was shifted out.
  std::pair<Value *, Value *> fmaFast(Value *X, Value *Y, Value *Z) {
    Value *Implicit = c64(F64.fracMask() + 1);
    // a zero's significand is 0 (and its exponent below any other's)
    auto Sig = [&](Value *V) {
      return select(isZero(V, F64), c64(0),
                    B.CreateOr(B.CreateAnd(V, c64(F64.fracMask())), Implicit));
    };
    auto Exp = [&](Value *V) {
      return B.CreateTrunc(B.CreateAnd(B.CreateLShr(V, 52), c64(F64.expAll())), I32);
    };
    Value *PZero = B.CreateOr(isZero(X, F64), isZero(Y, F64));
    Value *ZZero = isZero(Z, F64);
    Value *P = B.CreateMul(B.CreateZExt(Sig(X), I128), B.CreateZExt(Sig(Y), I128));
    Value *A = B.CreateShl(P, 20);
    Value *C = B.CreateShl(B.CreateZExt(Sig(Z), I128), 72);
    // exponents of the lowest bits of A and C
    Value *EA = B.CreateSub(B.CreateAdd(Exp(X), Exp(Y)), c32(2 * (F64.Bias + 52) + 20));
    Value *EC = B.CreateSub(Exp(Z), c32(F64.Bias + 52 + 72));
    EA = select(PZero, c32(-100000), EA);
    EC = select(ZZero, c32(-100000), EC);
    // only the one with the lower exponent is aligned
    Value *D = B.CreateSub(EA, EC);
    Value *Swap = B.CreateICmpSLT(D, c32(0));
    Value *Big = select(Swap, C, A), *Lo = select(Swap, A, C);
    Lo = lshrSticky(Lo, B.CreateBinaryIntrinsic(Intrinsic::abs, D, B.getFalse()));
    Value *Base = select(Swap, EC, EA);
    Value *SignP = B.CreateICmpSLT(B.CreateXor(X, Y), c64(0));
    Value *SignZ = B.CreateICmpSLT(Z, c64(0));
    Value *Same = B.CreateICmpEQ(SignP, SignZ);
    Value *S = select(Same, B.CreateAdd(Big, Lo), B.CreateSub(Big, Lo));
    Value *Neg = B.CreateICmpSLT(S, cw(S, 0));
    S = select(Neg, B.CreateNeg(S), S);
    Value *Sign = B.CreateXor(select(Swap, SignZ, SignP), Neg);
    Value *Lz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, S, B.getFalse());
    Lz = B.CreateBinaryIntrinsic(Intrinsic::umin, Lz, cw(S, 127));
    Value *Norm = B.CreateShl(S, Lz);
    Value *Hi = B.CreateTrunc(B.CreateLShr(Norm, 64), I64);
    Value *N = B.CreateOr(Hi, B.CreateZExt(B.CreateICmpNE(B.CreateTrunc(Norm, I64), c64(0)), I64));
    Value *E = B.CreateAdd(B.CreateSub(Base, B.CreateTrunc(Lz, I32)), c32(64 + 1086));
    auto [Bits, Valid] = roundNormalRNE(N, E);
    Bits = B.CreateOr(Bits, select(Sign, c64(F64.signBit()), c64(0)));
    // an exact zero: -0 only for a zero product and a zero Z both negative,
    // +0 for a cancellation
    Value *SZero = B.CreateICmpEQ(S, cw(S, 0));
    Value *ZeroSign = B.CreateAnd(B.CreateAnd(PZero, ZZero), B.CreateAnd(SignP, SignZ));
    Bits = select(SZero, select(ZeroSign, c64(F64.signBit()), c64(0)), Bits);
    Valid = B.CreateOr(Valid, SZero);
    Valid = B.CreateAnd(Valid, B.CreateAnd(isNormalOrZero(X),
                                           B.CreateAnd(isNormalOrZero(Y), isNormalOrZero(Z))));
    return {Bits, Valid};
  }

  /// A float (bits in an i64) as a double, where it is normal or zero, and
  /// whether it is: the exponent rebiased and the fraction 29 places up.
  std::pair<Value *, Value *> extFast(Value *X) {
    // in 32 bits: the float's fields go to the double's high word (the top 20
    // fraction bits) and low word (the other 3, at its top)
    Value *F = B.CreateTrunc(X, I32);
    Value *E = B.CreateAnd(B.CreateLShr(F, 23), c32(0xff));
    Value *Frac = B.CreateAnd(F, c32(0x7fffff));
    Value *Zero = B.CreateICmpEQ(B.CreateAnd(F, c32(0x7fffffff)), c32(0));
    Value *Hi = B.CreateOr(B.CreateShl(B.CreateAdd(E, c32(1023 - 127)), 20),
                           B.CreateLShr(Frac, 3));
    Value *Lo = B.CreateShl(Frac, 29);
    Hi = B.CreateSelect(Zero, c32(0), Hi);
    Lo = B.CreateSelect(Zero, c32(0), Lo);
    Hi = B.CreateOr(Hi, B.CreateAnd(F, c32(int32_t(0x80000000u))));
    Value *Bits = B.CreateOr(B.CreateShl(B.CreateZExt(Hi, I64), 32),
                             B.CreateZExt(Lo, I64));
    Value *Normal = B.CreateICmpULT(B.CreateSub(E, c32(1)), c32(254));
    return {Bits, B.CreateOr(Normal, Zero)};
  }
  /// A double rounded to nearest even to a float (bits in an i64), where the
  /// double is zero or the result normal, and whether it is.
  std::pair<Value *, Value *> truncFast(Value *X) {
    // in 32 bits: the exponent field is in the high word, the 23 bits kept
    // span both, the rest is in the low word
    Value *Hi = B.CreateTrunc(B.CreateLShr(X, 32), I32);
    Value *Lo = B.CreateTrunc(X, I32);
    Value *E = B.CreateAnd(B.CreateLShr(Hi, 20), c32(F64.expAll()));
    Value *M = B.CreateTrunc(
        B.CreateLShr(B.CreateAnd(X, c64(F64.fracMask())), 29), I32);
    Value *Rest = B.CreateAnd(Lo, c32((1 << 29) - 1));
    Value *Half = c32(1 << 28);
    Value *Inc = B.CreateOr(B.CreateICmpUGT(Rest, Half),
                            B.CreateAnd(B.CreateICmpEQ(Rest, Half), B.CreateTrunc(M, I1)));
    // the exponent field E - 896, the rounding carry into it (out of range,
    // the sum may wrap, but the field check fails)
    Value *EF = B.CreateSub(E, c32(1023 - 127));
    Value *Bits = B.CreateAdd(B.CreateAdd(B.CreateShl(EF, 23), M),
                              B.CreateZExt(Inc, I32));
    Value *Valid =
        B.CreateAnd(B.CreateICmpULT(B.CreateSub(EF, c32(1)), c32(254)),
                    B.CreateICmpULT(Bits, c32(0x7f800000)));
    Value *Zero = isZero(X, F64);
    Bits = B.CreateSelect(Zero, c32(0), Bits);
    Bits = B.CreateOr(Bits, B.CreateAnd(Hi, c32(int32_t(0x80000000u))));
    return {B.CreateZExt(Bits, I64), B.CreateOr(Valid, Zero)};
  }
  /// A double truncated to an integer of type Ty where it fits (finite, of
  /// a magnitude below 2^(W - 1), or below 2^W and not negative unsigned),
  /// and whether it does; a negative value is 0 unsigned (saturating).
  std::pair<Value *, Value *> toIntFast(Value *X, Type *Ty, bool Signed) {
    unsigned W = Ty->getIntegerBitWidth();
    Value *E = B.CreateTrunc(B.CreateAnd(B.CreateLShr(X, 52), c64(F64.expAll())), I32);
    Value *M = B.CreateOr(B.CreateAnd(X, c64(F64.fracMask())), c64(F64.fracMask() + 1));
    // X = M 2^(E - 1075); E below 1023 is below 1 (zero, denormal or not):
    // the shifts give 0
    Value *Sh = B.CreateSub(E, c32(F64.Bias + 52));
    // (where it fits, a left shift by at most W - 53 places; a right shift
    // by 53 or more gives 0, so 63 places do)
    Value *Left = B.CreateShl(M, B.CreateZExt(B.CreateAnd(Sh, c32(63)), I64));
    Value *Right = B.CreateLShr(
        M, B.CreateZExt(B.CreateBinaryIntrinsic(Intrinsic::umin,
                                                B.CreateNeg(Sh), c32(63)),
                        I64));
    Value *Mag = select(B.CreateICmpSGT(Sh, c32(0)), Left, Right);
    Value *Neg = B.CreateICmpSLT(X, c64(0));
    Value *Fits = B.CreateICmpULT(E, c32(F64.Bias + W - (Signed ? 1 : 0)));
    Value *R;
    if (Signed) {
      R = select(Neg, B.CreateNeg(Mag), Mag);
    } else {
      R = select(Neg, c64(0), Mag);
      // any negative value (not NaN) saturates to 0
      Fits = B.CreateOr(Fits, B.CreateAnd(Neg, B.CreateICmpULT(E, c32(F64.expAll()))));
    }
    return {B.CreateTrunc(R, Ty), Fits};
  }
  /// An integer as a double where it is exact (a magnitude below 2^53), and
  /// whether it is: one count of leading zeros places its top bit at 52.
  std::pair<Value *, Value *> fromIntFast(Value *V, bool Signed) {
    Value *V64 = Signed ? B.CreateSExt(V, I64) : B.CreateZExt(V, I64);
    Value *Neg = Signed ? B.CreateICmpSLT(V64, c64(0)) : B.getFalse();
    Value *A = select(Neg, B.CreateNeg(V64), V64);
    // (the counts in 32 bits)
    Value *Lz = B.CreateTrunc(
        umin64(B.CreateBinaryIntrinsic(Intrinsic::ctlz, A, B.getFalse()), 63),
        I32);
    Value *Sh = B.CreateSub(Lz, c32(11));
    Value *Valid = B.CreateICmpUGE(Lz, c32(11));
    // (only used where valid: a shift of at most 52 places)
    Value *M = B.CreateShl(
        A, B.CreateZExt(B.CreateBinaryIntrinsic(Intrinsic::umin, Sh, c32(52)),
                        I64));
    // biased exponent 1075 - Sh, minus one for the implicit bit added by M
    Value *Bits = B.CreateAdd(
        B.CreateShl(B.CreateZExt(B.CreateSub(c32(F64.Bias + 52 - 1), Sh), I64),
                    52),
        M);
    Bits = select(B.CreateICmpEQ(A, c64(0)), c64(0), Bits);
    Bits = B.CreateOr(Bits, select(Neg, c64(F64.signBit()), c64(0)));
    return {Bits, B.CreateOr(Valid, B.CreateICmpEQ(A, c64(0)))};
  }

  /// T (a double's significand) with its top bit at bit 52, unless it is 0.
  Term normalize53(Term T) {
    Value *S = B.CreateSub(c32(F64.P), bitLength(T.M));
    return {T.Sign, B.CreateShl(T.M, B.CreateZExt(S, I64)), B.CreateSub(T.E, S)};
  }

  Value *mul(Value *X, Value *Y) {
    // With both significands normalized the product has 105 or 106 bits (or
    // is 0): its top 64 bits and a sticky bit for the 42 below are exact
    // enough to round, without finding its length.
    Term TX = normalize53(unpack(X, F64)), TY = normalize53(unpack(Y, F64));
    Value *Sign = B.CreateXor(TX.Sign, TY.Sign);
    Value *P = B.CreateMul(B.CreateZExt(TX.M, I128), B.CreateZExt(TY.M, I128));
    Value *Low = B.CreateAnd(B.CreateTrunc(P, I64), c64((uint64_t(1) << 42) - 1));
    Value *M = B.CreateOr(B.CreateTrunc(B.CreateLShr(P, 42), I64),
                          B.CreateZExt(B.CreateICmpNE(Low, c64(0)), I64));
    Value *R = round({Sign, M, B.CreateAdd(B.CreateAdd(TX.E, TY.E), c32(42))}, F64);
    Value *XInf = isInf(X, F64), *YInf = isInf(Y, F64);
    Value *Inf = B.CreateOr(c64(F64.infBits()), select(Sign, c64(F64.signBit()), c64(0)));
    Value *NaN = B.CreateOr(
        B.CreateOr(isNaN(X, F64), isNaN(Y, F64)),
        B.CreateOr(B.CreateAnd(XInf, isZero(Y, F64)), B.CreateAnd(YInf, isZero(X, F64))));
    R = select(B.CreateOr(XInf, YInf), Inf, R);
    return select(NaN, nanResult({X, Y}, F64, F64), R);
  }

  Value *fma(Value *X, Value *Y, Value *Z, Fmt F = F64) {
    Term TX = unpack(X, F), TY = unpack(Y, F), TZ = unpack(Z, F);
    Term P{B.CreateXor(TX.Sign, TY.Sign),
           B.CreateMul(B.CreateZExt(TX.M, I128), B.CreateZExt(TY.M, I128)),
           B.CreateAdd(TX.E, TY.E)};
    // round() takes the 128-bit sum as it is (no compress to 64 bits first)
    Term S = addTerms(P, widen(TZ));
    Value *PZero = B.CreateOr(isZero(X, F), isZero(Y, F));
    Value *R = withZeroSign(
        S, zeroSumSign(B.CreateAnd(PZero, isZero(Z, F)), P.Sign, TZ.Sign), F);
    Value *XInf = isInf(X, F), *YInf = isInf(Y, F), *ZInf = isInf(Z, F);
    Value *PInf = B.CreateOr(XInf, YInf);
    Value *PInfBits = B.CreateOr(c64(F.infBits()),
                                 select(P.Sign, c64(F.signBit()), c64(0)));
    Value *NaN = B.CreateOr(B.CreateOr(isNaN(X, F), isNaN(Y, F)), isNaN(Z, F));
    NaN = B.CreateOr(NaN, B.CreateOr(B.CreateAnd(XInf, isZero(Y, F)),
                                     B.CreateAnd(YInf, isZero(X, F))));
    NaN = B.CreateOr(NaN, B.CreateAnd(B.CreateAnd(PInf, ZInf),
                                      B.CreateICmpNE(P.Sign, TZ.Sign)));
    R = select(ZInf, Z, R);
    R = select(PInf, PInfBits, R);
    return select(NaN, nanResult({X, Y, Z}, F, F), R);
  }

  /// An f32 estimate of an i64 below 2^24, through an AGX approximation
  /// (agx_rcp or agx_rsqrt, within a few units of the last place), scaled by
  /// 2^Scale and lowered by 2^-19 so that it is below the exact value, as an
  /// i64.
  Value *estimate(Intrinsic::ID ID, Value *V, int Scale) {
    Type *F32 = B.getFloatTy();
    Value *F = B.CreateUIToFP(B.CreateTrunc(V, I32), F32);
    Value *Est = B.CreateIntrinsic(ID, {F32}, {F});
    Est = B.CreateFMul(Est, ConstantFP::get(F32, std::ldexp(1.0 - 0x1p-19, Scale)));
    return B.CreateZExt(B.CreateFPToUI(Est, I32), I64);
  }
  Value *mulWide(Value *X, Value *Y, bool Signed = false) {
    auto Ext = [&](Value *V, bool S) {
      return S ? B.CreateSExt(V, I128) : B.CreateZExt(V, I128);
    };
    return B.CreateMul(Ext(X, false), Ext(Y, Signed));
  }
  Value *hi(Value *X, unsigned S, bool Signed = false) {
    return B.CreateTrunc(Signed ? B.CreateAShr(X, S) : B.CreateLShr(X, S), I64);
  }

  /// floor(N * 2^54 / Dv) for N and Dv with their top bits at 52 (N at 53
  /// where it is the smaller), doubled with a sticky bit for the remainder
  /// (56 bits). A reciprocal from below: Y0 ~ 2^82 / Dv to 2^-18, then two
  /// Newton steps, Z ~ 2^115 / Dv to 2^-36 and Z2 to 2^-60 (each step squares
  /// the error and stays below). Q is then at most 2 above (N * Z2) >> 61,
  /// and the exact remainder corrects it.
  Value *quotientSticky(Value *N, Value *Dv) {
    Value *Y0 = estimate(Intrinsic::agx_rcp, B.CreateLShr(Dv, 29), 53);
    Value *E0 = B.CreateNeg(B.CreateMul(Dv, Y0)); // 2^82 - Dv * Y0
    Value *Z = B.CreateAdd(B.CreateShl(Y0, 33), hi(mulWide(Y0, E0), 49));
    // E1 = 2^115 - Dv * Z (at least 0: Z stays below) in units of 2^53, from
    // the high word P of (Dv << 11) * Z (Dv < 2^53): 2^62 - 1 - P is below
    // it by less than one unit (2^53, inside the 2^55 that Z2 to 2^-60
    // tolerates), so that Z2 stays below too; clamped at 0. No 128-bit
    // product.
    Value *E1 = B.CreateBinaryIntrinsic(
        Intrinsic::smax,
        B.CreateSub(c64((uint64_t(1) << 62) - 1),
                    hi(mulWide(B.CreateShl(Dv, 11), Z), 64)),
        c64(0));
    Value *Z2 = B.CreateAdd(Z, hi(mulWide(Z, B.CreateShl(E1, 37)), 99));
    Value *Q = hi(mulWide(N, Z2), 61);
    Value *R = B.CreateSub(B.CreateShl(N, 54), B.CreateMul(Q, Dv));
    for (int I = 0; I < 2; ++I) {
      Value *Ge = B.CreateICmpUGE(R, Dv);
      Q = B.CreateAdd(Q, B.CreateZExt(Ge, I64));
      R = select(Ge, B.CreateSub(R, Dv), R);
    }
    Value *Inexact = B.CreateICmpNE(R, c64(0));
    return B.CreateOr(B.CreateShl(Q, 1), B.CreateZExt(Inexact, I64));
  }

  /// X / Y rounded to nearest even where both are normal and so is the
  /// result, and whether they are: the significands need no normalization
  /// and the 56-bit quotient goes 8 places up to be rounded.
  std::pair<Value *, Value *> divFast(Value *X, Value *Y) {
    Value *Implicit = c64(F64.fracMask() + 1);
    Value *MX = B.CreateOr(B.CreateAnd(X, c64(F64.fracMask())), Implicit);
    Value *MY = B.CreateOr(B.CreateAnd(Y, c64(F64.fracMask())), Implicit);
    Value *Less = B.CreateICmpULT(MX, MY);
    Value *N = select(Less, B.CreateShl(MX, 1), MX);
    Value *M = quotientSticky(N, MY);
    Value *EX = B.CreateTrunc(B.CreateAnd(B.CreateLShr(X, 52), c64(F64.expAll())), I32);
    Value *EY = B.CreateTrunc(B.CreateAnd(B.CreateLShr(Y, 52), c64(F64.expAll())), I32);
    // M 2^(EX - EY - 55 - Less), M << 8 as N 2^(E - 1086)
    Value *E = B.CreateSub(B.CreateAdd(B.CreateSub(EX, EY), c32(F64.Bias)),
                           B.CreateZExt(Less, I32));
    auto [Bits, Valid] = roundNormalRNE(B.CreateShl(M, 8), E);
    Value *SignBits = B.CreateAnd(B.CreateXor(X, Y), c64(F64.signBit()));
    Bits = B.CreateOr(Bits, SignBits);
    // a zero dividend: a zero with the quotient's sign
    Value *XZero = isZero(X, F64);
    Bits = select(XZero, SignBits, Bits);
    Valid = B.CreateOr(Valid, XZero);
    return {Bits, B.CreateAnd(Valid, B.CreateAnd(isNormalOrZero(X), isNormal(Y)))};
  }

  Value *div(Value *X, Value *Y) {
    Term TX = unpack(X, F64), TY = unpack(Y, F64);
    Value *Sign = B.CreateXor(TX.Sign, TY.Sign);
    // Both significands with their top bit at 52, the dividend doubled if it
    // is the smaller: the quotient Q = floor(N * 2^54 / Dv) has 55 bits.
    Value *SX = B.CreateSub(c32(53), bitLength(TX.M));
    Value *SY = B.CreateSub(c32(53), bitLength(TY.M));
    // At most 53 places (for a zero significand): no clamp needed.
    Value *N = B.CreateShl(TX.M, B.CreateZExt(SX, I64));
    Value *Dv = B.CreateShl(TY.M, B.CreateZExt(SY, I64));
    Value *Less = B.CreateICmpULT(N, Dv);
    N = select(Less, B.CreateShl(N, 1), N);
    // A reciprocal from below: Y0 ~ 2^82 / Dv to 2^-18, then two Newton
    // steps, Z ~ 2^115 / Dv to 2^-36 and Z2 to 2^-60 (each step squares the
    // error and stays below). Q is then at most 2 above (N * Z2) >> 61, and
    // the exact remainder corrects it. A zero divisor gives a result that is
    // replaced below, but not an out of range conversion.
    Value *DvE = select(B.CreateICmpEQ(Dv, c64(0)), c64(uint64_t(1) << 52), Dv);
    Value *M = quotientSticky(N, DvE);
    Value *E = B.CreateSub(B.CreateSub(TX.E, SX), B.CreateSub(TY.E, SY));
    E = B.CreateSub(E, B.CreateAdd(c32(55), B.CreateZExt(Less, I32)));
    Value *Res = round({Sign, M, E}, F64);
    Value *SignBits = select(Sign, c64(F64.signBit()), c64(0));
    Value *XZero = isZero(X, F64), *YZero = isZero(Y, F64);
    Value *XInf = isInf(X, F64), *YInf = isInf(Y, F64);
    Res = select(B.CreateOr(XZero, YInf), SignBits, Res);
    Res = select(B.CreateOr(XInf, YZero), B.CreateOr(c64(F64.infBits()), SignBits), Res);
    Value *NaN = B.CreateOr(B.CreateOr(isNaN(X, F64), isNaN(Y, F64)),
                            B.CreateOr(B.CreateAnd(XZero, YZero), B.CreateAnd(XInf, YInf)));
    return select(NaN, nanResult({X, Y}, F64, F64), Res);
  }

  /// floor(sqrt(N * 2^56)) for an N with its top bit at 52 or 53 (55 bits),
  /// doubled with a sticky bit for the remainder; NE is N, or any such value
  /// where N is 0. A reciprocal root Y0 ~ 2^57 / sqrt(N) to 2^-18, then two
  /// Newton steps (Y1 ~ 2^89 / sqrt(N) to 2^-35, Y2 to 2^-58) give the root to
  /// 2 units, which the exact remainder corrects.
  Value *rootSticky(Value *N, Value *NE) {
    Value *Y0 = estimate(Intrinsic::agx_rsqrt, B.CreateLShr(NE, 30), 42);
    // The steps' residuals D = 2^114 - NE * Y^2 in units of 2^54, from the
    // high word of (NE << 10) * Y^2 (NE < 2^54): off by less than 2^54, far
    // inside what each step tolerates (2^77 for Y1 to 2^-35, about 2^54 for
    // Y2 to 2^-58, and the remainder below corrects two units either way),
    // with no 128-bit product or difference.
    auto Residual = [&](Value *YSq) {
      Value *P = hi(mulWide(B.CreateShl(NE, 10), YSq), 64);
      return B.CreateSub(c64(uint64_t(1) << 60), P);
    };
    Value *D0 = Residual(B.CreateMul(Y0, Y0)); // Y0 < 2^31
    Value *Y1 = B.CreateAdd(
        B.CreateShl(Y0, 32),
        // (Y0 shifted by 21 as its headroom allows, Y0 < 2^31: the high word
        // is the product shifted by 43, exactly)
        hi(mulWide(B.CreateShl(Y0, 21), B.CreateShl(D0, 14), true), 64,
           true)); // D 2^-40
    Value *D1 = Residual(hi(mulWide(Y1, Y1), 64));
    Value *Y2 = B.CreateAdd(
        Y1, hi(mulWide(Y1, B.CreateShl(D1, 36), true), 97, true)); // D 2^-18
    Value *Root = hi(mulWide(B.CreateShl(NE, 3), Y2), 64); // NE < 2^54
    Value *Rem = B.CreateSub(B.CreateShl(N, 56), B.CreateMul(Root, Root));
    for (int I = 0; I < 2; ++I) {
      Value *Neg = B.CreateICmpSLT(Rem, c64(0));
      Rem = select(Neg, B.CreateAdd(Rem, B.CreateSub(B.CreateShl(Root, 1), c64(1))), Rem);
      Root = B.CreateSub(Root, B.CreateZExt(Neg, I64));
    }
    for (int I = 0; I < 2; ++I) {
      Value *Big = B.CreateICmpSGT(Rem, B.CreateShl(Root, 1));
      Rem = select(Big, B.CreateSub(Rem, B.CreateOr(B.CreateShl(Root, 1), c64(1))), Rem);
      Root = B.CreateAdd(Root, B.CreateZExt(Big, I64));
    }
    Value *Inexact = B.CreateICmpNE(Rem, c64(0));
    return B.CreateOr(B.CreateShl(Root, 1), B.CreateZExt(Inexact, I64));
  }

  /// sqrt(X) rounded to nearest even for a positive normal X (whose root is
  /// normal), and whether X is one.
  std::pair<Value *, Value *> sqrtFast(Value *X) {
    Value *M0 = B.CreateOr(B.CreateAnd(X, c64(F64.fracMask())), c64(F64.fracMask() + 1));
    Value *EX = B.CreateTrunc(B.CreateAnd(B.CreateLShr(X, 52), c64(F64.expAll())), I32);
    // X = M0 2^(EX - 1075): with an odd exponent, M0 one place up
    Value *E0 = B.CreateSub(EX, c32(F64.Bias + 52));
    Value *Odd = B.CreateTrunc(E0, I1);
    Value *N = B.CreateShl(M0, B.CreateZExt(Odd, I64));
    Value *E = B.CreateSub(E0, B.CreateZExt(Odd, I32));
    Value *M = rootSticky(N, N);
    // M 2^(E / 2 - 29), M << 8 as N 2^(E' - 1086)
    Value *EF = B.CreateAdd(B.CreateAShr(E, 1), c32(1086 - 29 - 8));
    auto [Bits, Valid] = roundNormalRNE(B.CreateShl(M, 8), EF);
    // the root of a zero is that zero
    Value *XZero = isZero(X, F64);
    Bits = select(XZero, X, Bits);
    Valid = B.CreateAnd(Valid, B.CreateAnd(isNormal(X), B.CreateICmpSGE(X, c64(0))));
    return {Bits, B.CreateOr(Valid, XZero)};
  }

  Value *sqrt(Value *X) {
    Term TX = unpack(X, F64);
    // The radicand with its top bit at 52 or 53 and an even exponent; the
    // root S = floor(sqrt(N * 2^56)) has 55 bits.
    Value *S = B.CreateSub(c32(53), bitLength(TX.M));
    Value *Odd = B.CreateTrunc(B.CreateSub(TX.E, S), I1);
    S = B.CreateAdd(S, B.CreateZExt(Odd, I32));
    Value *N = B.CreateShl(TX.M, B.CreateZExt(S, I64)); // at most 54 places
    Value *E = B.CreateSub(TX.E, S);
    // A reciprocal root Y0 ~ 2^57 / sqrt(N) to 2^-18, then two Newton steps
    // (Y1 ~ 2^89 / sqrt(N) to 2^-35, Y2 to 2^-58) give S to 2 units, which
    // the exact remainder corrects.
    Value *NE = select(B.CreateICmpEQ(N, c64(0)), c64(uint64_t(1) << 52), N);
    Value *M = rootSticky(N, NE);
    Value *R = round({B.getFalse(), M, B.CreateSub(B.CreateAShr(E, 1), c32(29))}, F64);
    R = select(B.CreateOr(isZero(X, F64), B.CreateICmpEQ(X, c64(F64.infBits()))), X, R);
    Value *NaN = B.CreateOr(isNaN(X, F64),
                            B.CreateAnd(signOf(X, F64), B.CreateNot(isZero(X, F64))));
    return select(NaN, nanResult({X}, F64, F64), R);
  }

  /// A value of format In converted to format Out.
  Value *convert(Value *X, Fmt In, Fmt Out) {
    Value *R = In.P < Out.P && In.Bias < Out.Bias ? widenExact(X, In, Out)
                                                  : round(unpack(X, In), Out);
    Value *Inf = B.CreateOr(c64(Out.infBits()),
                            select(signOf(X, In), c64(Out.signBit()), c64(0)));
    R = select(isInf(X, In), Inf, R);
    return select(isNaN(X, In), nanResult({X}, In, Out), R);
  }

  /// Finite X of format In in the wider format Out, which holds it exactly:
  /// the exponent rebiased and the fraction shifted, a denormal normalized
  /// (its leading one moved to the implicit bit).
  Value *widenExact(Value *X, Fmt In, Fmt Out) {
    Value *Exp = B.CreateAnd(B.CreateLShr(X, In.P - 1), c64(In.expAll()));
    Value *Frac = B.CreateAnd(X, c64(In.fracMask()));
    Value *Denormal = B.CreateICmpEQ(Exp, c64(0));
    Value *Clz = B.CreateBinaryIntrinsic(Intrinsic::ctlz, Frac, B.getFalse());
    Value *Sh = select(Denormal, B.CreateSub(Clz, c64(64 - In.P)), c64(0));
    Value *E = B.CreateAdd(B.CreateSub(Exp, Sh),
                           c64(Out.Bias - In.Bias));
    E = B.CreateAdd(E, B.CreateZExt(Denormal, I64));
    Value *M = B.CreateAnd(B.CreateShl(Frac, Sh), c64(In.fracMask()));
    Value *R = B.CreateOr(B.CreateShl(E, Out.P - 1),
                          B.CreateShl(M, Out.P - In.P));
    R = select(isZero(X, In), c64(0), R);
    return B.CreateOr(R, select(signOf(X, In), c64(Out.signBit()), c64(0)));
  }

  Value *fromInt(Value *V, bool Signed) {
    unsigned N = V->getType()->getIntegerBitWidth();
    Value *Neg = Signed ? B.CreateICmpSLT(V, ConstantInt::get(V->getType(), 0))
                        : B.getFalse();
    Value *Mag = Signed ? select(Neg, B.CreateNeg(V), V) : V;
    Value *M = N < 64 ? B.CreateZExt(Mag, I64) : Mag;
    return round({Neg, M, c32(0)}, F64);
  }

  /// X truncated toward zero to an integer of the type Ty, saturated (NaN
  /// gives 0).
  Value *toInt(Value *X, Type *Ty, bool Signed) {
    unsigned N = Ty->getIntegerBitWidth();
    Term T = widen(unpack(X, F64));
    Value *E = T.E;
    Value *Big = B.CreateICmpSGT(E, c32(int(N)));
    Value *V = select(B.CreateICmpSGE(E, c32(0)),
                      shl(T.M, umin(E, c32(127))),
                      lshr(T.M, B.CreateNeg(B.CreateBinaryIntrinsic(Intrinsic::smin, E, c32(0)))));
    APInt Max = Signed ? APInt::getSignedMaxValue(N) : APInt::getMaxValue(N);
    APInt MinMag = Signed ? APInt::getOneBitSet(N, N - 1) : APInt(N, 0);
    Value *MaxV = ConstantInt::get(I128, Max.zext(128));
    Value *MinMagV = ConstantInt::get(I128, MinMag.zext(128));
    Value *Lim = select(T.Sign, MinMagV, MaxV);
    Value *Mag = select(B.CreateOr(Big, B.CreateICmpUGT(V, Lim)), Lim, V);
    Value *R = B.CreateTrunc(Mag, Ty);
    R = select(T.Sign, B.CreateNeg(R), R);
    if (!Signed)
      R = select(T.Sign, ConstantInt::get(Ty, 0), R);
    return select(isNaN(X, F64), ConstantInt::get(Ty, 0), R);
  }

  enum Integral { Floor, Ceil, Trunc, Nearest, Even, Away };
  /// X rounded to an integral value with masks on its bits (Mode Nearest as
  /// Even: rounded to nearest), and whether X is not a NaN: at or above 2^52
  /// (or infinite) X itself; below 1 a zero or one of its sign; otherwise the
  /// fraction's bits cleared, and one unit of the lowest integral bit added
  /// to the magnitude where Mode rounds it away from zero (a carry into the
  /// exponent is right).
  std::pair<Value *, Value *> integralFast(Value *X, Integral Mode) {
    Value *E = B.CreateTrunc(B.CreateAnd(B.CreateLShr(X, 52), c64(F64.expAll())), I32);
    Value *Neg = B.CreateICmpSLT(X, c64(0));
    Value *Sign = B.CreateAnd(X, c64(F64.signBit()));
    Value *Mag = magnitude(X, F64);
    // 1 <= |X| < 2^52: K = 1075 - E fraction bits (1 to 52)
    Value *EC = umin(B.CreateBinaryIntrinsic(Intrinsic::umax, E, c32(F64.Bias)), c32(F64.Bias + 51));
    Value *K = B.CreateZExt(B.CreateSub(c32(F64.Bias + 52), EC), I64);
    Value *Unit = B.CreateShl(c64(1), K);
    Value *Mask = B.CreateSub(Unit, c64(1));
    Value *Frac = B.CreateAnd(X, Mask);
    Value *T = B.CreateAnd(X, B.CreateNot(Mask));
    Value *HasFrac = B.CreateICmpNE(Frac, c64(0));
    Value *Up;
    Value *One = B.CreateOr(Sign, c64(0x3ff0000000000000ULL));
    // below 1: whether the result is one (of X's sign)
    Value *SmallOne;
    switch (Mode) {
    case Trunc:
      Up = B.getFalse();
      SmallOne = B.getFalse();
      break;
    case Floor:
      Up = B.CreateAnd(Neg, HasFrac);
      SmallOne = B.CreateAnd(Neg, B.CreateICmpNE(Mag, c64(0)));
      break;
    case Ceil:
      Up = B.CreateAnd(B.CreateNot(Neg), HasFrac);
      SmallOne = B.CreateAnd(B.CreateNot(Neg), B.CreateICmpNE(Mag, c64(0)));
      break;
    case Away: {
      Value *Half = B.CreateLShr(Unit, 1);
      Up = B.CreateICmpUGE(Frac, Half);
      SmallOne = B.CreateICmpUGE(Mag, c64(0x3fe0000000000000ULL)); // >= 0.5
      break;
    }
    default: { // to nearest, ties to even
      Value *Half = B.CreateLShr(Unit, 1);
      Value *Odd = B.CreateICmpNE(B.CreateAnd(T, Unit), c64(0));
      Up = B.CreateOr(B.CreateICmpUGT(Frac, Half),
                      B.CreateAnd(B.CreateICmpEQ(Frac, Half), Odd));
      SmallOne = B.CreateICmpUGT(Mag, c64(0x3fe0000000000000ULL)); // > 0.5
      break;
    }
    }
    Value *Mid = B.CreateAdd(T, select(Up, Unit, c64(0)));
    // (for directed modes the small results: a zero of X's sign, or the one
    // of the side rounded to)
    if (Mode == Floor)
      One = c64(0xbff0000000000000ULL);
    else if (Mode == Ceil)
      One = c64(0x3ff0000000000000ULL);
    Value *Small = select(SmallOne, One, Sign);
    Value *R = select(B.CreateICmpUGE(E, c32(F64.Bias + 52)), X,
                      select(B.CreateICmpULT(E, c32(F64.Bias)), Small, Mid));
    return {R, B.CreateNot(isNaN(X, F64))};
  }
  Value *integral(Value *X, Integral Mode) {
    Term T = unpack(X, F64);
    Value *NE = B.CreateNeg(B.CreateBinaryIntrinsic(Intrinsic::smin, T.E, c32(0)));
    Value *Q = lshr(T.M, NE);
    Value *Rem = B.CreateSub(T.M, shl(Q, NE));
    Value *Half = shl(c64(1), B.CreateSub(NE, c32(1)));
    Value *HasFrac = B.CreateICmpNE(Rem, c64(0));
    // Far below 1/2 there is no half to reach (Half is then 0).
    Value *HasHalf = B.CreateAnd(B.CreateICmpNE(NE, c32(0)), B.CreateICmpULE(NE, c32(64)));
    Value *AboveHalf = B.CreateAnd(HasHalf, B.CreateICmpUGT(Rem, Half));
    Value *IsHalf = B.CreateAnd(HasHalf, B.CreateICmpEQ(Rem, Half));
    Value *Odd = B.CreateTrunc(Q, I1);
    Value *IncEven = B.CreateOr(AboveHalf, B.CreateAnd(IsHalf, Odd));
    Value *Inc;
    switch (Mode) {
    case Floor:
      Inc = B.CreateAnd(HasFrac, T.Sign);
      break;
    case Ceil:
      Inc = B.CreateAnd(HasFrac, B.CreateNot(T.Sign));
      break;
    case Trunc:
      Inc = B.getFalse();
      break;
    case Even:
      Inc = IncEven;
      break;
    case Away:
      Inc = B.CreateOr(AboveHalf, IsHalf);
      break;
    case Nearest:
      Inc = select(roundingIs(RoundNearestEven), IncEven,
                   select(roundingIs(RoundUp), B.CreateAnd(HasFrac, B.CreateNot(T.Sign)),
                          select(roundingIs(RoundDown), B.CreateAnd(HasFrac, T.Sign),
                                 B.getFalse())));
      break;
    }
    Q = B.CreateAdd(Q, B.CreateZExt(Inc, I64));
    Value *R = round({T.Sign, Q, c32(0)}, F64);
    // Values of at least 2^52 are integers.
    Value *Integer = B.CreateICmpUGE(magnitude(X, F64), c64(uint64_t(1075) << 52));
    R = select(Integer, X, R);
    return select(isNaN(X, F64), nanResult({X}, F64, F64), R);
  }

  /// A signed integer whose order is that of the non-NaN values, with -0 and
  /// +0 equal (or -0 below +0 with ZeroOrder).
  Value *orderKey(Value *X, bool ZeroOrder) {
    Value *Mag = magnitude(X, F64);
    Value *Neg = ZeroOrder ? B.CreateNot(Mag) : B.CreateNeg(Mag);
    return select(signOf(X, F64), Neg, Mag);
  }

  Value *compare(CmpInst::Predicate P, Value *X, Value *Y) {
    Value *Uno = B.CreateOr(isNaN(X, F64), isNaN(Y, F64));
    Value *KX = orderKey(X, false), *KY = orderKey(Y, false);
    Value *R;
    switch (P) {
    case CmpInst::FCMP_FALSE:
      return B.getFalse();
    case CmpInst::FCMP_TRUE:
      return B.getTrue();
    case CmpInst::FCMP_ORD:
      return B.CreateNot(Uno);
    case CmpInst::FCMP_UNO:
      return Uno;
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_UEQ:
      R = B.CreateICmpEQ(KX, KY);
      break;
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_UNE:
      R = B.CreateICmpNE(KX, KY);
      break;
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_UGT:
      R = B.CreateICmpSGT(KX, KY);
      break;
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_UGE:
      R = B.CreateICmpSGE(KX, KY);
      break;
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_ULT:
      R = B.CreateICmpSLT(KX, KY);
      break;
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ULE:
      R = B.CreateICmpSLE(KX, KY);
      break;
    default:
      llvm_unreachable("not a floating point predicate");
    }
    return CmpInst::isUnordered(P) ? B.CreateOr(Uno, R)
                                   : B.CreateAnd(B.CreateNot(Uno), R);
  }

  /// minnum, maxnum and the IEEE 754-2019 minimum, maximum, minimumNumber
  /// and maximumNumber.
  Value *minMax(Value *X, Value *Y, bool Min, bool PropagateNaN, bool ZeroOrder) {
    Value *KX = orderKey(X, ZeroOrder), *KY = orderKey(Y, ZeroOrder);
    Value *PickX = Min ? B.CreateICmpSLE(KX, KY) : B.CreateICmpSGE(KX, KY);
    Value *R = select(PickX, X, Y);
    Value *XNaN = isNaN(X, F64), *YNaN = isNaN(Y, F64);
    if (PropagateNaN)
      return select(B.CreateOr(XNaN, YNaN), nanResult({X, Y}, F64, F64), R);
    R = select(XNaN, Y, select(YNaN, X, R));
    return select(B.CreateAnd(XNaN, YNaN), nanResult({X, Y}, F64, F64), R);
  }

  Value *isFPClass(Value *X, uint64_t Mask) {
    Value *Sign = signOf(X, F64);
    Value *Mag = magnitude(X, F64);
    Value *R = B.getFalse();
    auto Add = [&](unsigned Test, Value *C) {
      if (Mask & Test)
        R = B.CreateOr(R, C);
    };
    Value *SNaN = B.CreateAnd(isNaN(X, F64),
                              B.CreateICmpEQ(B.CreateAnd(X, c64(F64.quietBit())), c64(0)));
    Value *QNaN = B.CreateAnd(isNaN(X, F64), B.CreateNot(SNaN));
    Value *Inf = isInf(X, F64);
    Value *Zero = isZero(X, F64);
    Value *Sub = B.CreateAnd(B.CreateNot(Zero), B.CreateICmpULT(Mag, c64(F64.fracMask() + 1)));
    Value *Normal = B.CreateAnd(B.CreateICmpUGE(Mag, c64(F64.fracMask() + 1)),
                                B.CreateICmpULT(Mag, c64(F64.infBits())));
    Value *Pos = B.CreateNot(Sign);
    Add(fcSNan, SNaN);
    Add(fcQNan, QNaN);
    Add(fcNegInf, B.CreateAnd(Inf, Sign));
    Add(fcNegNormal, B.CreateAnd(Normal, Sign));
    Add(fcNegSubnormal, B.CreateAnd(Sub, Sign));
    Add(fcNegZero, B.CreateAnd(Zero, Sign));
    Add(fcPosZero, B.CreateAnd(Zero, Pos));
    Add(fcPosSubnormal, B.CreateAnd(Sub, Pos));
    Add(fcPosNormal, B.CreateAnd(Normal, Pos));
    Add(fcPosInf, B.CreateAnd(Inf, Pos));
    return R;
  }
};

} // namespace

static bool isDouble(const Value *V) {
  return V->getType()->isDoubleTy();
}

static Fmt formatOf(Type *Ty) {
  if (Ty->isFloatTy())
    return F32;
  if (Ty->isBFloatTy())
    return BF16;
  return F16;
}

bool AGX::isDoubleArithmetic(const Instruction &I) {
  if (I.getType()->isVectorTy())
    return false;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FNeg:
  case Instruction::FPTrunc:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    return isDouble(&I) || isDouble(I.getOperand(0));
  case Instruction::FCmp:
  case Instruction::FPToSI:
  case Instruction::FPToUI:
  case Instruction::FPExt:
    return isDouble(I.getOperand(0)) || isDouble(&I);
  default:
    break;
  }
  if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
    case Intrinsic::sqrt:
    case Intrinsic::fabs:
    case Intrinsic::copysign:
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
      return isDouble(II);
    case Intrinsic::is_fpclass:
    case Intrinsic::fptosi_sat:
    case Intrinsic::fptoui_sat:
      return isDouble(II->getArgOperand(0));
    default:
      break;
    }
  }
  return false;
}

Value *AGX::buildSoftFMA(IRBuilderBase &B, Value *X, Value *Y, Value *Z,
                         Value *Rounding, bool CanonicalNaN) {
  Type *Ty = X->getType();
  Fmt F = Ty->isDoubleTy() ? F64 : formatOf(Ty);
  Type *IntTy = B.getIntNTy(F.W);
  auto Bits = [&](Value *V) {
    return B.CreateZExt(B.CreateBitCast(V, IntTy), B.getInt64Ty());
  };
  SoftDouble SD(B, Rounding, CanonicalNaN);
  Value *R = SD.fma(Bits(X), Bits(Y), Bits(Z), F);
  return B.CreateBitCast(B.CreateTrunc(R, IntTy), Ty);
}

/// Fast where Valid, and otherwise the general computation (Build, given a
/// SoftDouble at the slow block) in a conditional block before I; B is left
/// at I.
template <typename BuildFn>
static Value *withSlowPath(Instruction &I, IRBuilder<> &B, Value *Fast,
                           Value *Valid, BuildFn Build, Value *Rounding,
                           bool CanonicalNaN) {
  // (unlikely: block placement keeps the general code out of the way of the
  // fast path's instruction fetch)
  Instruction *ThenTerm = SplitBlockAndInsertIfThen(
      B.CreateNot(Valid), I.getIterator(), false,
      MDBuilder(I.getContext()).createUnlikelyBranchWeights());
  IRBuilder<> SB(ThenTerm);
  SoftDouble Slow(SB, Rounding, CanonicalNaN);
  Value *SlowBits = Build(Slow);
  BasicBlock *Then = ThenTerm->getParent();
  BasicBlock *Head = Then->getSinglePredecessor();
  B.SetInsertPoint(I.getParent(), I.getParent()->begin());
  PHINode *Phi = B.CreatePHI(Fast->getType(), 2);
  Phi->addIncoming(Fast, Head);
  Phi->addIncoming(SlowBits, Then);
  B.SetInsertPoint(&I);
  return Phi;
}

/// frem of the doubles of bits X and Y, exact, on integers (before I, which
/// ends up in the join block): with a = mx 2^ex and b = my 2^ey (mx, my of 53
/// bits, denormals normalized), mx 2^(ex - ey) mod my one bit at a time (r =
/// 2 r - my where that is not negative), then normalized at b's exponent (a
/// denormal result shifted back: exact, the remainder a multiple of the
/// smaller of a's and b's last bits). The result has the sign of a: zero
/// where b divides a, a itself where |a| < |b| (also an infinite b or a zero
/// a), and NaN for an infinite or NaN a, a NaN or zero b (the first NaN
/// operand quieted, or the canonical NaN).
static Value *fmodBits(Instruction &I, IRBuilder<> &B, Value *X, Value *Y,
                       bool CanonicalNaN) {
  LLVMContext &Ctx = I.getContext();
  Type *I32 = B.getInt32Ty(), *I64 = B.getInt64Ty();
  auto K = [&](uint64_t V) { return ConstantInt::get(I64, V); };
  const uint64_t Sign = 0x8000000000000000ull, Abs = ~Sign,
                 Inf = 0x7ff0000000000000ull, Frac = (1ull << 52) - 1,
                 One = 1ull << 52, Quiet = 1ull << 51;
  Value *SX = B.CreateAnd(X, K(Sign));
  Value *AX = B.CreateAnd(X, K(Abs)), *AY = B.CreateAnd(Y, K(Abs));
  Value *XNaN = B.CreateICmpUGT(AX, K(Inf)), *YNaN = B.CreateICmpUGT(AY, K(Inf));
  Value *NaN = B.CreateOr(B.CreateOr(B.CreateICmpUGE(AX, K(Inf)), YNaN),
                          B.CreateICmpEQ(AY, K(0)));
  Value *NaNR = K(CanonicalNaN ? 0x7fffffffffffffffull : 0x7ff8000000000000ull);
  if (!CanonicalNaN)
    NaNR = B.CreateSelect(XNaN, B.CreateOr(X, K(Quiet)),
                          B.CreateSelect(YNaN, B.CreateOr(Y, K(Quiet)), NaNR));
  Value *Small = B.CreateICmpULT(AX, AY);
  Value *Special = B.CreateOr(B.CreateOr(NaN, Small), B.CreateICmpEQ(AX, AY));
  Value *SpecialR =
      B.CreateSelect(NaN, NaNR, B.CreateSelect(Small, X, SX));
  // The significand of 53 bits and its exponent (a denormal's 1 - s, s the
  // shift normalizing it); only used for finite nonzero operands.
  auto Split = [&](Value *V) -> std::pair<Value *, Value *> {
    Value *E = B.CreateTrunc(B.CreateLShr(V, 52), I32);
    Value *Denormal = B.CreateICmpEQ(E, B.getInt32(0));
    Value *S = B.CreateSub(
        B.CreateTrunc(B.CreateBinaryIntrinsic(Intrinsic::ctlz, V, B.getTrue()),
                      I32),
        B.getInt32(11));
    Value *M = B.CreateSelect(Denormal, B.CreateShl(V, B.CreateZExt(S, I64)),
                              B.CreateOr(B.CreateAnd(V, K(Frac)), K(One)));
    return {M, B.CreateSelect(Denormal, B.CreateSub(B.getInt32(1), S), E)};
  };
  auto [MX, EX] = Split(AX);
  auto [MY, EY] = Split(AY);
  Value *N0 = B.CreateSub(EX, EY);
  Instruction *ThenTerm =
      SplitBlockAndInsertIfThen(B.CreateNot(Special), I.getIterator(), false);
  BasicBlock *Head = ThenTerm->getParent()->getSinglePredecessor();
  BasicBlock *Pre = ThenTerm->getParent();
  BasicBlock *Join = I.getParent();
  Function *Fn = Join->getParent();
  BasicBlock *Loop = BasicBlock::Create(Ctx, "fmod.loop", Fn, Join);
  BasicBlock *Tail = BasicBlock::Create(Ctx, "fmod.tail", Fn, Join);
  ThenTerm->eraseFromParent();
  IRBuilder<>(Pre).CreateBr(Loop);

  IRBuilder<> LB(Loop);
  PHINode *N = LB.CreatePHI(I32, 2);
  PHINode *M = LB.CreatePHI(I64, 2);
  N->addIncoming(N0, Pre);
  M->addIncoming(MX, Pre);
  Value *D = LB.CreateSub(M, MY);
  Value *R = LB.CreateSelect(LB.CreateICmpSGE(D, K(0)), D, M);
  N->addIncoming(LB.CreateSub(N, LB.getInt32(1)), Loop);
  M->addIncoming(LB.CreateShl(R, K(1)), Loop);
  LB.CreateCondBr(LB.CreateICmpEQ(N, LB.getInt32(0)), Tail, Loop);

  // r below 2^53 at b's exponent: normalized (r = 0 is a signed zero).
  IRBuilder<> TB(Tail);
  Value *Z = TB.CreateTrunc(
      TB.CreateBinaryIntrinsic(Intrinsic::ctlz, R, TB.getFalse()), I32);
  Value *S = TB.CreateSub(Z, TB.getInt32(11));
  Value *E = TB.CreateSub(EY, S);
  Value *RN = TB.CreateShl(R, TB.CreateZExt(S, I64));
  Value *Normal = TB.CreateOr(TB.CreateAnd(RN, K(Frac)),
                              TB.CreateShl(TB.CreateZExt(E, I64), K(52)));
  Value *Denormal = TB.CreateLShr(
      RN, TB.CreateZExt(TB.CreateSub(TB.getInt32(1), E), I64));
  Value *Bits = TB.CreateSelect(TB.CreateICmpSGT(E, TB.getInt32(0)), Normal,
                                Denormal);
  Bits = TB.CreateSelect(TB.CreateICmpEQ(R, K(0)), K(0), Bits);
  Value *LoopR = TB.CreateOr(Bits, SX);
  TB.CreateBr(Join);

  B.SetInsertPoint(Join->begin());
  PHINode *Result = B.CreatePHI(I64, 2);
  Result->addIncoming(SpecialR, Head);
  Result->addIncoming(LoopR, Tail);
  B.SetInsertPoint(&I);
  return Result;
}

Value *AGX::lowerDoubleArithmetic(Instruction &I, Value *Rounding,
                                  bool CanonicalNaN, bool FastPathsHere) {
  if (!isDoubleArithmetic(I))
    return nullptr;
  IRBuilder<> B(&I);
  SoftDouble SD(B, Rounding, CanonicalNaN);
  Type *I64 = B.getInt64Ty();
  auto Bits = [&](unsigned N) {
    Value *V = I.getOperand(N);
    if (V->getType()->isDoubleTy())
      return B.CreateBitCast(V, I64);
    unsigned W = V->getType()->getPrimitiveSizeInBits();
    return B.CreateZExt(B.CreateBitCast(V, B.getIntNTy(W)), I64);
  };
  auto Dbl = [&](Value *V) { return B.CreateBitCast(V, B.getDoubleTy()); };
  // conversions rounded to nearest (or truncating) with short paths
  auto *RoundC = dyn_cast<ConstantInt>(Rounding);
  bool Fast = AGXSoftF64FastPaths && FastPathsHere && RoundC &&
              RoundC->getZExtValue() == RoundNearestEven;
  Value *R = nullptr;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul: {
    bool Mul = I.getOpcode() == Instruction::FMul;
    bool Sub = I.getOpcode() == Instruction::FSub;
    Value *X = Bits(0), *Y = Bits(1);
    auto *RC = dyn_cast<ConstantInt>(Rounding);
    if (!AGXSoftF64FastPaths || !FastPathsHere || !RC || RC->getZExtValue() != RoundNearestEven) {
      R = Dbl(Mul ? SD.mul(X, Y) : SD.add(X, Y, Sub));
      break;
    }
    // Normal operands with a normal result take a short path; the others
    // (zeros, denormals, infinities, NaNs, underflow, overflow) the general
    // one, in a conditional block.
    auto [Fast, Valid] =
        Mul ? SD.mulFast(X, Y)
            : SD.addFast(X, Sub ? B.CreateXor(Y, SD.c64(F64.signBit())) : Y);
    R = Dbl(withSlowPath(I, B, Fast, Valid, [&](SoftDouble &Slow) {
      return Mul ? Slow.mul(X, Y) : Slow.add(X, Y, Sub);
    }, Rounding, CanonicalNaN));
    break;
  }
  case Instruction::FDiv: {
    Value *X = Bits(0), *Y = Bits(1);
    auto *RC = dyn_cast<ConstantInt>(Rounding);
    if (!AGXSoftF64FastPaths || !FastPathsHere || !RC || RC->getZExtValue() != RoundNearestEven) {
      R = Dbl(SD.div(X, Y));
      break;
    }
    auto [Fast, Valid] = SD.divFast(X, Y);
    R = Dbl(withSlowPath(I, B, Fast, Valid, [&](SoftDouble &Slow) {
      return Slow.div(X, Y);
    }, Rounding, CanonicalNaN));
    break;
  }
  case Instruction::FRem:
    R = Dbl(fmodBits(I, B, Bits(0), Bits(1), CanonicalNaN));
    break;
  case Instruction::FNeg:
    R = Dbl(B.CreateXor(Bits(0), SD.c64(F64.signBit())));
    break;
  case Instruction::FPExt: {
    Fmt In = formatOf(I.getOperand(0)->getType());
    Value *X = Bits(0);
    if (Fast && In.P == F32.P && In.Bias == F32.Bias) {
      auto [FastBits, Valid] = SD.extFast(X);
      R = Dbl(withSlowPath(I, B, FastBits, Valid, [&](SoftDouble &Slow) {
        return Slow.convert(X, In, F64);
      }, Rounding, CanonicalNaN));
      break;
    }
    R = Dbl(SD.convert(X, In, F64));
    break;
  }
  case Instruction::FPTrunc: {
    Type *Ty = I.getType();
    Fmt Out = formatOf(Ty);
    Value *X = Bits(0);
    if (Fast && Ty->isFloatTy()) {
      auto [FastBits, Valid] = SD.truncFast(X);
      Value *V = withSlowPath(I, B, FastBits, Valid, [&](SoftDouble &Slow) {
        return Slow.convert(X, F64, Out);
      }, Rounding, CanonicalNaN);
      R = B.CreateBitCast(B.CreateTrunc(V, B.getIntNTy(Out.W)), Ty);
      break;
    }
    Value *V = SD.convert(X, F64, Out);
    R = B.CreateBitCast(B.CreateTrunc(V, B.getIntNTy(Out.W)), Ty);
    break;
  }
  case Instruction::SIToFP:
  case Instruction::UIToFP: {
    if (!isDouble(&I))
      return nullptr;
    bool Signed = I.getOpcode() == Instruction::SIToFP;
    Value *V = I.getOperand(0);
    if (Fast && V->getType()->getIntegerBitWidth() <= 64) {
      auto [FastBits, Valid] = SD.fromIntFast(V, Signed);
      // 32-bit integers are always exact
      if (V->getType()->getIntegerBitWidth() <= 32)
        R = Dbl(FastBits);
      else
        R = Dbl(withSlowPath(I, B, FastBits, Valid, [&](SoftDouble &Slow) {
          return Slow.fromInt(V, Signed);
        }, Rounding, CanonicalNaN));
      break;
    }
    R = Dbl(SD.fromInt(V, Signed));
    break;
  }
  case Instruction::FPToSI:
  case Instruction::FPToUI: {
    bool Signed = I.getOpcode() == Instruction::FPToSI;
    Value *X = Bits(0);
    if (Fast && I.getType()->isIntegerTy() && I.getType()->getIntegerBitWidth() <= 64) {
      auto [FastV, Valid] = SD.toIntFast(X, I.getType(), Signed);
      R = withSlowPath(I, B, FastV, Valid, [&](SoftDouble &Slow) {
        return Slow.toInt(X, I.getType(), Signed);
      }, Rounding, CanonicalNaN);
      break;
    }
    R = SD.toInt(X, I.getType(), Signed);
    break;
  }
  case Instruction::FCmp:
    R = SD.compare(cast<FCmpInst>(I).getPredicate(), Bits(0), Bits(1));
    break;
  default: {
    auto &II = cast<IntrinsicInst>(I);
    switch (II.getIntrinsicID()) {
    case Intrinsic::fma:
    case Intrinsic::fmuladd: {
      Value *X = Bits(0), *Y = Bits(1), *Z = Bits(2);
      auto *RC = dyn_cast<ConstantInt>(Rounding);
      if (!AGXSoftF64FastPaths || !FastPathsHere || !RC || RC->getZExtValue() != RoundNearestEven) {
        R = Dbl(SD.fma(X, Y, Z));
        break;
      }
      auto [Fast, Valid] = SD.fmaFast(X, Y, Z);
      R = Dbl(withSlowPath(I, B, Fast, Valid, [&](SoftDouble &Slow) {
        return Slow.fma(X, Y, Z);
      }, Rounding, CanonicalNaN));
      break;
    }
    case Intrinsic::sqrt: {
      Value *X = Bits(0);
      auto *RC = dyn_cast<ConstantInt>(Rounding);
      if (!AGXSoftF64FastPaths || !FastPathsHere || !RC || RC->getZExtValue() != RoundNearestEven) {
        R = Dbl(SD.sqrt(X));
        break;
      }
      auto [Fast, Valid] = SD.sqrtFast(X);
      R = Dbl(withSlowPath(I, B, Fast, Valid, [&](SoftDouble &Slow) {
        return Slow.sqrt(X);
      }, Rounding, CanonicalNaN));
      break;
    }
    case Intrinsic::fabs:
      R = Dbl(SD.magnitude(Bits(0), F64));
      break;
    case Intrinsic::copysign:
      R = Dbl(B.CreateOr(SD.magnitude(Bits(0), F64),
                         B.CreateAnd(Bits(1), SD.c64(F64.signBit()))));
      break;
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
      R = Dbl(SD.minMax(Bits(0), Bits(1), II.getIntrinsicID() == Intrinsic::minnum,
                        false, true));
      break;
    case Intrinsic::minimum:
    case Intrinsic::maximum:
      R = Dbl(SD.minMax(Bits(0), Bits(1), II.getIntrinsicID() == Intrinsic::minimum,
                        true, true));
      break;
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum:
      R = Dbl(SD.minMax(Bits(0), Bits(1),
                        II.getIntrinsicID() == Intrinsic::minimumnum, false, true));
      break;
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::trunc:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::roundeven:
    case Intrinsic::round: {
      SoftDouble::Integral Mode =
          StringSwitch<SoftDouble::Integral>(Intrinsic::getBaseName(II.getIntrinsicID()))
              .Case("llvm.floor", SoftDouble::Floor)
              .Case("llvm.ceil", SoftDouble::Ceil)
              .Case("llvm.trunc", SoftDouble::Trunc)
              .Case("llvm.roundeven", SoftDouble::Even)
              .Case("llvm.round", SoftDouble::Away)
              .Default(SoftDouble::Nearest);
      Value *X = Bits(0);
      // rint / nearbyint round in the current direction: nearest here only
      bool Directional = Mode == SoftDouble::Nearest;
      if (AGXSoftF64FastPaths && FastPathsHere && (!Directional || Fast)) {
        auto [FastBits, Valid] = SD.integralFast(X, Mode);
        R = Dbl(withSlowPath(I, B, FastBits, Valid, [&](SoftDouble &Slow) {
          return Slow.integral(X, Mode);
        }, Rounding, CanonicalNaN));
        break;
      }
      R = Dbl(SD.integral(X, Mode));
      break;
    }
    case Intrinsic::canonicalize: {
      Value *X = Bits(0);
      R = Dbl(SD.select(SD.isNaN(X, F64), SD.nanResult({X}, F64, F64), X));
      break;
    }
    case Intrinsic::is_fpclass:
      R = SD.isFPClass(Bits(0),
                       cast<ConstantInt>(II.getArgOperand(1))->getZExtValue());
      break;
    case Intrinsic::fptosi_sat:
    case Intrinsic::fptoui_sat: {
      bool Signed = II.getIntrinsicID() == Intrinsic::fptosi_sat;
      Value *X = Bits(0);
      if (Fast && I.getType()->isIntegerTy() && I.getType()->getIntegerBitWidth() <= 64) {
        auto [FastV, Valid] = SD.toIntFast(X, I.getType(), Signed);
        R = withSlowPath(I, B, FastV, Valid, [&](SoftDouble &Slow) {
          return Slow.toInt(X, I.getType(), Signed);
        }, Rounding, CanonicalNaN);
        break;
      }
      R = SD.toInt(X, I.getType(), Signed);
      break;
    }
    default:
      return nullptr;
    }
  }
  }
  R->takeName(&I);
  I.replaceAllUsesWith(R);
  I.eraseFromParent();
  return R;
}

namespace {

class AGXSoftF64 : public FunctionPass {
public:
  static char ID;
  AGXSoftF64() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Soft F64"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char AGXSoftF64::ID = 0;

INITIALIZE_PASS_BEGIN(AGXSoftF64, DEBUG_TYPE, "AGX Soft F64", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXSoftF64, DEBUG_TYPE, "AGX Soft F64", false, false)

FunctionPass *llvm::createAGXSoftF64Pass() { return new AGXSoftF64(); }

bool AGXSoftF64::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  SmallVector<Instruction *, 16> Worklist;
  for (Instruction &I : instructions(F))
    if (AGX::isDoubleArithmetic(I))
      Worklist.push_back(&I);
  bool Changed = false;
  Value *Nearest = ConstantInt::get(Type::getInt32Ty(F.getContext()),
                                    RoundNearestEven);
  bool FastPaths = Worklist.size() <= AGXSoftF64FastPathsMax;
  for (Instruction *I : Worklist)
    Changed |= AGX::lowerDoubleArithmetic(*I, Nearest, ST.hasCanonicalNaN(),
                                          FastPaths) != nullptr;
  return Changed;
}
