//===-- AGXMathExpansion.cpp - Math functions without hardware ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The math functions AGX has no instruction for, built from the ones it has
// (sin, cos, exp2, log2, sqrt, division) in float: tangent, the inverse and
// hyperbolic functions and their inverses, and sinpi / cospi / tanpi. Half
// and bfloat values are computed in float. The reductions and polynomials
// are those of the Cephes single precision library (atanf, asinf, acosf,
// sinhf, coshf, tanhf, asinhf, acoshf, atanhf).
//
// Also the precise float sin, cos, exp, exp2, exp10, log, log2, log10 and
// pow, where the hardware's approximations (which the afn forms keep) are
// tens of units in the last place off: the argument reduced exactly (a
// three-part pi / 2, the exponent split from a log or exp argument), Cephes'
// sinf, cosf, expf and logf polynomials, and for pow a log and exp carried
// with a low-order part.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include <cmath>

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<bool> ExpMagicRound(
    "agx-exp-magic-round", cl::init(true), cl::Hidden,
    cl::desc("exp's n = round(x / ln 2) by adding and subtracting 1.5 2^23, "
             "its integer the sum's low bits (no rint or convert)"));

static cl::opt<bool> LogAbsCore(
    "agx-log-abs-core", cl::init(true), cl::Hidden,
    cl::desc("log, log2 and log10's polynomial of |x| (what pow's is of): "
             "the same value for every x it is not replaced for"));

static cl::opt<bool> LogExponentByAdd(
    "agx-log-exponent-by-add", cl::init(true), cl::Hidden,
    cl::desc("log's exponent e as the float 2^23 + e (e under 2^23's "
             "bits) less 2^23 + 127: one add, no subtract and convert"));

static cl::opt<bool> LogOffsetSplit(
    "agx-log-offset-split", cl::init(true), cl::Hidden,
    cl::desc("log's 2^e (1 + f) with 1 + f in (sqrt(1/2), sqrt(2)] from x's "
             "bits plus 1 - sqrt(1/2)'s: no compare and selects"));

static cl::opt<bool> LogFastTwoSum(
    "agx-log-fast-two-sum", cl::init(true), cl::Hidden,
    cl::desc("log's sums with their errors as Fast2Sum (the first addend "
             "the larger): the same exact error in three operations, not "
             "six"));

static cl::opt<bool> ExpNaNThroughScale(
    "agx-exp-nan-through-scale", cl::init(true), cl::Hidden,
    cl::desc("exp without the +inf select: no NaN select, the NaN's sum "
             "bits adding nothing to the scale"));

static cl::opt<bool> ExpOverflowByArg(
    "agx-exp-overflow-by-arg", cl::init(true), cl::Hidden,
    cl::desc("exp's +inf select by its argument's compare, and no NaN "
             "select"));

static cl::opt<bool> ExpUnclamped(
    "agx-exp-unclamped", cl::init(true), cl::Hidden,
    cl::desc("exp's +0 select by its argument's compare too, and no clamps "
             "of the argument"));

static cl::opt<bool> Exp2BoundsByArg(
    "agx-exp2-bounds-by-arg",
    cl::desc("2^x and 10^x: +inf and +0 selected by compares of x, no "
             "clamps of x and no NaN select"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> HalfExpOverflowByArg(
    "agx-half-exp-overflow-by-arg",
    cl::desc("sinh / cosh's e^a / 2: +inf by a compare of a, no clamp and "
             "no NaN select"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> ExpFlushedUnderflow(
    "agx-exp-flushed-underflow", cl::init(true), cl::Hidden,
    cl::desc("exp below a maximum whose users flush denormal operands: "
             "clamped at its +0 bound, no +0 select"));

static cl::opt<bool> PowConstantEnds(
    "agx-pow-constant-ends", cl::init(true), cl::Hidden,
    cl::desc("pow(x, y) of a finite nonzero constant y: |x| = 0 and inf by "
             "two selects of the results y's sign gives"));

static cl::opt<bool> FusedExactScales(
    "agx-math-fused-exact-scales", cl::init(true), cl::Hidden,
    cl::desc("c - 2 y and the like as one fma where the scaling is exact: "
             "tanh's 1 - 2 y, asin's pi / 2 - 2 p, asin's and acos's "
             "(1 - a) / 2, x - 2 round(x / 2)"));

static cl::opt<bool> TanhDoubledClamp(
    "agx-tanh-doubled-clamp", cl::init(true), cl::Hidden,
    cl::desc("tanh's 2 min(|x|, 9) as min(|x| + |x|, 18)"));

static cl::opt<bool> SinCosMagicRound(
    "agx-sincos-magic-round", cl::init(true), cl::Hidden,
    cl::desc("sin / cos's j = round(2 x / pi) likewise, the quadrant the "
             "sum's low bits"));

static cl::opt<bool> SinCosSharedOdd(
    "agx-sincos-shared-odd", cl::init(true), cl::Hidden,
    cl::desc("cos's polynomial select by sin's odd quadrant, its operands "
             "swapped, not by (q + 1) & 1"));

static cl::opt<bool> ExpCoreProduct(
    "agx-exp-core-product", cl::init(true), cl::Hidden,
    cl::desc("expf's r + r^2 p as fma(p r, r, r), two registers, not "
             "fma(p, r r, r), three"));

static cl::opt<bool> CosHorner(
    "agx-cos-horner", cl::init(true), cl::Hidden,
    cl::desc("cosf's 1 - z / 2 + z^2 p as fma(z, fma(z, p, -1/2), 1)"));

static cl::opt<bool> ExpFusedRound(
    "agx-exp-fused-round", cl::init(true), cl::Hidden,
    cl::desc("expf's magic-number round of x log2(e) as one fma"));

static cl::opt<bool> SinCosFusedRound(
    "agx-sincos-fused-round", cl::init(true), cl::Hidden,
    cl::desc("sin / cos's magic-number round of x 2 / pi as one fma"));

static cl::opt<bool> LogLowFma(
    "agx-log-low-fma", cl::init(true), cl::Hidden,
    cl::desc("With the constants in uniforms, logf's low part (errors) + e "
             "ln2_lo as one fma"));

static cl::opt<bool> ExpLowSelect(
    "agx-exp-low-select", cl::init(true), cl::Hidden,
    cl::desc("e^(hi + lo) (pow): lo as 0 where |hi| >= 89, not the result's "
             "+inf / +0 selects"));

static cl::opt<bool> ExpPoly5(
    "agx-exp-poly5", cl::init(true), cl::Hidden,
    cl::desc("expf's e^r - 1 - r as r^2 times a minimax polynomial of "
             "degree 4, not Cephes' of degree 5"));

static cl::opt<bool> PowConstantTail(
    "agx-pow-constant-tail", cl::init(true), cl::Hidden,
    cl::desc("pow by a finite nonzero constant: no select of x = 1 and one "
             "NaN select for a NaN or a negative finite x"));

static cl::opt<bool> TanQuotientSigns(
    "agx-tan-quotient-signs", cl::init(true), cl::Hidden,
    cl::desc("tan as sin / cos's polynomials by quadrant, the quotient's sign "
             "the odd quadrant's, without the two halves' signs"));

static cl::opt<bool> NegatedClamp(
    "agx-exp-negated-clamp",
    cl::desc("e^-y clamps y and negates the clamp"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> ExpMaxDifference(
    "agx-exp-max-difference",
    cl::desc("e^(a - max(a, b)) without the overflow clamp and select"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> IntegerLdexp(
    "agx-exp-integer-scale",
    cl::desc("the exponentials' 2^n scale as an exponent addition where f32 "
             "flushes"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> ExpNonNegative(
    "agx-exp-non-negative",
    cl::desc("tanh's, sinh's and cosh's exponentials of a non-negative "
             "argument without the underflow clamp and selects (and tanh's, "
             "of at most 18, without the overflow select)"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> HalfFastHyperbolic(
    "agx-half-fast-hyperbolic",
    cl::desc("half tanh, sinh and cosh with the hardware's exponential"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> HalfFastDivision(
    "agx-half-fast-division",
    cl::desc("half math functions' divisions by the hardware's reciprocal"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> NarrowFastSqrt(
    "agx-narrow-fast-sqrt",
    cl::desc("half and bfloat math functions' square roots by the "
             "hardware's approximation"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> Exp2DirectReduction(
    "agx-exp2-direct-reduction",
    cl::desc("precise exp2 reduced by n = roundeven(x) directly"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> AbsNegatedClamp(
    "agx-exp-abs-negated-clamp", cl::init(true), cl::Hidden,
    cl::desc("exp(-|y|) below a maximum: |y| clamped by the mirrored bound, "
             "the select of y itself"));

static cl::opt<bool> Exp2BelowMax(
    "agx-exp2-below-max",
    cl::desc("2^x and 10^x of x at most 0 (below a maximum): no upper clamp"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> AtanQuotientStep(
    "agx-atan-quotient-step", cl::init(true), cl::Hidden,
    cl::desc("atan's -1 / a and (a - 1) / (a + 1) by the reciprocal step and "
             "a Markstein correction"));

static cl::opt<bool> NarrowSinCosReduction(
    "agx-narrow-sincos-reduction",
    cl::desc("bfloat sin / cos: pi / 2 in two parts"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> BFloatShortPolys(
    "agx-bfloat-short-polys",
    cl::desc("bfloat tanh, sinh, cosh, asin, acos (and AIR's asinh, acosh, "
             "atanh, tanpi) with their polynomials' highest terms left out"),
    cl::init(true), cl::Hidden);

namespace {
/// The argument whose compares give the +inf / +0 results (exp's, or
/// 2^x's / 10^x's: L15-E485): +inf from OverAt, +0 at or below ZeroAt.
struct ArgBounds {
  Value *Arg = nullptr;
  double OverAt = 88.72283935546875;
  double ZeroAt = -87.3365478515625;
  // +0 by the compare too (no clamp below)
  bool Zero = false;
  // Arg is y of x = -y: the compares mirrored (a NaN fails them too).
  bool Negated = false;
};

class MathBuilder {
  IRBuilderBase &B;
  Type *Ty; // float or a vector of float
  // No float operand is a denormal the expansion must keep: f32 arithmetic
  // flushes (no ieee-fp32), or the operands are widened halves.
  bool NoDenormals;
  // The result is rounded to half (L15-E329).
  bool HalfResult;
  // The result is rounded to half or bfloat (L15-E329, E331).
  bool NarrowResult;
  // Divisions correctly rounded even for a half result (atan2: with the
  // reciprocal's, results near a tie between two halves were rounded the
  // other way, on the functional model).
  bool PreciseDivision = false;
  // Leading polynomial terms left out (a bfloat result, where every bfloat
  // input's result was checked to be the same).
  unsigned PolyDrop = 0;
  bool bfloatResult() const { return NarrowResult && !HalfResult; }

public:
  // f32 arithmetic's NaN is 0x7fc00000 (no canonical-nan feature).
  bool NativeNaN = false;
  // exp's result only reaches arithmetic that flushes denormal operands.
  bool ExpResultFlushed = false;

  MathBuilder(IRBuilderBase &B, Type *Ty, bool NoDenormals = false,
              bool HalfResult = false, bool NarrowResult = false)
      : B(B), Ty(Ty), NoDenormals(NoDenormals), HalfResult(HalfResult),
        NarrowResult(NarrowResult) {}

  /// The hardware's approximation of \p ID (afn's).
  Value *hw(Intrinsic::ID ID, Value *X) {
    Value *V = B.CreateIntrinsic(ID, {Ty}, {X});
    cast<Instruction>(V)->setHasApproxFunc(true);
    return V;
  }

  Value *k(double V) { return ConstantFP::get(Ty, V); }
  Value *call(Intrinsic::ID ID, ArrayRef<Value *> Args) {
    return B.CreateIntrinsic(ID, {Ty}, Args);
  }
  Value *fma(Value *A, Value *X, Value *C) {
    return call(Intrinsic::fma, {A, X, C});
  }
  /// The polynomial with coefficients \p C, highest degree first, at \p Z.
  Value *poly(Value *Z, ArrayRef<double> C) {
    // (a bfloat result: the highest-degree terms PolyDrop says, at most down
    // to two, L15-E351)
    if (PolyDrop && BFloatShortPolys)
      C = C.drop_front(std::min<size_t>(PolyDrop, C.size() - 2));
    Value *P = k(C[0]);
    for (double V : C.drop_front())
      P = fma(P, Z, k(V));
    return P;
  }
  Value *abs(Value *X) { return call(Intrinsic::fabs, {X}); }
  Value *copysign(Value *Mag, Value *Sign) {
    return call(Intrinsic::copysign, {Mag, Sign});
  }
  /// \p Approx: the hardware's square root for a half or bfloat result,
  /// where every such result was checked to be the same (L15-E333).
  Value *sqrt(Value *X, bool Approx = false) {
    if (Approx && NarrowResult && NarrowFastSqrt)
      return hw(Intrinsic::sqrt, X);
    return call(Intrinsic::sqrt, {X});
  }
  Value *exp(Value *X) { return call(Intrinsic::exp, {X}); }
  Value *expPrecise(Value *X, bool NoOverflow = false) {
    return expHiLo(X, nullptr, /*Bounded=*/false,
                   (ExpMaxDifference && NoOverflow) || noExpOverflow(X));
  }

  /// Whether 2^x / 10^x's x is known below a maximum (the caller's analysis
  /// or noExpOverflow's form).
  static bool belowMax(Value *X, bool NoOverflow) {
    return Exp2BelowMax && ExpMaxDifference &&
           (NoOverflow || noExpOverflow(X));
  }

  /// x = a - max(a, b) (maxnum, maximumnum or maximum, either order, or a
  /// SIMD-group's fmax of a: a softmax's exponent): at most 0, or a denormal where the maximum's
  /// comparison flushes, or a NaN; e^x does not overflow, so the clamp above
  /// and the +inf select go (L15-E270).
  static bool noExpOverflow(Value *X) {
    if (!ExpMaxDifference || !X->getType()->isFloatTy())
      return false;
    Value *A, *M;
    if (!match(X, m_FSub(m_Value(A), m_Value(M))))
      return false;
    auto *II = dyn_cast<IntrinsicInst>(M);
    if (!II)
      return false;
    // A SIMD-group's (or quad's) fmax of a itself: at least this lane's a.
    if ((II->getIntrinsicID() == Intrinsic::agx_simd_reduce ||
         II->getIntrinsicID() == Intrinsic::agx_quad_reduce) &&
        match(II->getArgOperand(1), m_SpecificInt(AGX::SIMD_FMAX)))
      return II->getArgOperand(0) == A;
    switch (II->getIntrinsicID()) {
    case Intrinsic::maxnum:
    case Intrinsic::maximumnum:
    case Intrinsic::maximum:
      return II->getArgOperand(0) == A || II->getArgOperand(1) == A;
    default:
      return false;
    }
  }
  Value *log(Value *X) { return call(Intrinsic::log, {X}); }
  Value *sel(Value *C, Value *T, Value *F) { return B.CreateSelect(C, T, F); }
  Value *gt(Value *A, double V) { return B.CreateFCmpOGT(A, k(V)); }
  Value *lt(Value *A, double V) { return B.CreateFCmpOLT(A, k(V)); }
  Value *mul(Value *A, Value *X) { return B.CreateFMul(A, X); }
  Value *add(Value *A, Value *X) { return B.CreateFAdd(A, X); }
  Value *sub(Value *A, Value *X) { return B.CreateFSub(A, X); }
  Value *div(Value *A, Value *X) {
    // A half or bfloat result: a times the hardware's reciprocal (L15-E329,
    // E331; scalars: the reciprocal intrinsic does not split)
    if (NarrowResult && HalfFastDivision && !PreciseDivision &&
        !Ty->isVectorTy())
      return mul(A, B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {X}));
    return B.CreateFDiv(A, X);
  }

  /// n / d where the result is used only for d at least 1 (+inf, a NaN)
  /// and n zero, -1 or at least 2^-24 in magnitude: the correctly rounded
  /// reciprocal y of d by E282's step (rcp, two fmas, +0 past 2^126) and
  /// Markstein's q0 = n y, q = q0 + y (n - d q0) (E409), without the
  /// division's exponent scaling and special cases (atan's -1 / a and
  /// (a - 1) / (a + 1): every float input's atan the same, L15-E501). The
  /// division otherwise (a half or bfloat result's reciprocal product).
  Value *quotientAboveOne(Value *N, Value *D) {
    if ((NarrowResult && HalfFastDivision && !PreciseDivision &&
         !Ty->isVectorTy()) ||
        !AtanQuotientStep || !NoDenormals)
      return div(N, D);
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {D});
    Value *Y = fma(fma(B.CreateFNeg(D), Y0, k(1.0)), Y0, Y0);
    // (past 2^126, +inf included, the quotient flushes: q0 = n * +0, whose
    // correction's d q0 would be inf * 0 for an infinite d)
    Value *Huge = gt(D, 0x1p126);
    Y = sel(Huge, k(0.0), Y);
    Value *Q0 = mul(N, Y);
    return sel(Huge, Q0, fma(fma(B.CreateFNeg(D), Q0, N), Y, Q0));
  }

  /// atan of a value in [0, inf] or NaN: the argument reduced to
  /// [0, tan(pi / 8)] around 0, pi / 4 or pi / 2.
  Value *atanNonNegative(Value *A, bool AtMostOne = false) {
    Value *Mid = gt(A, 0.4142135623730950);
    Value *Big = AtMostOne ? ConstantInt::getFalse(Mid->getType())
                           : gt(A, 2.414213562373095);
    // One division, of the operands the range picks (-1 / a or
    // (a - 1) / (a + 1)): a select of two divisions computed both ways.
    Value *Nm = sel(Big, k(-1.0), sub(A, k(1.0)));
    Value *Dn = sel(Big, A, add(A, k(1.0)));
    Value *Q = quotientAboveOne(Nm, Dn);
    Value *X = sel(B.CreateOr(Big, Mid), Q, A);
    Value *Y0 = sel(Big, k(M_PI_2), sel(Mid, k(M_PI_4), k(0.0)));
    Value *Z = mul(X, X);
    Value *P = poly(Z, {8.05374449538e-2, -1.38776856032e-1, 1.99777106478e-1,
                        -3.33329491539e-1});
    return add(Y0, fma(mul(P, Z), X, X));
  }

  /// asin of a value in [0, 0.5].
  Value *asinSmall(Value *A) {
    Value *Z = mul(A, A);
    Value *P = poly(Z, {4.2163199048e-2, 2.4181311049e-2, 4.5470025998e-2,
                        7.4953002686e-2, 1.6666752422e-1});
    return fma(mul(P, Z), A, A);
  }

  /// (1 - a) / 2 for asin's and acos's square root, where it is read for a
  /// in [0.5, 1]: 1 - a is exact there (Sterbenz), and so its half, the
  /// fma's 0.5 - 0.5 a the same; any a > 1 is negative either way, the
  /// square root's NaN (L15-E638).
  Value *halfOneMinus(Value *A) {
    return FusedExactScales ? fma(k(-0.5), A, k(0.5))
                            : mul(k(0.5), sub(k(1.0), A));
  }

  Value *asin(Value *X) {
    if (bfloatResult())
      PolyDrop = 1;
    Value *A = abs(X);
    // Above 0.5: asin(a) = pi / 2 - 2 asin(sqrt((1 - a) / 2)).
    Value *Big = gt(A, 0.5);
    Value *S = sqrt(halfOneMinus(A), /*Approx=*/true);
    // One polynomial, of the argument the range picks.
    Value *P = asinSmall(sel(Big, S, A));
    // (2 p is exact: one fma, L15-E638)
    Value *R = sel(Big,
                   FusedExactScales ? fma(k(-2.0), P, k(M_PI_2))
                                    : sub(k(M_PI_2), mul(k(2.0), P)),
                   P);
    // |x| > 1: the square root is NaN.
    R = sel(gt(A, 1.0), S, R);
    return copysign(R, X);
  }

  Value *acos(Value *X) {
    if (bfloatResult())
      PolyDrop = 1;
    Value *A = abs(X);
    // |x| > 0.5: acos(|x|) = 2 asin(sqrt((1 - |x|) / 2)), and
    // acos(-|x|) = pi - acos(|x|); else pi / 2 - asin(x).
    // (bfloat: the hardware's square root; half: one result was further
    // off with it)
    Value *S = sqrt(halfOneMinus(A), /*Approx=*/!HalfResult);
    Value *Big = gt(A, 0.5);
    // One polynomial, of the argument the range picks.
    Value *P = asinSmall(sel(Big, S, A));
    Value *Twice = mul(k(2.0), P);
    Value *Outer = sel(lt(X, 0.0), sub(k(M_PI), Twice), Twice);
    Value *Inner = sub(k(M_PI_2), copysign(P, X));
    Value *R = sel(Big, Outer, Inner);
    return sel(gt(A, 1.0), S, R);
  }

  Value *atan(Value *X) { return copysign(atanNonNegative(abs(X)), X); }

  Value *atan2(Value *Y, Value *X) {
    Value *AX = abs(X), *AY = abs(Y);
    Value *Swap = B.CreateFCmpOGT(AY, AX);
    Value *Mn = sel(Swap, AX, AY), *Mx = sel(Swap, AY, AX);
    // 0 / 0 is 0 (atan2(+-0, +-0)) and inf / inf is 1 (the diagonals).
    PreciseDivision = true;
    Value *T = div(Mn, Mx);
    T = sel(B.CreateFCmpOEQ(Mx, k(0.0)), k(0.0), T);
    T = sel(B.CreateFCmpOEQ(Mn, k(INFINITY)), k(1.0), T);
    Value *R = atanNonNegative(T, /*AtMostOne=*/true);
    R = sel(Swap, sub(k(M_PI_2), R), R);
    // x < 0 or x = -0: the left half plane.
    Value *Left = lt(copysign(k(1.0), X), 0.0);
    R = sel(Left, sub(k(M_PI), R), R);
    R = copysign(R, Y);
    Value *NaN = B.CreateFCmpUNO(X, Y);
    return sel(NaN, add(X, Y), R);
  }

  /// e^a / 2 for a >= 0, without overflow below the largest float.
  Value *halfExp(Value *A) {
    // A half result: the hardware's 2^(a log2(e)), halved (from a = 88 an
    // infinity, as is the half result from a = 11.8: L15-E329)
    if (HalfResult && HalfFastHyperbolic)
      return mul(k(0.5), hw(Intrinsic::exp2, mul(A, k(M_LOG2E))));
    // One exponential: e^a / 2, or e^(a - ln 2) from 88 on, a - ln 2 as
    // (a - 0.693359375) + 2.12194440e-4 (the first exact for a >= 88: a
    // rounded a - ln 2 put the result up to 24 ulp off near the overflow).
    // (Its +inf exactly from a = 89.4159927 (0x42b2d4fd), every float
    // checked: by a compare of a, without the clamp at 89 and the NaN
    // select, as E485's 2^x: L15-E486.)
    ArgBounds AB;
    if (HalfExpOverflowByArg && ExpNonNegative) {
      AB = boundsByArg(A, 89.4159927368164, 0.0);
      AB.Zero = false;
    }
    Value *Lt = lt(A, 88.0);
    return mul(sel(Lt, k(0.5), k(1.0)),
               expHiLo(sel(Lt, A, sub(A, k(0.693359375))),
                       sel(Lt, k(0.0), k(2.12194440e-4)), /*Bounded=*/false,
                       /*NoOverflow=*/false, /*NonNegative=*/ExpNonNegative,
                       AB));
  }

  /// 0.25 / h for h = e^a / 2 >= 0.5 (or a NaN): 0.25 times the correctly
  /// rounded reciprocal y0 + y0 (1 - h y0) (as in tanh) while the quotient is
  /// above the flushed range, else 0 (h > 2^124, an infinite h too, whose
  /// reciprocal step would be a NaN), without the general division.
  Value *quarterOver(Value *H) {
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {H});
    // (a half or bfloat result: no Newton step, L15-E329, E338)
    Value *Y = NarrowResult && HalfFastHyperbolic
                   ? Y0
                   : fma(fma(B.CreateFNeg(H), Y0, k(1.0)), Y0, Y0);
    // (a half result is infinite from h = 2^16: no select)
    if (HalfResult && HalfFastHyperbolic)
      return mul(k(0.25), Y);
    return sel(gt(H, 0x1p124), k(0.0), mul(k(0.25), Y));
  }

  Value *sinh(Value *X) {
    if (bfloatResult())
      PolyDrop = 2;
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small = fma(mul(poly(Z, {2.03721912945e-4, 8.33028376239e-3,
                                    1.66667160211e-1}),
                           Z),
                       X, X);
    Value *H = halfExp(A);
    Value *Big = copysign(sub(H, quarterOver(H)), X);
    return sel(gt(A, 1.0), Big, Small);
  }

  Value *cosh(Value *X) {
    if (bfloatResult())
      PolyDrop = 2;
    Value *H = halfExp(abs(X));
    return add(H, quarterOver(H));
  }

  Value *tanh(Value *X) {
    if (bfloatResult())
      PolyDrop = 1;
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small =
        fma(mul(poly(Z, {-5.70498872745e-3, 2.06390887954e-2,
                         -5.37397155531e-2, 1.33314422036e-1,
                         -3.33332819422e-1}),
                Z),
            X, X);
    // 1 - 2 / (e^2a + 1), which is 1 in float from a = 9 on. 2a is in
    // [0, 18] (or a NaN, which the exponential keeps): its exponential needs
    // no clamp of the argument.
    // (Its n is in [0, 26]: e^r 2^n is a normal float, neither the +inf nor
    // the +0 of ldexp's selects (L15-E274).)
    Value *S =
        HalfResult && HalfFastHyperbolic
            ? hw(Intrinsic::exp2, mul(k(2.0 * M_LOG2E), A))
            : expHiLo(TanhDoubledClamp
                          // 2 min(a, 9) as min(a + a, 18): the doubling
                          // reads x with the abs modifier, no select of |x|
                          // (an and with 0x7fffffff) (L15-E471)
                          ? sel(gt(A, 9.0), k(18.0), add(A, A))
                          : mul(k(2.0), sel(gt(A, 9.0), k(9.0), A)),
                      nullptr,
                      /*Bounded=*/true, /*NoOverflow=*/ExpNonNegative,
                      /*NonNegative=*/ExpNonNegative);
    // 2 / d for d = s + 1 in [2, e^18 + 1] (or a NaN): twice the correctly
    // rounded reciprocal y0 + y0 (1 - d y0) (lowerFDIV32's, whose rcp and
    // fmas scale with d's exponent here, no result near the flushed or the
    // infinite range), without the general division's exponent scaling and
    // special cases.
    Value *D = add(S, k(1.0));
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {D});
    Value *Y = NarrowResult && HalfFastHyperbolic
                   ? Y0
                   : fma(fma(B.CreateFNeg(D), Y0, k(1.0)), Y0, Y0);
    // (2 y is exact: one fma, L15-E638)
    Value *Big = copysign(FusedExactScales ? fma(k(-2.0), Y, k(1.0))
                                           : sub(k(1.0), mul(k(2.0), Y)),
                          X);
    return sel(lt(A, 0.625), Small, Big);
  }

  Value *tan(Value *X) {
    // A half or bfloat result: one sine-and-cosine expansion (its two-part
    // reduction, E350), every half and bfloat input's tangent the same
    // (L15-E353; the hardware's sin and cos were not)
    if (NarrowResult && NarrowSinCosReduction) {
      auto [S, C] = sinAndCos(X, TanQuotientSigns);
      return div(S, C);
    }
    // (float: the expansions sin and cos get, as one)
    if (TanQuotientSigns && !NarrowResult) {
      auto [S, C] = sinAndCos(X, true);
      return div(S, C);
    }
    return div(call(Intrinsic::sin, {X}), call(Intrinsic::cos, {X}));
  }

  Value *asinh(Value *X) {
    if (bfloatResult())
      PolyDrop = 1;
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small =
        fma(mul(poly(Z, {2.0122003309e-2, -4.2699340972e-2, 7.4847586088e-2,
                         -1.6666288134e-1}),
                Z),
            X, X);
    // log(2a) = log(a) + ln 2 once a * a + 1 is a * a in float (and before
    // it overflows): one log, of the argument the range picks.
    Value *H = gt(A, 1500.0);
    Value *L = log(sel(H, A, add(A, sqrt(fma(A, A, k(1.0))))));
    Value *Big = copysign(add(L, sel(H, k(M_LN2), k(0.0))), X);
    return sel(lt(A, 0.5), Small, Big);
  }

  Value *acosh(Value *X) {
    if (bfloatResult())
      PolyDrop = 1;
    Value *Z = sub(X, k(1.0));
    // Near 1: sqrt(z) times a polynomial (and NaN below 1).
    Value *Near = mul(sqrt(Z),
                      poly(Z, {1.7596881071e-3, -7.5272886713e-3,
                               2.6454905019e-2, -1.1784741703e-1,
                               1.4142135263e0}));
    // As in asinh: one log.
    Value *H = gt(X, 1500.0);
    Value *L = log(sel(H, X, add(X, sqrt(mul(Z, add(X, k(1.0)))))));
    return sel(lt(Z, 0.5), Near, add(L, sel(H, k(M_LN2), k(0.0))));
  }

  Value *atanh(Value *X) {
    if (bfloatResult())
      PolyDrop = 1;
    Value *A = abs(X);
    Value *Z = mul(X, X);
    Value *Small =
        fma(mul(poly(Z, {1.81740078349e-1, 8.24370301058e-2, 1.46691431730e-1,
                         1.99782164500e-1, 3.33337300303e-1}),
                Z),
            X, X);
    // (bfloat: the correctly rounded division; the reciprocal of 1 - a
    // underflows for a beyond 2^126, which gave +-inf, not a NaN. half: the
    // same results either way, L15-E339)
    PreciseDivision = !HalfResult;
    Value *Big = copysign(
        mul(k(0.5), log(div(add(k(1.0), A), sub(k(1.0), A)))), X);
    return sel(lt(A, 0.5), Small, Big);
  }

  /// x - 2 round(x / 2), exactly: in [-1, 1].
  Value *halfTurns(Value *X) {
    Value *R = call(Intrinsic::roundeven, {mul(X, k(0.5))});
    return FusedExactScales ? fma(k(-2.0), R, X) : sub(X, mul(k(2.0), R));
  }

  /// sin(pi r) and cos(pi r) for |r| <= 1/4, with pi r as a rounded
  /// product and its error.
  std::pair<Value *, Value *> sinCosPi(Value *R) {
    constexpr double PiLo = 3.14159265358979323846 - (double)(float)M_PI;
    Value *Hi = mul(R, k(M_PI));
    Value *Lo = fma(R, k(M_PI), B.CreateFNeg(Hi));
    Lo = fma(R, k(PiLo), Lo);
    auto [S, C] = sinCosPolys(Hi);
    return {fma(Lo, C, S), fma(B.CreateFNeg(Lo), S, C)};
  }

  Value *sinpi(Value *X) {
    // sin(pi r) = sin(pi (sign(r) - r)) folds r into [-1/2, 1/2], and
    // = cos(pi (sign(r) / 2 - r)) beyond 1/4; every step is exact.
    Value *R = halfTurns(X);
    Value *R2 = sel(gt(abs(R), 0.5), sub(copysign(k(1.0), R), R), R);
    Value *UseCos = gt(abs(R2), 0.25);
    Value *R3 = sel(UseCos, sub(copysign(k(0.5), R2), R2), R2);
    auto [S, C] = sinCosPi(R3);
    Value *V = sel(UseCos, copysign(C, R2), S);
    // Integers: a zero with the sign of x.
    return sel(B.CreateFCmpOEQ(X, call(Intrinsic::trunc, {X})),
               copysign(k(0.0), X), V);
  }

  Value *cospi(Value *X) {
    // cos(pi r) = -cos(pi (1 - r)) folds |r| into [0, 1/2], and
    // = sin(pi (1/2 - r)) beyond 1/4, exactly 0 at the half integers.
    Value *R = abs(halfTurns(X));
    Value *Big = gt(R, 0.5);
    Value *R2 = sel(Big, sub(k(1.0), R), R);
    Value *UseSin = gt(R2, 0.25);
    Value *R3 = sel(UseSin, sub(k(0.5), R2), R2);
    auto [S, C] = sinCosPi(R3);
    Value *V = sel(UseSin, S, C);
    return sel(Big, B.CreateFNeg(V), V);
  }

  Value *tanpi(Value *X) {
    if (bfloatResult())
      PolyDrop = 1;
    // (half: the correctly rounded division; with the reciprocal's, 2 half
    // results were further off. bfloat: the same results either way)
    PreciseDivision = HalfResult;
    return div(sinpi(X), cospi(X));
  }

  // Precise sin, cos, exp and log.

  Type *intTy() { return Ty->getWithNewType(B.getInt32Ty()); }
  Value *ki(int32_t V) { return ConstantInt::get(intTy(), V); }
  Value *isNaN(Value *X) { return B.CreateFCmpUNO(X, X); }
  /// a + b as the rounded sum and its error (Knuth's two-sum).
  std::pair<Value *, Value *> twoSum(Value *A, Value *X) {
    Value *S = add(A, X);
    Value *BB = sub(S, A);
    Value *Err = add(sub(A, sub(S, BB)), sub(X, BB));
    return {S, Err};
  }

  /// a + x and its error for |a| >= |x| or a = 0 (Fast2Sum): the error is
  /// exact, so the same as twoSum's.
  std::pair<Value *, Value *> fastTwoSum(Value *A, Value *X) {
    Value *S = add(A, X);
    return {S, sub(X, sub(S, A))};
  }

  /// 2^i for an i32 i in [-126, 127].
  Value *pow2i(Value *I) {
    return B.CreateBitCast(B.CreateShl(B.CreateAdd(I, ki(127)), ki(23)), Ty);
  }
  /// v * 2^n for an integral n in [-252, 254], in two steps. The split is
  /// done on the integer: i >> 1 and i - (i >> 1) are both in [-126, 127].
  /// \p MagicBits: n's integer in the low bits of a float's (n + 1.5 2^23,
  /// whose bits are 0x4b400000 + n), instead of converting n.
  /// \p ExpArg: the clamped argument of exp (no low part), whose result
  /// overflows exactly from 0x42b17218 (88.7228394) on.
  Value *ldexp(Value *V, Value *N, bool NoOverflow = false,
              bool NoUnderflow = false, Value *MagicBits = nullptr,
              ArgBounds AB = {}) {
    Value *ExpArg = AB.Arg;
    // Frozen: a NaN n (from a NaN argument) gives some scale, and the NaN
    // reaches the result through the value it scales, without a select.
    Value *I = MagicBits ? nullptr : B.CreateFreeze(B.CreateFPToSI(N, intTy()));
    if (NoDenormals && IntegerLdexp) {
      // (0x4b400000 << 23 is 0 mod 2^32: the magic bits shift as n does)
      if (MagicBits)
        I = B.CreateFreeze(MagicBits);
      // v is the exponential's e^r in [0.7, 1.42] (or a NaN): v 2^n is its
      // bits plus n << 23 while that is a normal float; at or past the
      // infinite exponent +inf, below the normal range the +0 that the
      // flushing multiplications gave, and a NaN v itself (its n arbitrary).
      // Bit-identical to the two multiplications where results flush.
      Value *S = B.CreateAdd(B.CreateBitCast(V, intTy()),
                             B.CreateShl(I, ki(23)));
      Value *R = B.CreateBitCast(S, Ty);
      // A NaN's magic bits add 0 to its own bits (E472), but they are past
      // the +inf bound: for exp, its argument's compare instead, which a NaN
      // fails, and then no NaN select (L15-E473).
      bool ArgBound = ExpArg && MagicBits && NativeNaN && ExpOverflowByArg;
      if (!NoOverflow)
        R = ArgBound ? sel(AB.Negated ? B.CreateFCmpOLE(ExpArg, k(-AB.OverAt))
                                      : B.CreateFCmpOGE(ExpArg, k(AB.OverAt)),
                           k(INFINITY), R)
                     : sel(B.CreateICmpSGE(S, ki(0x7f800000)), k(INFINITY), R);
      // (and +0 exactly up to -87.3365479, 0xc2aeac50: L15-E474)
      if (!NoUnderflow)
        R = ArgBound && AB.Zero
                ? sel(AB.Negated ? B.CreateFCmpOGE(ExpArg, k(-AB.ZeroAt))
                                 : B.CreateFCmpOLE(ExpArg, k(AB.ZeroAt)),
                      k(0.0), R)
                : sel(B.CreateICmpSLT(S, ki(0x00800000)), k(0.0), R);
      // A NaN v comes from a NaN argument, whose sum t + 1.5 2^23 is then
      // the NaN 0x7fc00000, 0 shifted by 23: s is v's bits, which no +inf
      // select overrides (L15-E472).
      if (MagicBits && NoOverflow && NativeNaN && ExpNaNThroughScale)
        return R;
      if (ArgBound)
        return R;
      return sel(B.CreateFCmpUNO(V, V), V, R);
    }
    if (!I)
      I = B.CreateFreeze(B.CreateSub(MagicBits, ki(0x4b400000)));
    Value *I1 = B.CreateAShr(I, ki(1));
    return mul(mul(V, pow2i(I1)), pow2i(B.CreateSub(I, I1)));
  }

  /// e^r for |r| <= ln(2) / 2 (Cephes expf).
  Value *expCore(Value *R) {
    // (degree 4, weighted minimax over |r| <= ln(2) / 2, the coefficients
    // then searched in float: 0.05 ulp of e^r where Cephes' degree 5 is
    // 0.02, one fma fewer; the maximum error stays 1 ulp, L15-E712)
    // (for float results: half and bfloat ones keep Cephes' terms, which
    // a bfloat result drops two of, L15-E351)
    Value *P =
        ExpPoly5 && !NarrowResult
            ? poly(R, {0.0013814327539876103, 0.008368846029043198,
                       0.04166828468441963, 0.1666651964187622,
                       0.4999999403953552})
            : poly(R, {1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3,
                       4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1});
    // (fma(p r, r, r) reads p r and r, a 1-cycle fma where p, r r and r
    // took 2, L15-E704)
    if (ExpCoreProduct)
      return add(fma(mul(P, R), R, R), k(1.0));
    return add(fma(P, mul(R, R), R), k(1.0));
  }

  /// e^(hi + lo), lo a small correction (or none: an addition of +0.0 only
  /// changes r = -0 to +0, for which expCore gives 1 either way): n =
  /// round(hi / ln 2) is split off exactly (ln 2 in two parts) and the rest
  /// goes to expCore.
  /// \p Bounded: the caller clamped its argument so that hi is finite and
  /// its n within ldexp's range (exp2, exp10), and the result over- and
  /// underflows by itself: no clamp of hi here and no +inf / +0 selects.
  /// \p NonNegative: hi is at least 0 (or a NaN) and lo, if any, at least 0:
  /// n >= 0 and the result at least e^r >= 0.7, so neither the clamp at -104
  /// nor ldexp's +0 select, nor (lo being one) the selects below (L15-E274).
  /// \p AB: 2^x / 10^x with their +inf / +0 by their argument's compares;
  /// Bounded then means only that no selects of hi are needed here (the
  /// argument is not clamped, its garbage beyond the bounds selected away).
  Value *expHiLo(Value *Hi, Value *Lo, bool Bounded = false,
                 bool NoOverflow = false, bool NonNegative = false,
                 ArgBounds AB = {}) {
    Value *C;
    Value *Y;
    // exp's +inf and +0 selects by its argument's compares (E473, E474) give
    // +inf and +0 beyond the clamps as well (a NaN failing both): no clamps.
    // (Its n and r are then garbage there, an infinity's a NaN, and the
    // integer sum wraps: all selected away.)
    // (Not below a maximum: there only the clamp at -104 would go, and the
    // argument held to the end took a register: attn_decode_f16 +1.1% tput.)
    bool FlushedUnderflow = ExpResultFlushed && ExpFlushedUnderflow &&
                            ExpMagicRound && NoDenormals && IntegerLdexp;
    bool Unclamped = !Lo && !Bounded && !NoOverflow && ExpUnclamped &&
                     ExpOverflowByArg && ExpMagicRound && NativeNaN &&
                     NoDenormals && IntegerLdexp;
    if (Unclamped || AB.Arg) {
      C = Hi;
    } else if (NoOverflow && !Bounded && !Lo && AbsNegatedClamp &&
               match(Hi, m_FNeg(m_FAbs(m_Value(Y))))) {
      // -|y| (an online softmax's shared exponent, L15-E529): |y| above the
      // mirrored bound selects it, else y itself, and -|that| is the clamped
      // value: the compare takes y's .abs and the multiplication and fma
      // its .neg.abs, so neither -|y| nor the select's operand needs an
      // instruction of its own (a select's operand takes no modifier).
      double Bound = FlushedUnderflow ? 87.3365478515625 : 104.0;
      Value *AbsY = cast<Instruction>(Hi)->getOperand(0);
      C = B.CreateFNeg(B.CreateUnaryIntrinsic(
          Intrinsic::fabs, sel(gt(AbsY, Bound), k(Bound), Y)));
    } else if (NoOverflow && !Bounded && !Lo && FlushedUnderflow) {
      // (E474's +0 bound as the clamp: there the scaled bits are 0x007fffxx,
      // a denormal its users flush to the +0 the select gave: L15-E479)
      C = sel(lt(Hi, -87.3365478515625), k(-87.3365478515625), Hi);
    } else if (NoOverflow && !Bounded && !Lo) {
      // (at most a denormal above 0: only the clamp below)
      C = sel(lt(Hi, -104.0), k(-104.0), Hi);
    } else if (!Bounded && NegatedClamp && match(Hi, m_FNeg(m_Value(Y)))) {
      // e^-y (sigmoid, silu): clamp y to the mirrored bounds and negate the
      // result, the same value (a NaN y stays a NaN), so that the negation
      // folds into the multiplication's constant and the fma's .neg instead
      // of being an xor (a select's operand takes no modifier).
      C = B.CreateFNeg(sel(lt(Y, -89.0), k(-89.0),
                           sel(gt(Y, 104.0), k(104.0), Y)));
    } else {
      C = Bounded       ? Hi
          : NonNegative ? sel(gt(Hi, 89.0), k(89.0), Hi)
                        : sel(gt(Hi, 89.0), k(89.0),
                              sel(lt(Hi, -104.0), k(-104.0), Hi));
    }
    // (the fma reads x and two uniforms; with the constants in registers it
    // would read three, no faster than the product and sum)
    bool Fused = ExpMagicRound && ExpFusedRound && AGX::uniformConstantsEnabled();
    Value *T = Fused ? nullptr : mul(C, k(M_LOG2E));
    Value *N, *MagicBits = nullptr;
    if (ExpMagicRound) {
      // |t| <= 151: t + 1.5 2^23 rounds to the integer nearest t, ties to
      // even (1.5 2^23 is even), on the FP pipe instead of the shared
      // units' rint and convert (L15-E469); n = that - 1.5 2^23 exactly.
      // (As one fma, the integer nearest x log2(e) unrounded: any n near
      // it reduces x as well, r staying within a few ulps of ln(2) / 2,
      // L15-E706.)
      Value *M = Fused ? fma(C, k(M_LOG2E), k(12582912.0))
                               : add(T, k(12582912.0));
      N = sub(M, k(12582912.0));
      MagicBits = B.CreateBitCast(M, intTy());
    } else {
      N = call(Intrinsic::roundeven, {T});
    }
    Value *R = fma(B.CreateFNeg(N), k(0.693359375), C);
    R = fma(B.CreateFNeg(N), k(-2.12194440e-4), R);
    // e^(hi + lo) of an unbounded hi (pow): lo as 0 where |hi| >= 89 (or a
    // NaN). There the clamped hi alone overflows (e^89 > the largest float)
    // or flushes (below -89 the result is below 2^-126), so the result's
    // +inf / +0 selects below are needed only for an infinite hi, whose lo
    // is a NaN or an infinity: one select of lo for two (L15-E709).
    bool LowSelect = ExpLowSelect && Lo && !Bounded && !NonNegative &&
                     NoDenormals;
    if (LowSelect)
      Lo = sel(lt(abs(Hi), 89.0), Lo, k(0.0));
    // (Bounded with a low part and NoOverflow: 2^x / 10^x below a maximum,
    // hi <= 0 and so n <= 0, the scaled e^r below 2: L15-E484.)
    Value *E = ldexp(expCore(Lo ? add(R, Lo) : R), N,
                     NoOverflow && (!Lo || Bounded),
                     NonNegative ||
                         (NoOverflow && !Bounded && !Lo && FlushedUnderflow),
                     MagicBits,
                     AB.Arg ? AB
                     : !Lo && !Bounded
                         ? ArgBounds{C, 88.72283935546875, -87.3365478515625,
                                     ExpUnclamped && !NoOverflow}
                         : ArgBounds{});
    // Beyond the clamps the result is +inf (e^89 r overflows) and +0 (e^-104
    // rounds to 0) by itself: for exp, checked for every float x, flushing
    // and not; with a low part (exp2, exp10, pow) the selects stay.
    // Non-negative: past 89, hi is clamped to 89, whose n = 128 and e^r >=
    // e^0.277 make ldexp's exponent sum at least the infinite one, its +inf
    // select: neither select here.
    if (Lo && !Bounded && !NonNegative && !LowSelect)
      E = sel(gt(Hi, 89.0), k(INFINITY), sel(lt(Hi, -104.0), k(0.0), E));
    // A NaN hi is a NaN r, which makes E a NaN (ldexp freezes its
    // conversion of the NaN n, which would be poison).
    return E;
  }

  /// x clamped to [Lo, Hi], a little beyond where b^x overflows and rounds
  /// to 0: no infinity whose product error would be a NaN, and the same
  /// result for every float x as clamping e^(hi + lo) (checked exhaustively
  /// for 2^x and 10^x, flushing and not). A NaN stays one.
  /// \p BelowMax: x is at most 0, a denormal or a NaN (below a maximum), so
  /// only the lower clamp is needed (L15-E484).
  Value *clampArg(Value *X, double Lo, double Hi, bool BelowMax = false) {
    Value *Y;
    if (BelowMax) {
      if (NegatedClamp && match(X, m_FNeg(m_Value(Y))))
        return B.CreateFNeg(sel(gt(Y, -Lo), k(-Lo), Y));
      return sel(lt(X, Lo), k(Lo), X);
    }
    // -y: y clamped to the mirrored bounds, then negated (as in expHiLo),
    // so that the negation folds into the multiplications that follow.
    if (NegatedClamp && match(X, m_FNeg(m_Value(Y))))
      return B.CreateFNeg(sel(lt(Y, -Hi), k(-Hi), sel(gt(Y, -Lo), k(-Lo), Y)));
    return sel(gt(X, Hi), k(Hi), sel(lt(X, Lo), k(Lo), X));
  }

  /// 2^x's / 10^x's +inf and +0 by compares of x (for -y, of y), where the
  /// scaling is the magic sum's integer addition and NaNs are native: then
  /// no clamps of x, whose n, r and integer sum beyond the bounds are
  /// garbage the selects replace (no poison: no conversion, no nsw), and no
  /// NaN select (a NaN x fails both compares; its magic bits shift to 0 and
  /// leave e^r's NaN) (L15-E485).
  ArgBounds boundsByArg(Value *X, double OverAt, double ZeroAt) {
    if (!Exp2BoundsByArg || Exp2DirectReduction || !ExpMagicRound ||
        !NativeNaN || !NoDenormals || !IntegerLdexp || !ExpOverflowByArg)
      return {};
    Value *Y;
    if (NegatedClamp && match(X, m_FNeg(m_Value(Y))))
      return {Y, OverAt, ZeroAt, true, true};
    return {X, OverAt, ZeroAt, true, false};
  }

  Value *exp2Precise(Value *X, bool NoOverflow = false) {
    // 2^x = e^(x ln 2), with x ln 2 as a rounded product and its error.
    // (Up to 128, where 2^x overflows: n = 128 at most, which the exponent
    // addition of ldexp needs.)
    bool BelowMax = belowMax(X, NoOverflow);
    // (+inf exactly from 128, +0 exactly at or below -126 - 2^-17, every
    // float checked: L15-E485)
    ArgBounds AB = boundsByArg(X, 128.0, -126.00000762939453);
    if (!AB.Arg)
      X = clampArg(X, -151.0, 128.0, BelowMax);
    if (Exp2DirectReduction) {
      // n = roundeven(x), f = x - n exact in [-1/2, 1/2], e^(f ln 2) with
      // f ln 2 as a rounded product and its error (L15-E340)
      Value *N = call(Intrinsic::roundeven, {X});
      Value *F = sub(X, N);
      Value *H = mul(F, k(M_LN2));
      Value *L = fma(F, k(M_LN2), B.CreateFNeg(H));
      L = fma(F, k(M_LN2 - (double)(float)M_LN2), L);
      return ldexp(expCore(add(H, L)), N);
    }
    Value *Hi = mul(X, k(M_LN2));
    Value *Lo = fma(X, k(M_LN2), B.CreateFNeg(Hi));
    Lo = fma(X, k(M_LN2 - (double)(float)M_LN2), Lo);
    return expHiLo(Hi, Lo, /*Bounded=*/true, BelowMax, false, AB);
  }
  Value *exp10Precise(Value *X, bool NoOverflow = false) {
    constexpr double LN10 = 2.302585092994045684;
    // (Up to 38.6, past 10^x's overflow at 38.53: n = 128 at most.)
    bool BelowMax = belowMax(X, NoOverflow);
    // (+inf exactly from 38.5318413 (0x421a209b), +0 exactly at or below
    // -37.9297829 (0xc217b819), every float checked: L15-E485)
    ArgBounds AB = boundsByArg(X, 38.53184127807617, -37.92978286743164);
    if (!AB.Arg)
      X = clampArg(X, -46.0, 38.6, BelowMax);
    Value *Hi = mul(X, k(LN10));
    Value *Lo = fma(X, k(LN10), B.CreateFNeg(Hi));
    Lo = fma(X, k(LN10 - (double)(float)LN10), Lo);
    return expHiLo(Hi, Lo, /*Bounded=*/true, BelowMax, false, AB);
  }

  /// A positive finite x as 2^e * (1 + f), 1 + f in [sqrt(1/2), sqrt(2)),
  /// with log(1 + f) = f + y (Cephes logf). Denormals are scaled first.
  struct LogParts {
    Value *E, *F, *Y;
  };
  LogParts logParts(Value *X) {
    // A denormal x is scaled by 2^23 first, where it must be kept; where f32
    // arithmetic flushes, it may be taken as the zero it compares equal to,
    // which logSpecial's x == 0 (a flushing compare) turns into -inf.
    Value *Tiny = NoDenormals ? nullptr : lt(X, 1.17549435e-38);
    Value *XS = Tiny ? sel(Tiny, mul(X, k(8388608.0)), X) : X;
    Value *Bits = B.CreateBitCast(XS, intTy());
    // x's bits plus those of 1 less those of the float above sqrt(1/2): the
    // exponent field one more from a mantissa above sqrt(2)'s on, the
    // mantissa less 1 - sqrt(1/2)'s, so that 1 + f is the low bits plus
    // sqrt(1/2)'s, in (sqrt(1/2), sqrt(2)] and e as the compare and selects
    // below give for every positive finite x (L15-E615).
    // Doubled, which shifts the sign out (no and of |x|'s, one iadd with its
    // shift): the exponent field from bit 24, the mantissa from bit 1.
    Value *E;
    if (LogOffsetSplit) {
      Bits = B.CreateAdd(B.CreateShl(Bits, ki(1)),
                         ki(2 * (0x3f800000 - 0x3f3504f4)));
      Value *Field = B.CreateLShr(Bits, ki(24));
      E = LogExponentByAdd
              ? add(B.CreateBitCast(B.CreateOr(Field, ki(0x4b000000)), Ty),
                    k(-8388735.0))
              : B.CreateSIToFP(B.CreateSub(Field, ki(127)), Ty);
    } else {
    // (the biased exponent field under 2^23's bits is the float 2^23 + e;
    // less 2^23 + 127 exactly: L15-E614)
    E =
        LogExponentByAdd
            ? add(B.CreateBitCast(
                      B.CreateOr(B.CreateAnd(B.CreateLShr(Bits, ki(23)),
                                             ki(0xff)),
                                 ki(0x4b000000)),
                      Ty),
                  k(-8388735.0))
            : B.CreateSIToFP(
                  B.CreateSub(B.CreateLShr(Bits, ki(23)), ki(127)), Ty);
    }
    if (Tiny)
      E = sel(Tiny, sub(E, k(23.0)), E);
    Value *M;
    if (LogOffsetSplit) {
      M = B.CreateBitCast(
          B.CreateAdd(B.CreateAnd(B.CreateLShr(Bits, ki(1)), ki(0x7fffff)),
                      ki(0x3f3504f4)),
          Ty);
    } else {
      M = B.CreateBitCast(
          B.CreateOr(B.CreateAnd(Bits, ki(0x7fffff)), ki(0x3f800000)), Ty);
      Value *Big = gt(M, 1.41421356);
      M = sel(Big, mul(M, k(0.5)), M);
      E = sel(Big, add(E, k(1.0)), E);
    }
    Value *F = sub(M, k(1.0));
    Value *Z = mul(F, F);
    Value *P = poly(F, {7.0376836292e-2, -1.1514610310e-1, 1.1676998740e-1,
                        -1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1,
                        2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1});
    Value *Y = fma(k(-0.5), Z, mul(mul(F, Z), P));
    return {E, F, Y};
  }
  /// The value of log for the special inputs (NaN, negative, 0, infinity),
  /// else \p R.
  Value *logSpecial(Value *X, Value *R) {
    R = sel(B.CreateFCmpOEQ(X, k(INFINITY)), X, R);
    R = sel(B.CreateFCmpOEQ(X, k(0.0)), k(-INFINITY), R);
    return sel(B.CreateFCmpULT(X, k(0.0)), k(NAN), R);
  }
  /// log(x) for a positive finite x as a sum hi + lo.
  std::pair<Value *, Value *> logHiLo(Value *X) {
    LogParts L = logParts(X);
    // (|f| > |y|: y is about -f^2 / 2 with |f| <= 0.415; |e ln 2| from
    // 0.693 above |f + y| <= 0.347 unless e = 0, its product +0: L15-E616)
    auto Sum = [&](Value *A, Value *X) {
      return LogFastTwoSum ? fastTwoSum(A, X) : twoSum(A, X);
    };
    auto [S, Err] = Sum(L.F, L.Y);
    // e ln 2 in two parts, the first exact for |e| < 2^15.
    auto [Hi, Err2] = Sum(mul(L.E, k(0.693359375)), S);
    // (e ln2_lo and the errors' sum as one fma, one rounding: with the
    // constant a uniform it reads e and the sum, L15-E708)
    Value *Lo = LogLowFma && AGX::uniformConstantsEnabled()
                    ? fma(L.E, k(-2.12194440e-4), add(Err, Err2))
                    : add(add(Err, Err2), mul(L.E, k(-2.12194440e-4)));
    return {Hi, Lo};
  }
  /// The x of log's polynomial: |x|, pow's, where both are of one value
  /// (logSpecial replaces every result of a negative x or a NaN).
  Value *logCoreArg(Value *X) { return LogAbsCore ? abs(X) : X; }
  Value *logPrecise(Value *X) {
    auto [Hi, Lo] = logHiLo(logCoreArg(X));
    return logSpecial(X, add(Hi, Lo));
  }
  /// log2 and log10: e log_b(2) + (f + y) log_b(e), each constant in a
  /// part with few bits and a correction (Cephes log10f).
  Value *logBase(Value *X, double EHi, double ELo, double L2Hi, double L2Lo) {
    LogParts L = logParts(logCoreArg(X));
    Value *R = mul(L.Y, k(ELo));
    R = fma(L.F, k(ELo), R);
    R = fma(L.Y, k(EHi), R);
    R = fma(L.F, k(EHi), R);
    // (log2: no low part. e is a finite integer, so e 0 + r only turns an
    // r = -0 into +0, which e + r does as well where e = +0 and leaves out
    // otherwise: the same sum without it, L15-E328)
    if (L2Lo != 0.0)
      R = fma(L.E, k(L2Lo), R);
    R = fma(L.E, k(L2Hi), R);
    return logSpecial(X, R);
  }
  Value *log2Precise(Value *X) {
    // log2(e) = 1.4375 + 0.0051950..., log2(2) = 1.
    return logBase(X, 1.4375, M_LOG2E - 1.4375, 1.0, 0.0);
  }
  Value *log10Precise(Value *X) {
    return logBase(X, 0.43359375, 7.00731903251827651129e-4, 3.0078125e-1,
                   2.48745663981195213739e-4);
  }

  Value *powPrecise(Value *X, Value *Y) {
    Value *AX = abs(X);
    auto [LHi, LLo] = logHiLo(AX);
    Value *Hi = mul(Y, LHi);
    Value *Lo = fma(Y, LHi, B.CreateFNeg(Hi));
    Lo = fma(Y, LLo, Lo);
    Value *R = expHiLo(Hi, Lo);
    // log(0) = -inf and log(inf) = inf: the product is +-inf (or NaN for a
    // zero y, handled below), the result 0 or inf.
    // (a finite nonzero constant y of at most 2^20: y log|x| is finite but
    // for |x| = 0 and inf, whose results y's sign gives, two selects, not
    // logSpecial's three, the product and its test: L15-E648)
    auto *YC = dyn_cast<ConstantFP>(Y);
    if (PowConstantEnds && YC && YC->getValueAPF().isFinite() &&
        !YC->isZero() && std::fabs(YC->getValueAPF().convertToDouble()) <= 0x1p20) {
      bool Pos = !YC->isNegative();
      R = sel(B.CreateFCmpOEQ(AX, k(0.0)), k(Pos ? 0.0 : INFINITY), R);
      R = sel(B.CreateFCmpOEQ(AX, k(INFINITY)), k(Pos ? INFINITY : 0.0), R);
    } else {
      Value *L = logSpecial(AX, LHi);
      Value *Inf = mul(Y, L);
      R = sel(B.CreateFCmpOEQ(abs(Inf), k(INFINITY)),
              sel(gt(Inf, 0.0), k(INFINITY), k(0.0)), R);
    }
    // (a finite nonzero constant y: x = 1 gives exactly 1 already, log(1)
    // being 0 and e^(+-0) 1, and a non-integer y's NaN of a negative finite
    // x and a NaN x's are one select of ult(x, 0) and x != -inf, L15-E718)
    if (PowConstantTail && YC && YC->getValueAPF().isFinite() &&
        !YC->isZero()) {
      APFloat YV = YC->getValueAPF();
      APFloat T = YV;
      T.roundToIntegral(APFloat::rmTowardZero);
      bool Integer = T.compare(YV) == APFloat::cmpEqual;
      if (Integer) {
        APFloat H = YV;
        H.divide(APFloat(YV.getSemantics(), "2"), APFloat::rmNearestTiesToEven);
        APFloat HT = H;
        HT.roundToIntegral(APFloat::rmTowardZero);
        if (HT.compare(H) != APFloat::cmpEqual)
          R = sel(lt(copysign(k(1.0), X), 0.0), B.CreateFNeg(R), R);
        return sel(B.CreateFCmpUNO(X, X), k(NAN), R);
      }
      return sel(B.CreateAnd(B.CreateFCmpULT(X, k(0.0)),
                             B.CreateFCmpUNE(X, k(-INFINITY))),
                 k(NAN), R);
    }
    // A negative x: an odd integer y gives the sign, any other finite
    // non-integer a NaN (for x = -inf, the infinity or zero above).
    Value *YInt = B.CreateFCmpOEQ(call(Intrinsic::trunc, {Y}), Y);
    Value *Half = mul(Y, k(0.5));
    Value *YOdd = B.CreateAnd(
        YInt, B.CreateFCmpONE(call(Intrinsic::trunc, {Half}), Half));
    Value *Neg = lt(copysign(k(1.0), X), 0.0);
    R = sel(B.CreateAnd(Neg, YOdd), B.CreateFNeg(R), R);
    Value *YFinite = B.CreateFCmpOLT(abs(Y), k(INFINITY));
    Value *NegFinite = B.CreateAnd(lt(X, 0.0), gt(X, -INFINITY));
    R = sel(B.CreateAnd(B.CreateAnd(NegFinite, YFinite), B.CreateNot(YInt)),
            k(NAN), R);
    // |x| = 1 with y infinite, x = 1, y = 0: 1 (even with a NaN).
    R = sel(B.CreateAnd(B.CreateFCmpOEQ(AX, k(1.0)),
                        B.CreateNot(YFinite)),
            k(1.0), R);
    R = sel(B.CreateFCmpUNO(X, Y), k(NAN), R);
    R = sel(B.CreateFCmpOEQ(X, k(1.0)), k(1.0), R);
    return sel(B.CreateFCmpOEQ(Y, k(0.0)), k(1.0), R);
  }

  /// sin(r) and cos(r) for |r| <= pi / 4 (Cephes sinf and cosf).
  std::pair<Value *, Value *> sinCosPolys(Value *R) {
    Value *Z = mul(R, R);
    Value *S = fma(mul(poly(Z, {-1.9515295891e-4, 8.3321608736e-3,
                                -1.6666654611e-1}),
                       Z),
                   R, R);
    // (fma(z, fma(z, p, -1/2), 1): two fmas of two registers and a
    // constant, where z^2, 1 - z / 2 and their fma of three registers took
    // four cycles; fewer results an ulp off, L15-E705)
    Value *CP = poly(Z, {2.443315711809948e-5, -1.388731625493765e-3,
                         4.166664568298827e-2});
    Value *C = CosHorner ? fma(Z, fma(Z, CP, k(-0.5)), k(1.0))
                         : fma(mul(Z, Z), CP, fma(k(-0.5), Z, k(1.0)));
    return {S, C};
  }

  /// sin(x) and cos(x): x - j pi / 2 with pi / 2 in three parts, exact for
  /// |x| below 2^22, then Cephes' sinf and cosf polynomials by quadrant.
  /// Beyond that the hardware's reduction (afn) is used.
  /// \p ForQuotient: a numerator and denominator whose quotient is
  /// sin(x) / cos(x), signs left out: sin's sign is bit 1 of q and cos's bit
  /// 1 of q + 1, which differ exactly where q is odd, so the quotient is
  /// -(c / s) there and s / c elsewhere, and (-c) / s is -(c / s) exactly
  /// (L15-E503).
  std::pair<Value *, Value *> sinAndCos(Value *X, bool ForQuotient = false) {
    bool Fused = SinCosMagicRound && SinCosFusedRound &&
                 AGX::uniformConstantsEnabled();
    Value *T = Fused ? nullptr : mul(X, k(M_2_PI));
    Value *J, *QBits = nullptr;
    if (SinCosMagicRound) {
      // Where |x| < 2^22 (the expansion's range; beyond, the hardware's is
      // taken) |t| < 2^22: j = t + 1.5 2^23 - 1.5 2^23 rounds to even as
      // rint does, and the sum's bits are 0x4b400000 + j, whose low two
      // bits are the quadrant's (L15-E470).
      // (as one fma, as expf's: j the integer nearest x 2 / pi unrounded,
      // L15-E707)
      Value *M = Fused ? fma(X, k(M_2_PI), k(12582912.0))
                                  : add(T, k(12582912.0));
      J = sub(M, k(12582912.0));
      QBits = B.CreateBitCast(M, intTy());
    } else {
      J = call(Intrinsic::roundeven, {T});
    }
    Value *NJ = B.CreateFNeg(J);
    Value *R = fma(NJ, k(1.57079637050628662109375), X);
    R = fma(NJ, k(-4.371139000189375e-8), R);
    // (a bfloat result: pi / 2 in two parts, every bfloat's sin and cos the
    // same, L15-E350)
    if (!(NarrowResult && NarrowSinCosReduction))
      R = fma(NJ, k(-1.7151245100059780e-15), R);
    Value *Small = lt(abs(X), 4194304.0);
    Value *Q = QBits ? QBits : B.CreateFPToSI(sel(Small, J, k(0.0)), intTy());
    auto [S, C] = sinCosPolys(R);
    // sin: quadrants 1 and 3 take the cosine, 2 and 3 the minus sign; cos
    // is a quadrant ahead.
    auto Quadrant = [&](Value *Q, bool Ahead) {
      Value *Q1 = Ahead ? B.CreateAdd(Q, ki(1)) : Q;
      // (q + 1 is odd where q is not: cos's select by q's test, its operands
      // swapped, one test for both of sincos's halves; the backend does not
      // see through the add)
      bool Swap = Ahead && SinCosSharedOdd;
      Value *Odd = B.CreateICmpNE(B.CreateAnd(Swap ? Q : Q1, ki(1)), ki(0));
      Value *Minus = B.CreateICmpNE(B.CreateAnd(Q1, ki(2)), ki(0));
      Value *V = Swap ? sel(Odd, S, C) : sel(Odd, C, S);
      return sel(Minus, B.CreateFNeg(V), V);
    };
    auto Hw = [&](Intrinsic::ID ID) {
      Value *V = B.CreateIntrinsic(ID, {Ty}, {X});
      cast<Instruction>(V)->setHasApproxFunc(true);
      // (for |x| from 2^22 only: AGXCodeGenPrepare branches around it)
      cast<Instruction>(V)->setMetadata("agx.rare",
                                        MDNode::get(B.getContext(), {}));
      return V;
    };
    if (ForQuotient) {
      Value *Odd = B.CreateICmpNE(B.CreateAnd(Q, ki(1)), ki(0));
      // (both before the selects of the range: AGXRareFallbacks moves the
      // hardware's into one block)
      Value *N = sel(Odd, B.CreateFNeg(C), S), *D = sel(Odd, S, C);
      Value *HwN = Hw(Intrinsic::sin), *HwD = Hw(Intrinsic::cos);
      return {sel(Small, N, HwN), sel(Small, D, HwD)};
    }
    Value *Sin = Quadrant(Q, false), *Cos = Quadrant(Q, true);
    return {sel(Small, Sin, Hw(Intrinsic::sin)),
            sel(Small, Cos, Hw(Intrinsic::cos))};
  }

  /// The half not asked for is left to dead code elimination.
  Value *sinCos(Value *X, bool Cos) {
    auto [S, C] = sinAndCos(X);
    return Cos ? C : S;
  }

};

/// Computes \p Fn in float for values of \p Ty, a floating point type or a
/// vector of one.
template <typename FnT>
static Value *inFloat(IRBuilderBase &B, Type *Ty, ArrayRef<Value *> Args,
                      FnT Fn, bool FlushF32 = false) {
  Type *F32 = Ty->getWithNewType(B.getFloatTy());
  SmallVector<Value *, 2> Ext;
  for (Value *A : Args)
    Ext.push_back(Ty == F32 ? A : B.CreateFPExt(A, F32));
  MathBuilder M(B, F32, FlushF32 || Ty != F32,
                Ty->getScalarType()->isHalfTy(),
                Ty->getScalarType()->isHalfTy() ||
                    Ty->getScalarType()->isBFloatTy());
  Value *R = Fn(M, Ext);
  return Ty == F32 ? R : B.CreateFPTrunc(R, Ty);
}
} // namespace

bool AGX::canExpandMath(Intrinsic::ID ID, Type *Ty, bool Approx) {
  Type *E = Ty->getScalarType();
  switch (ID) {
  case Intrinsic::tan:
  case Intrinsic::asin:
  case Intrinsic::acos:
  case Intrinsic::atan:
  case Intrinsic::atan2:
  case Intrinsic::sinh:
  case Intrinsic::cosh:
  case Intrinsic::tanh:
    return E->isFloatTy() || E->isHalfTy() || E->isBFloatTy();
  case Intrinsic::sin:
  case Intrinsic::cos:
    // bfloat too: promoted to the hardware's reduction (afn's), its sin was
    // up to 256 ulp off even below 16 (tiny results lost) and worse for large
    // arguments (simulator, every bfloat); the float expansion rounded to
    // bfloat is within an ulp below 10^6.
    return (E->isFloatTy() || E->isBFloatTy()) && !Approx;
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::exp10:
  case Intrinsic::log:
  case Intrinsic::log2:
  case Intrinsic::log10:
  case Intrinsic::pow:
  case Intrinsic::sincos:
    // The hardware's are precise enough for half, and are what afn asks.
    return E->isFloatTy() && !Approx;
  default:
    return false;
  }
}

Value *AGX::expandMath(IRBuilderBase &B, Intrinsic::ID ID,
                       ArrayRef<Value *> Args, bool FlushF32,
                       bool ExpNoOverflow, bool NativeNaN,
                       bool ExpResultFlushed) {
  return inFloat(B, Args[0]->getType(), Args,
                 [&](MathBuilder &M, ArrayRef<Value *> A) -> Value * {
                   M.NativeNaN = NativeNaN;
                   M.ExpResultFlushed = ExpResultFlushed;
                   switch (ID) {
                   case Intrinsic::tan:
                     return M.tan(A[0]);
                   case Intrinsic::asin:
                     return M.asin(A[0]);
                   case Intrinsic::acos:
                     return M.acos(A[0]);
                   case Intrinsic::atan:
                     return M.atan(A[0]);
                   case Intrinsic::atan2:
                     return M.atan2(A[0], A[1]);
                   case Intrinsic::sinh:
                     return M.sinh(A[0]);
                   case Intrinsic::cosh:
                     return M.cosh(A[0]);
                   case Intrinsic::tanh:
                     return M.tanh(A[0]);
                   case Intrinsic::sin:
                     return M.sinCos(A[0], false);
                   case Intrinsic::cos:
                     return M.sinCos(A[0], true);
                   case Intrinsic::exp:
                     return M.expPrecise(A[0], ExpNoOverflow);
                   case Intrinsic::exp2:
                     return M.exp2Precise(A[0], ExpNoOverflow);
                   case Intrinsic::exp10:
                     return M.exp10Precise(A[0], ExpNoOverflow);
                   case Intrinsic::log:
                     return M.logPrecise(A[0]);
                   case Intrinsic::log2:
                     return M.log2Precise(A[0]);
                   case Intrinsic::log10:
                     return M.log10Precise(A[0]);
                   case Intrinsic::pow:
                     return M.powPrecise(A[0], A[1]);
                   case Intrinsic::sincos: {
                     auto [S, C] = M.sinAndCos(A[0]);
                     Value *R = PoisonValue::get(StructType::get(
                         S->getType()->getContext(), {S->getType(), C->getType()}));
                     R = B.CreateInsertValue(R, S, 0);
                     return B.CreateInsertValue(R, C, 1);
                   }
                   default:
                     llvm_unreachable("not an expanded math intrinsic");
                   }
                 },
                 FlushF32);
}

Value *AGX::expandMath(IRBuilderBase &B, StringRef Fn, Value *X) {
  using MemberT = Value *(MathBuilder::*)(Value *);
  MemberT Member = StringSwitch<MemberT>(Fn)
                       .Case("asinh", &MathBuilder::asinh)
                       .Case("acosh", &MathBuilder::acosh)
                       .Case("atanh", &MathBuilder::atanh)
                       .Case("sinpi", &MathBuilder::sinpi)
                       .Case("cospi", &MathBuilder::cospi)
                       .Case("tanpi", &MathBuilder::tanpi)
                       .Default(nullptr);
  Type *E = X->getType()->getScalarType();
  if (!Member || !(E->isFloatTy() || E->isHalfTy() || E->isBFloatTy()))
    return nullptr;
  return inFloat(B, X->getType(), {X},
                 [&](MathBuilder &M, ArrayRef<Value *> A) {
                   return (M.*Member)(A[0]);
                 });
}
