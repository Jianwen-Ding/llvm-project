; Directed-rounding sqrt (the lifter: LLVM's constrained sqrt in that mode).
; Alive2 times out on these even at 600 s per query; run explicitly with
; isa_selftest.py --smt-to 600000 slow_sqrt.

; rz for a >= 0: the largest float r with r*r <= a (r = RN result, minus one
; ulp if it rounded up); NaN/negative/inf/zero as RN.
define float @sqrt_rz_f32(float %a) {
  %r = call float @llvm.sqrt.f32(float %a)
  %rd = fpext float %r to double
  %ad = fpext float %a to double
  %sq = fmul double %rd, %rd
  %up = fcmp ogt double %sq, %ad
  %fin = fcmp one float %r, 0x7FF0000000000000
  %pos = fcmp ogt float %r, 0.0
  %adj = and i1 %up, %fin
  %adj2 = and i1 %adj, %pos
  %bits = bitcast float %r to i32
  %b1 = sub i32 %bits, 1
  %rm = bitcast i32 %b1 to float
  %d0 = select i1 %adj2, float %rm, float %r
  ; poison unless the FP environment rounds toward zero, as for the
  ; lifter's constrained operations (an exact multiplication by 1.0)
  %d = call float @llvm.experimental.constrained.fmul.f32(float %d0, float 1.0, metadata !"round.towardzero", metadata !"fpexcept.ignore") strictfp
  ret float %d
}

; rp for f64: the RN result plus one ulp if it rounded down; whether it did
; is the sign of the exact residual r*r - a, which fma computes with one
; rounding (the sign survives rounding; a zero residual is exact here as
; the inputs are kept away from the subnormal range by the precondition).
define double @sqrt_rp_f64(double %a) {
  %r = call double @llvm.sqrt.f64(double %a)
  %na = fneg double %a
  %res = call double @llvm.fma.f64(double %r, double %r, double %na)
  %dn = fcmp olt double %res, 0.0
  %fin = fcmp one double %r, 0x7FF0000000000000
  %ok = fcmp oge double %a, 0x3000000000000000
  %adj0 = and i1 %dn, %fin
  %adj = and i1 %adj0, %ok
  %bits = bitcast double %r to i64
  %b1 = add i64 %bits, 1
  %rp = bitcast i64 %b1 to double
  %d = select i1 %adj, double %rp, double %r
  %small = fcmp olt double %a, 0x3000000000000000
  %pz = fcmp ogt double %a, 0.0
  %tiny = and i1 %small, %pz
  %any = freeze double poison
  %out0 = select i1 %tiny, double %any, double %d
  %out = call double @llvm.experimental.constrained.fmul.f64(double %out0, double 1.0, metadata !"round.upward", metadata !"fpexcept.ignore") strictfp
  ret double %out
}

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
