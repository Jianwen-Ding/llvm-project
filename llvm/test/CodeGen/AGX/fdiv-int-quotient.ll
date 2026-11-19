; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
;
; A float division of converted 32-bit integers (or their sums): Markstein's
; quotient step on the correctly rounded reciprocal, no exponent scaling, and
; a * rcp(b) where b is 0 (L15-E465).

; CHECK-LABEL: ratio:
; CHECK-DAG:   convert s32_to_f, [[B:r[0-9]+]], u3, rte
; CHECK-DAG:   convert s32_to_f, [[A:r[0-9]+]], u2, rte
; CHECK:       rcp [[Y0:r[0-9]+]], [[B]]
; CHECK-NOT:   bfi
; CHECK:       fmul32 [[Z:r[0-9]+]], [[A]], [[Y0]]
; CHECK-NOT:   bfi
; CHECK:       fcmpsel eq, [[R:r[0-9]+]], [[B]], 0.0, [[Z]],
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @ratio(ptr addrspace(1) %p, i32 %i, i32 %n) {
  %a = sitofp i32 %i to float
  %b = sitofp i32 %n to float
  %q = fdiv float %a, %b
  store float %q, ptr addrspace(1) %p
  ret void
}
