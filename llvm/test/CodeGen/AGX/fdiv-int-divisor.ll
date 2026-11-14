; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-fdiv-int-divisor=0 < %s | FileCheck %s --check-prefix=OFF

; A float division by an integer converted to float (a mean's sum / n): the
; divisor is +-0 or of magnitude in [1, 2^64], so its scale 2^(128 - eb) is
; applied first and the dividend's exponent after, with no test of the
; dividend's size (2^-124, 0x1800000) and its selects (L15-E343).

; CHECK-LABEL: mean:
; CHECK: rcp
; CHECK-NOT: 25165824
; CHECK: stop
; OFF-LABEL: mean:
; OFF: mov_imm {{r[0-9]+}}, 25165824
define void @mean(ptr addrspace(1) %p, float %s, i32 %n) {
  %f = sitofp i32 %n to float
  %r = fdiv float %s, %f
  store float %r, ptr addrspace(1) %p
  ret void
}
