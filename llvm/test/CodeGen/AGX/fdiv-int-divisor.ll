; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-fdiv-int-divisor=0 -agx-fdiv-bounded-operands=0 < %s | FileCheck %s --check-prefix=OFF

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

; More generally (L15-E344): a divisor bounded by how it is computed
; (1 + sqrt(v) is below 2^65) and a dividend that is +-0 or not below 2^-124
; (a half extended): no size test either.
; CHECK-LABEL: by_sqrt:
; CHECK-NOT: 25165824
; CHECK: stop
; OFF-LABEL: by_sqrt:
; OFF: 25165824
define void @by_sqrt(ptr addrspace(1) %p, float %a, float %v) {
  %s = call float @llvm.sqrt.f32(float %v)
  %d = fadd float %s, 1.0
  %r = fdiv float %a, %d
  store float %r, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: half_num:
; CHECK-NOT: 25165824
; CHECK: stop
define void @half_num(ptr addrspace(1) %p, half %h, float %b) {
  %a = fpext half %h to float
  %r = fdiv float %a, %b
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.sqrt.f32(float)
