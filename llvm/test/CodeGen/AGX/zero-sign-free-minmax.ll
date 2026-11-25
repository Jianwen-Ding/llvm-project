; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=ASM

; A float min / max is lowered with a fix ordering -0 below +0 (an and of
; the operands' bits, its flush and a select on their equality). Where
; nothing its value reaches depends on the sign of a zero (here a running
; maximum that only offsets an exponential, e^(x - m), as in attn_decode)
; it is nsz and lowered as one fcmpsel (L15-E389). @stored stores the
; maximum and @divided divides by it (1 / +-0 = +-inf): both keep the fix.

declare float @llvm.maximumnum.f32(float, float)
declare float @llvm.exp2.f32(float)

; CHECK-LABEL: @offset(
; CHECK:       call nsz float @llvm.maximumnum.f32
; ASM-LABEL:   offset:
; ASM-NOT:     and
; ASM:         stop
define void @offset(ptr addrspace(1) %x, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %m = phi float [ -3.000000e+38, %entry ], [ %m.next, %loop ]
  %p = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = call float @llvm.maximumnum.f32(float %m, float %v)
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  %w = load float, ptr addrspace(1) %x, align 4
  %d = fsub float %w, %m.next
  %s = fmul float %d, 0x3FF7154760000000
  %e = call float @llvm.exp2.f32(float %s)
  store float %e, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: @stored(
; CHECK:       call float @llvm.maximumnum.f32
define void @stored(ptr addrspace(1) %x, ptr addrspace(1) %out) {
  %v = load float, ptr addrspace(1) %x, align 4
  %q = getelementptr float, ptr addrspace(1) %x, i32 1
  %w = load float, ptr addrspace(1) %q, align 4
  %m = call float @llvm.maximumnum.f32(float %v, float %w)
  store float %m, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: @divided(
; CHECK:       call float @llvm.maximumnum.f32
define void @divided(ptr addrspace(1) %x, ptr addrspace(1) %out) {
  %v = load float, ptr addrspace(1) %x, align 4
  %q = getelementptr float, ptr addrspace(1) %x, i32 1
  %w = load float, ptr addrspace(1) %q, align 4
  %m = call float @llvm.maximumnum.f32(float %v, float %w)
  %s = fsub float %v, %m
  %r = fdiv float 1.000000e+00, %s
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
