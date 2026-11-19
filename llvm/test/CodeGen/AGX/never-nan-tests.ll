; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-never-nan-tests=0 -stop-after=agx-codegenprepare < %s | FileCheck %s --check-prefix=OFF

; The PTX front end's max.f32: maximumnum, then the default NaN where it is
; one. A running maximum from a number is never a NaN (maximumnum gives one
; only where both operands are), through its loop's phi, so the test and its
; select go (L15-E388). @start_arg starts from an argument that may be a
; NaN and @two_loads takes the maximum of two loads, each possibly a NaN:
; both keep it.

declare float @llvm.maximumnum.f32(float, float)

; CHECK-LABEL: @running(
; CHECK-NOT:   fcmp uno
; CHECK:       ret void
; OFF-LABEL:   @running(
; OFF:         fcmp uno
define void @running(ptr addrspace(1) %x, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %m = phi float [ -3.000000e+38, %entry ], [ %m.next, %loop ]
  %p = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %a = call float @llvm.maximumnum.f32(float %m, float %v)
  %u = fcmp uno float %a, 0.000000e+00
  %m.next = select i1 %u, float 0x7FFFFFFFE0000000, float %a
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  store float %m.next, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: @start_arg(
; CHECK:       fcmp uno
define void @start_arg(ptr addrspace(1) %x, ptr addrspace(1) %out, i32 %n, float %s) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %m = phi float [ %s, %entry ], [ %m.next, %loop ]
  %p = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %a = call float @llvm.maximumnum.f32(float %m, float %v)
  %u = fcmp uno float %a, 0.000000e+00
  %m.next = select i1 %u, float 0x7FFFFFFFE0000000, float %a
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  store float %m.next, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: @two_loads(
; CHECK:       fcmp uno
define void @two_loads(ptr addrspace(1) %x, ptr addrspace(1) %out) {
  %v = load float, ptr addrspace(1) %x, align 4
  %q = getelementptr float, ptr addrspace(1) %x, i32 1
  %w = load float, ptr addrspace(1) %q, align 4
  %a = call float @llvm.maximumnum.f32(float %w, float %v)
  %u = fcmp uno float %a, 0.000000e+00
  %r = select i1 %u, float 0x7FFFFFFFE0000000, float %a
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

; InstCombine's form of an unrolled chain: maximumnum of the canonicalized
; value, select(uno a, b, maximumnum(a, b)) (agx-bench attn_decode).
; CHECK-LABEL: @chain(
; CHECK-NOT:   fcmp uno
; CHECK:       ret void
define void @chain(ptr addrspace(1) %x, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %m = phi float [ -3.000000e+38, %entry ], [ %m.next, %loop ]
  %p = getelementptr <2 x float>, ptr addrspace(1) %x, i32 %i
  %v2 = load <2 x float>, ptr addrspace(1) %p, align 8
  %v = extractelement <2 x float> %v2, i64 0
  %w = extractelement <2 x float> %v2, i64 1
  %a = call float @llvm.maximumnum.f32(float %m, float %v)
  %u = fcmp uno float %a, 0.000000e+00
  %b = call float @llvm.maximumnum.f32(float %a, float %w)
  %s = select i1 %u, float %w, float %b
  %u2 = fcmp uno float %s, 0.000000e+00
  %m.next = select i1 %u2, float 0x7FFFFFFFE0000000, float %s
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  store float %m.next, ptr addrspace(1) %out, align 4
  ret void
}
