; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-store-light=0 -S < %s | FileCheck %s --check-prefix=SIXTEEN

; A row loop of at most two word loads, word stores and light arithmetic
; (L15-E564): 8 iterations, not the store factor's 16, whose loads held 32
; registers. An exponential a step keeps 16.

; CHECK-LABEL: @scale(
; CHECK-COUNT-8: store float
; CHECK-NOT: store float
; CHECK: br i1
; SIXTEEN-LABEL: @scale(
; SIXTEEN-COUNT-16: store float
; CHECK-LABEL: @expo(
; CHECK-COUNT-16: store float
define void @scale(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %pw = getelementptr inbounds float, ptr addrspace(1) %w, i64 %i
  %u = load float, ptr addrspace(1) %pw, align 4
  %o = fmul float %v, %u
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @expo(ptr addrspace(1) %x, ptr addrspace(1) %y) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %o = call float @llvm.exp.f32(float %v)
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
declare float @llvm.exp.f32(float)
