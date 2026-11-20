; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-contiguous=0 -S < %s | FileCheck %s --check-prefix=OFF

; A thread's own row of a runtime length (L15-E566): its loads start at the
; thread position times a row's bytes. A sum (one load an iteration) is
; unrolled 8 times with a remainder, a dot (two) 4 times. A filter's taps,
; x[i + k] (the thread position times 1), are left (fir_taps lost).

; CHECK-LABEL: @rowsum(
; CHECK-COUNT-8: fadd float
; CHECK: .epil
; OFF-LABEL: @rowsum(
; OFF: fadd float
; OFF-NOT: fadd float
; OFF: ret void
; CHECK-LABEL: @rowdot(
; CHECK-COUNT-4: call float @llvm.fma.f32
; CHECK: .epil
; CHECK-LABEL: @taps(
; CHECK: call float @llvm.fma.f32
; CHECK-NOT: call float @llvm.fma.f32
; CHECK: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.fma.f32(float, float, float)

define void @rowsum(ptr addrspace(1) %x, ptr addrspace(1) %o, i64 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %it = trunc nuw nsw i64 %i to i32
  %k = add i32 %row, %it
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %s1, %loop ]
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @rowdot(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %o, i64 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %it = trunc nuw nsw i64 %i to i32
  %k = add i32 %row, %it
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds float, ptr addrspace(1) %w, i64 %i
  %u = load float, ptr addrspace(1) %q, align 4
  %s1 = call float @llvm.fma.f32(float %v, float %u, float %s)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %s1, %loop ]
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @taps(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %o, i64 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %it = trunc nuw nsw i64 %i to i32
  %k = add i32 %t, %it
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds float, ptr addrspace(1) %w, i64 %i
  %u = load float, ptr addrspace(1) %q, align 4
  %s1 = call float @llvm.fma.f32(float %v, float %u, float %s)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %s1, %loop ]
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}
