; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-contiguous=0 -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-exp=0 -agx-device-unroll-runtime-heavy-factor=1 -S < %s | FileCheck %s --check-prefix=NOEXP
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-max=0 -S < %s | FileCheck %s --check-prefix=NOMAX
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-narrow-rows=0 -S < %s | FileCheck %s --check-prefix=WIDE
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-heavy-max=0 -S < %s | FileCheck %s --check-prefix=LIGHTMAX
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-heavy-max-signed=0 -S < %s | FileCheck %s --check-prefix=SIGNED

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
; A thread's row of exponentials (a softmax's sum) likewise (L15-E568): 8
; times, not the generic runtime unroll's 4.
; CHECK-LABEL: @expsum(
; CHECK-COUNT-8: call float @llvm.exp.f32
; CHECK: .epil
; NOEXP-LABEL: @expsum(
; NOEXP-COUNT-4: call float @llvm.exp.f32
; NOEXP-NOT: call float @llvm.exp.f32
; NOEXP: .epil
; A thread's row maximum likewise (L15-E582): 8 times, not the store rows'
; 4 (E565).
; CHECK-LABEL: @rowmax(
; CHECK-COUNT-8: call float @llvm.maxnum.f32
; CHECK: .epil
; NOMAX-LABEL: @rowmax(
; NOMAX-COUNT-4: call float @llvm.maxnum.f32
; NOMAX-NOT: call float @llvm.maxnum.f32
; NOMAX: .epil
; A maximum that propagates NaNs twice that, 16 times (L15-E686).
; CHECK-LABEL: @rowmaximum(
; CHECK-COUNT-16: call float @llvm.maximum.f32
; CHECK-NOT: call float @llvm.maximum.f32
; CHECK: .epil
; LIGHTMAX-LABEL: @rowmaximum(
; LIGHTMAX-COUNT-8: call float @llvm.maximum.f32
; LIGHTMAX-NOT: call float @llvm.maximum.f32
; LIGHTMAX: .epil
; Not one whose zero sign nothing reads, a softmax's maximum that only
; offsets its exponentials: one instruction, as maxnum's, 8 times.
; CHECK-LABEL: @rowmaximumexp(
; CHECK-COUNT-8: call float @llvm.maximum.f32
; CHECK-NOT: call float @llvm.maximum.f32
; CHECK: .epil
; SIGNED-LABEL: @rowmaximumexp(
; SIGNED-COUNT-16: call float @llvm.maximum.f32
; SIGNED-NOT: call float @llvm.maximum.f32
; SIGNED: .epil
; Rows of halves or bytes as many bytes an iteration as of floats
; (L15-E583): a half row's sum 16 times, a byte row's 32.
; CHECK-LABEL: @hsum(
; CHECK-COUNT-16: fpext half
; CHECK-NOT: fpext half
; CHECK: .epil
; WIDE-LABEL: @hsum(
; WIDE-COUNT-8: fpext half
; WIDE-NOT: fpext half
; WIDE: .epil
; CHECK-LABEL: @bsum(
; CHECK-COUNT-32: sext i8
; CHECK-NOT: sext i8
; CHECK: .epil
; CHECK-LABEL: @taps(
; CHECK: call float @llvm.fma.f32
; CHECK-NOT: call float @llvm.fma.f32
; CHECK: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.fma.f32(float, float, float)
declare float @llvm.exp.f32(float)
declare float @llvm.maxnum.f32(float, float)

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

define void @expsum(ptr addrspace(1) %x, ptr addrspace(1) %o, i64 %n, float %m) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 6
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
  %d = fsub float %v, %m
  %ev = call float @llvm.exp.f32(float %d)
  %s1 = fadd float %s, %ev
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %s1, %loop ]
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @rowmax(ptr addrspace(1) %x, ptr addrspace(1) %o, i64 %n) {
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
  %s1 = call float @llvm.maxnum.f32(float %s, float %v)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %s1, %loop ]
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @rowmaximum(ptr addrspace(1) %x, ptr addrspace(1) %o, i64 %n) {
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
  %s1 = call float @llvm.maximum.f32(float %s, float %v)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %s1, %loop ]
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @rowmaximumexp(ptr addrspace(1) %x, ptr addrspace(1) %o, i64 %n, float %w) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0xFFF0000000000000, %entry ], [ %s1, %loop ]
  %it = trunc nuw nsw i64 %i to i32
  %k = add i32 %row, %it
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %s1 = call float @llvm.maximum.f32(float %s, float %v)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0xFFF0000000000000, %entry ], [ %s1, %loop ]
  %d = fsub float %w, %r
  %ex = call float @llvm.exp.f32(float %d)
  store float %ex, ptr addrspace(1) %o, align 4
  ret void
}

define void @hsum(ptr addrspace(1) %x, ptr addrspace(1) %o, i64 %n) {
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
  %p = getelementptr inbounds half, ptr addrspace(1) %x, i64 %z
  %h = load half, ptr addrspace(1) %p, align 2
  %v = fpext half %h to float
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %s1, %loop ]
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @bsum(ptr addrspace(1) %x, ptr addrspace(1) %o, i64 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi i32 [ 0, %entry ], [ %s1, %loop ]
  %it = trunc nuw nsw i64 %i to i32
  %k = add i32 %row, %it
  %z = zext i32 %k to i64
  %p = getelementptr inbounds i8, ptr addrspace(1) %x, i64 %z
  %h = load i8, ptr addrspace(1) %p, align 1
  %v = sext i8 %h to i32
  %s1 = add i32 %s, %v
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi i32 [ 0, %entry ], [ %s1, %loop ]
  store i32 %r, ptr addrspace(1) %o, align 4
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

declare float @llvm.maximum.f32(float, float)
