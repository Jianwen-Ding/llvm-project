; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime=0 -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-columns-only=0 -S < %s | FileCheck %s --check-prefix=EIGHT

; A device-load loop of a runtime trip count with light work and a load that
; does not step by its own size (a matrix product's column of B, n floats a
; step) is unrolled with a remainder loop (L15-E536): 8 times for a runtime
; stride (L15-E537), 4 for a gather; contiguous loads alone (an FIR's taps)
; are not.

; CHECK-LABEL: @dot_column(
; CHECK-COUNT-8: call float @llvm.fma.f32
; CHECK: epil
; OFF-LABEL: @dot_column(
; OFF-COUNT-1: call float @llvm.fma.f32
; OFF-NOT: call float @llvm.fma.f32
; OFF: store float
define void @dot_column(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %n, i32 %col) {
entry:
  %e = icmp eq i32 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %m, %loop ]
  %pa = getelementptr inbounds float, ptr addrspace(1) %a, i32 %k
  %u = load float, ptr addrspace(1) %pa, align 4
  %kn = mul i32 %k, %n
  %ib = add i32 %kn, %col
  %zb = zext i32 %ib to i64
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %zb
  %v = load float, ptr addrspace(1) %pb, align 4
  %m = call float @llvm.fma.f32(float %u, float %v, float %acc)
  %k1 = add nuw i32 %k, 1
  %cnd = icmp ult i32 %k1, %n
  br i1 %cnd, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %m, %loop ]
  store float %r, ptr addrspace(1) %c, align 4
  ret void
}

; A column alone (every load a runtime stride apart) 16 times (L15-E603).
; CHECK-LABEL: @colsum(
; CHECK-COUNT-16: fadd float
; CHECK-NOT: fadd float
; CHECK: epil
; EIGHT-LABEL: @colsum(
; EIGHT-COUNT-8: fadd float
; EIGHT-NOT: fadd float
; EIGHT: epil
define void @colsum(ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %n, i32 %col) {
entry:
  %e = icmp eq i32 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %m, %loop ]
  %kn = mul i32 %k, %n
  %ib = add i32 %kn, %col
  %zb = zext i32 %ib to i64
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %zb
  %v = load float, ptr addrspace(1) %pb, align 4
  %m = fadd float %acc, %v
  %k1 = add nuw i32 %k, 1
  %cnd = icmp ult i32 %k1, %n
  br i1 %cnd, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %m, %loop ]
  store float %r, ptr addrspace(1) %c, align 4
  ret void
}

; A column's running maximum likewise 16 times (L15-E604).
; CHECK-LABEL: @colmax(
; CHECK-COUNT-16: call float @llvm.maxnum.f32
; CHECK-NOT: call float @llvm.maxnum.f32
; CHECK: epil
; EIGHT-LABEL: @colmax(
; EIGHT-COUNT-8: call float @llvm.maxnum.f32
; EIGHT-NOT: call float @llvm.maxnum.f32
; EIGHT: epil
define void @colmax(ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %n, i32 %col) {
entry:
  %e = icmp eq i32 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %m, %loop ]
  %kn = mul i32 %k, %n
  %ib = add i32 %kn, %col
  %zb = zext i32 %ib to i64
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %zb
  %v = load float, ptr addrspace(1) %pb, align 4
  %m = call float @llvm.maxnum.f32(float %acc, float %v)
  %k1 = add nuw i32 %k, 1
  %cnd = icmp ult i32 %k1, %n
  br i1 %cnd, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %m, %loop ]
  store float %r, ptr addrspace(1) %c, align 4
  ret void
}

; CHECK-LABEL: @taps(
; CHECK-COUNT-1: call float @llvm.fma.f32
; CHECK-NOT: call float @llvm.fma.f32
define void @taps(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y, i32 %n, i32 %i) {
entry:
  %e = icmp eq i32 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %m, %loop ]
  %ik = add i32 %i, %k
  %zx = zext i32 %ik to i64
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %zx
  %xv = load float, ptr addrspace(1) %px, align 4
  %zk = zext i32 %k to i64
  %pw = getelementptr inbounds float, ptr addrspace(1) %w, i64 %zk
  %wv = load float, ptr addrspace(1) %pw, align 4
  %m = call float @llvm.fma.f32(float %xv, float %wv, float %acc)
  %k1 = add nuw i32 %k, 1
  %cnd = icmp ult i32 %k1, %n
  br i1 %cnd, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %m, %loop ]
  store float %r, ptr addrspace(1) %y, align 4
  ret void
}
declare float @llvm.fma.f32(float, float, float)

; CHECK-LABEL: @gather(
; CHECK-COUNT-4: call float @llvm.fma.f32
; CHECK: epil
define void @gather(ptr addrspace(1) %x, ptr addrspace(1) %idx, ptr addrspace(1) %y, i32 %n) {
entry:
  %e = icmp eq i32 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %m, %loop ]
  %zk = zext i32 %k to i64
  %pi = getelementptr inbounds i32, ptr addrspace(1) %idx, i64 %zk
  %j = load i32, ptr addrspace(1) %pi, align 4
  %zj = zext i32 %j to i64
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %zj
  %xv = load float, ptr addrspace(1) %px, align 4
  %m = call float @llvm.fma.f32(float %xv, float %xv, float %acc)
  %k1 = add nuw i32 %k, 1
  %cnd = icmp ult i32 %k1, %n
  br i1 %cnd, label %loop, label %exit
exit:
  %r = phi float [ 0.0, %entry ], [ %m, %loop ]
  store float %r, ptr addrspace(1) %y, align 4
  ret void
}
declare float @llvm.maxnum.f32(float, float)
