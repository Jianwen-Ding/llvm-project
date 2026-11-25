; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-diff-checks=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-diff-thread=0 < %s | FileCheck %s --check-prefix=NOTHREAD
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-diff-runtime=0 < %s | FileCheck %s --check-prefix=NORT

; A versioned loop's ranges [a, a + m) and [b, b + n) overlap where
; -n < b - a < m: one compare of b - a + n - 1 with m + n - 1 (L15-E636)
; instead of two of four ends. Where both ranges start at the thread's own
; row, b - a is the buffers' difference alone, the same for every thread.

; Rows of 32 floats of x and z into y: 127 + x - y and 127 + z - y below 255.
; CHECK-LABEL: define agx_kernel void @rows(
; CHECK: loop.lver.check:
; CHECK-DAG: %[[Z:.*]] = ptrtoaddr ptr addrspace(1) %z to i64
; CHECK-DAG: %[[Y:.*]] = ptrtoaddr ptr addrspace(1) %y to i64
; CHECK-DAG: %[[X:.*]] = ptrtoaddr ptr addrspace(1) %x to i64
; CHECK: %[[XA:.*]] = add i64 %[[X]], 127
; CHECK: %[[XD:.*]] = sub i64 %[[XA]], %[[Y]]
; CHECK: %agx.alias.diff = icmp ult i64 %[[XD]], 255
; CHECK: %[[ZA:.*]] = add i64 %[[Z]], 127
; CHECK: %[[ZD:.*]] = sub i64 %[[ZA]], %[[Y]]
; CHECK: %[[ZC:.*]] = icmp ult i64 %[[ZD]], 255
; CHECK: %agx.alias.conflict = or i1 %agx.alias.diff, %[[ZC]]
; CHECK-NEXT: br i1 %agx.alias.conflict, label %loop.ph.lver.orig, label %loop.ph
; OFF-LABEL: define agx_kernel void @rows(
; OFF: %bound0 = icmp ult ptr addrspace(1)
; OFF: %conflict.rdx = or i1
; OFF-NOT: agx.alias.diff

; A row of x times a vector w that every row shares: w - y less the row's
; own offset 4 * 32 * r (L15-E637).
; CHECK-LABEL: define agx_kernel void @rows_weights(
; CHECK: %agx.alias.diff = icmp ult i64 %{{.*}}, 255
; CHECK: %[[ROW:.*]] = sext i32 %row to i64
; CHECK: %[[SH:.*]] = shl nsw i64 %[[ROW]], 2
; CHECK: %[[WD:.*]] = sub i64 %{{.*}}, %[[SH]]
; CHECK: icmp ult i64 %[[WD]], 255
; CHECK: br i1 %agx.alias.conflict,
; NOTHREAD-LABEL: define agx_kernel void @rows_weights(
; NOTHREAD: %bound0 = icmp ult ptr addrspace(1)
; NOTHREAD-NOT: agx.alias.diff
; NOTHREAD: %conflict.rdx = or i1

; A row of a runtime length n2: m = n = 8 k + 8 bytes, m + n - 1 computed.
; CHECK-LABEL: define agx_kernel void @rows_dyn(
; CHECK: loop.lver.check:
; CHECK: %[[D:.*]] = sub i64 %{{.*}}, %y
; CHECK: %[[W:.*]] = add nuw nsw i64 %{{.*}}, 15
; CHECK: %agx.alias.diff = icmp ult i64 %[[D]], %[[W]]
; CHECK-NEXT: br i1 %agx.alias.diff, label %loop.ph.lver.orig, label %loop.ph
; NORT-LABEL: define agx_kernel void @rows_dyn(
; NORT: %bound0 = icmp ult ptr addrspace(1)
; NORT-NOT: agx.alias.diff
; NORT: br i1 %found.conflict,

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp.f32(float)

define agx_kernel void @rows(ptr addrspace(1) %x, ptr addrspace(1) %z, ptr addrspace(1) %y) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 32
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j2, %loop ]
  %i = add i32 %row, %j
  %px = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %pz = getelementptr float, ptr addrspace(1) %z, i32 %i
  %zv = load float, ptr addrspace(1) %pz, align 4
  %o = call float @llvm.fma.f32(float %v, float 2.0, float %zv)
  %py = getelementptr float, ptr addrspace(1) %y, i32 %i
  store float %o, ptr addrspace(1) %py, align 4
  %j1 = add nuw nsw i32 %j, 1
  %i1 = add i32 %row, %j1
  %px1 = getelementptr float, ptr addrspace(1) %x, i32 %i1
  %v1 = load float, ptr addrspace(1) %px1, align 4
  %pz1 = getelementptr float, ptr addrspace(1) %z, i32 %i1
  %zv1 = load float, ptr addrspace(1) %pz1, align 4
  %o1 = call float @llvm.fma.f32(float %v1, float 2.0, float %zv1)
  %py1 = getelementptr float, ptr addrspace(1) %y, i32 %i1
  store float %o1, ptr addrspace(1) %py1, align 4
  %j2 = add nuw nsw i32 %j, 2
  %c = icmp ult i32 %j2, 32
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define agx_kernel void @rows_weights(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 32
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j2, %loop ]
  %i = add i32 %row, %j
  %px = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %pw = getelementptr float, ptr addrspace(1) %w, i32 %j
  %wv = load float, ptr addrspace(1) %pw, align 4
  %o = fmul float %v, %wv
  %py = getelementptr float, ptr addrspace(1) %y, i32 %i
  store float %o, ptr addrspace(1) %py, align 4
  %j1 = add nuw nsw i32 %j, 1
  %i1 = add i32 %row, %j1
  %px1 = getelementptr float, ptr addrspace(1) %x, i32 %i1
  %v1 = load float, ptr addrspace(1) %px1, align 4
  %pw1 = getelementptr float, ptr addrspace(1) %w, i32 %j1
  %wv1 = load float, ptr addrspace(1) %pw1, align 4
  %o1 = fmul float %v1, %wv1
  %py1 = getelementptr float, ptr addrspace(1) %y, i32 %i1
  store float %o1, ptr addrspace(1) %py1, align 4
  %j2 = add nuw nsw i32 %j, 2
  %c = icmp ult i32 %j2, 32
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define agx_kernel void @rows_dyn(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %h = lshr i32 %n, 1
  %n2 = shl i32 %h, 1
  %row = mul i32 %r, %n2
  %e = icmp eq i32 %h, 0
  br i1 %e, label %exit, label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j2, %loop ]
  %i = add nuw i32 %row, %j
  %ix = zext i32 %i to i64
  %px = getelementptr float, ptr addrspace(1) %x, i64 %ix
  %v = load float, ptr addrspace(1) %px, align 4
  %o = fmul float %v, 3.0
  %py = getelementptr float, ptr addrspace(1) %y, i64 %ix
  store float %o, ptr addrspace(1) %py, align 4
  %j1 = add nuw i32 %j, 1
  %i1 = add nuw i32 %row, %j1
  %ix1 = zext i32 %i1 to i64
  %px1 = getelementptr float, ptr addrspace(1) %x, i64 %ix1
  %v1 = load float, ptr addrspace(1) %px1, align 4
  %o1 = fmul float %v1, 3.0
  %py1 = getelementptr float, ptr addrspace(1) %y, i64 %ix1
  store float %o1, ptr addrspace(1) %py1, align 4
  %j2 = add nuw i32 %j, 2
  %c = icmp ult i32 %j2, %n2
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
