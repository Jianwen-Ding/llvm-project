; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-narrow-stores=0 < %s | FileCheck %s --check-prefix=OFFN
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-narrow-loads=0 < %s | FileCheck %s --check-prefix=OFFL
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=ASM

; A loop that loads from some buffers and stores to another (L15-E527) runs
; as two copies: one where the ranges it reads and writes do not overlap,
; its accesses in disjoint alias scopes, so that its loads need not wait for
; the stores before them, and the loop as it was.

; CHECK-LABEL: define agx_kernel void @norm_apply(
; CHECK: .lver.check:
; CHECK: br i1 %{{.*}}, label %{{.*}}.lver.orig, label %{{.*}}.ph
; CHECK: load float, {{.*}} !alias.scope
; CHECK: store float {{.*}} !alias.scope {{.*}} !noalias
; OFF-LABEL: define agx_kernel void @norm_apply(
; OFF-NOT: lver
; OFF: ret void

; In the copy, the second element's loads need not wait for the first's
; store: the load / store vectorizer merges the two elements' accesses into
; 8-byte ones. The original loads, stores, then loads again.
; ASM-LABEL: norm_apply:
; ASM: // %loop{{$}}
; ASM-COUNT-3: device_load {{[01]}}, i32, xy,
; ASM-NOT: device_load
; ASM: device_store 0, i32, xy,
; ASM: // %loop.lver.orig
; ASM-COUNT-3: device_load {{[01]}}, i32, x,
; ASM-NOT: device_load
; ASM: device_store 0, i32, x,
; ASM-NEXT: device_load 0, i32, x,

; A loop with much arithmetic per word it loads (three exponentials) hides
; the loads' latency behind it: left. One exponential a word is versioned
; (L15-E549: row_exp's cycles 62435 -> 18045).
; CHECK-LABEL: define agx_kernel void @exp_row(
; CHECK-NOT: lver
; CHECK: ret void

; Byte stores of word loads (L15-E546): versioned too; only loads of 1- or
; 2-byte elements make alignment versioning copy the loop again.
; CHECK-LABEL: define agx_kernel void @quant_row(
; CHECK: .lver.check:
; CHECK: store i8 {{.*}} !alias.scope
; OFFN-LABEL: define agx_kernel void @quant_row(
; OFFN-NOT: lver
; OFFN: ret void

; Two-byte loads: versioned too, since alignment versioning leaves the copy
; for overlapping ranges (L15-E550).
; CHECK-LABEL: define agx_kernel void @half_row(
; CHECK: .lver.check:
; CHECK: load half, {{.*}} !alias.scope
; OFFL-LABEL: define agx_kernel void @half_row(
; OFFL-NOT: lver
; OFFL: ret void

; 52 words loaded an iteration, which hoisted would cost registers: left.
; CHECK-LABEL: define agx_kernel void @wide_copy(
; CHECK-NOT: lver
; CHECK: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp.f32(float)
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)

define agx_kernel void @norm_apply(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %z, ptr addrspace(1) %y, float %m) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 512
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j2, %loop ]
  %i = add i32 %row, %j
  %px = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %pw = getelementptr float, ptr addrspace(1) %w, i32 %j
  %wv = load float, ptr addrspace(1) %pw, align 4
  %pz = getelementptr float, ptr addrspace(1) %z, i32 %i
  %zv = load float, ptr addrspace(1) %pz, align 4
  %d = fsub float %v, %m
  %o = call float @llvm.fma.f32(float %d, float %wv, float %zv)
  %py = getelementptr float, ptr addrspace(1) %y, i32 %i
  store float %o, ptr addrspace(1) %py, align 4
  %j1 = add nuw nsw i32 %j, 1
  %i1 = add i32 %row, %j1
  %px1 = getelementptr float, ptr addrspace(1) %x, i32 %i1
  %v1 = load float, ptr addrspace(1) %px1, align 4
  %pw1 = getelementptr float, ptr addrspace(1) %w, i32 %j1
  %wv1 = load float, ptr addrspace(1) %pw1, align 4
  %pz1 = getelementptr float, ptr addrspace(1) %z, i32 %i1
  %zv1 = load float, ptr addrspace(1) %pz1, align 4
  %d1 = fsub float %v1, %m
  %o1 = call float @llvm.fma.f32(float %d1, float %wv1, float %zv1)
  %py1 = getelementptr float, ptr addrspace(1) %y, i32 %i1
  store float %o1, ptr addrspace(1) %py1, align 4
  %j2 = add nuw nsw i32 %j, 2
  %c = icmp ult i32 %j2, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define agx_kernel void @exp_row(ptr addrspace(1) %x, ptr addrspace(1) %y, float %m) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 512
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %i = add i32 %row, %j
  %px = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %d = fsub float %v, %m
  %e = call float @llvm.exp.f32(float %d)
  %e1 = call float @llvm.exp.f32(float %e)
  %e2 = call float @llvm.exp.f32(float %e1)
  %py = getelementptr float, ptr addrspace(1) %y, i32 %i
  store float %e2, ptr addrspace(1) %py, align 4
  %j1 = add nuw nsw i32 %j, 1
  %c = icmp ult i32 %j1, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define agx_kernel void @quant_row(ptr addrspace(1) %x, ptr addrspace(1) %q, float %s) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %i = add i32 %row, %j
  %px = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %m = fmul float %v, %s
  %c1 = call float @llvm.maximum.f32(float %m, float -127.0)
  %c2 = call float @llvm.minimum.f32(float %c1, float 127.0)
  %qi = fptosi float %c2 to i8
  %pq = getelementptr i8, ptr addrspace(1) %q, i32 %i
  store i8 %qi, ptr addrspace(1) %pq, align 1
  %j1 = add nuw nsw i32 %j, 1
  %c = icmp ult i32 %j1, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define agx_kernel void @half_row(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 512
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %i = add i32 %row, %j
  %px = getelementptr half, ptr addrspace(1) %x, i32 %i
  %v = load half, ptr addrspace(1) %px, align 2
  %pw = getelementptr half, ptr addrspace(1) %w, i32 %j
  %wv = load half, ptr addrspace(1) %pw, align 2
  %o = fmul half %v, %wv
  %py = getelementptr half, ptr addrspace(1) %y, i32 %i
  store half %o, ptr addrspace(1) %py, align 2
  %j1 = add nuw nsw i32 %j, 1
  %c = icmp ult i32 %j1, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define agx_kernel void @wide_copy(ptr addrspace(1) %x, ptr addrspace(1) %y) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 8192
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %i = add i32 %row, %j
  %p0 = getelementptr <4 x float>, ptr addrspace(1) %x, i32 %i
  %q0 = getelementptr <4 x float>, ptr addrspace(1) %y, i32 %i
  %v0 = load <4 x float>, ptr addrspace(1) %p0, align 16
  %p1 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 1
  %v1 = load <4 x float>, ptr addrspace(1) %p1, align 16
  %p2 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 2
  %v2 = load <4 x float>, ptr addrspace(1) %p2, align 16
  %p3 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 3
  %v3 = load <4 x float>, ptr addrspace(1) %p3, align 16
  %p4 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 4
  %v4 = load <4 x float>, ptr addrspace(1) %p4, align 16
  %p5 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 5
  %v5 = load <4 x float>, ptr addrspace(1) %p5, align 16
  %p6 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 6
  %v6 = load <4 x float>, ptr addrspace(1) %p6, align 16
  %p7 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 7
  %v7 = load <4 x float>, ptr addrspace(1) %p7, align 16
  %p8 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 8
  %v8 = load <4 x float>, ptr addrspace(1) %p8, align 16
  %p9 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 9
  %v9 = load <4 x float>, ptr addrspace(1) %p9, align 16
  %p10 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 10
  %v10 = load <4 x float>, ptr addrspace(1) %p10, align 16
  %p11 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 11
  %v11 = load <4 x float>, ptr addrspace(1) %p11, align 16
  %p12 = getelementptr <4 x float>, ptr addrspace(1) %p0, i32 12
  %v12 = load <4 x float>, ptr addrspace(1) %p12, align 16
  store <4 x float> %v0, ptr addrspace(1) %q0, align 16
  %q1 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 1
  store <4 x float> %v1, ptr addrspace(1) %q1, align 16
  %q2 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 2
  store <4 x float> %v2, ptr addrspace(1) %q2, align 16
  %q3 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 3
  store <4 x float> %v3, ptr addrspace(1) %q3, align 16
  %q4 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 4
  store <4 x float> %v4, ptr addrspace(1) %q4, align 16
  %q5 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 5
  store <4 x float> %v5, ptr addrspace(1) %q5, align 16
  %q6 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 6
  store <4 x float> %v6, ptr addrspace(1) %q6, align 16
  %q7 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 7
  store <4 x float> %v7, ptr addrspace(1) %q7, align 16
  %q8 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 8
  store <4 x float> %v8, ptr addrspace(1) %q8, align 16
  %q9 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 9
  store <4 x float> %v9, ptr addrspace(1) %q9, align 16
  %q10 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 10
  store <4 x float> %v10, ptr addrspace(1) %q10, align 16
  %q11 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 11
  store <4 x float> %v11, ptr addrspace(1) %q11, align 16
  %q12 = getelementptr <4 x float>, ptr addrspace(1) %q0, i32 12
  store <4 x float> %v12, ptr addrspace(1) %q12, align 16
  %j1 = add nuw nsw i32 %j, 13
  %c = icmp ult i32 %j1, 8190
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
