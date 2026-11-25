; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-narrow-stores=0 < %s | FileCheck %s --check-prefix=OFFN
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-narrow-loads=0 < %s | FileCheck %s --check-prefix=OFFL
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-word-bytes=0 < %s | FileCheck %s --check-prefix=OFFW
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

; A loop with much arithmetic per word it loads (five exponentials) hides
; the loads' latency behind it: left. One exponential a word is versioned
; (L15-E549: row_exp's cycles 62435 -> 18045), two (sin and cos, log and
; pow) too since L15-E580.
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

; 52 2-byte loads an iteration: 26 words, counted from their bytes together
; (L15-E554; each a word, 52, it was left).
; CHECK-LABEL: define agx_kernel void @i16_wide(
; CHECK: .lver.check:
; OFFW-LABEL: define agx_kernel void @i16_wide(
; OFFW-NOT: lver
; OFFW: ret void

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
  %e3 = call float @llvm.exp.f32(float %e2)
  %e4 = call float @llvm.exp.f32(float %e3)
  %py = getelementptr float, ptr addrspace(1) %y, i32 %i
  store float %e4, ptr addrspace(1) %py, align 4
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

define agx_kernel void @i16_wide(ptr addrspace(1) %x, ptr addrspace(1) %b, ptr addrspace(1) %y) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 8192
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %jn, %loop ]
  %i0 = add i32 %row, %j
  %px0 = getelementptr i16, ptr addrspace(1) %x, i32 %i0
  %v0 = load i16, ptr addrspace(1) %px0, align 2
  %pb0 = getelementptr i16, ptr addrspace(1) %b, i32 %j
  %w0 = load i16, ptr addrspace(1) %pb0, align 2
  %s0 = add i16 %v0, %w0
  %py0 = getelementptr i16, ptr addrspace(1) %y, i32 %i0
  store i16 %s0, ptr addrspace(1) %py0, align 2
  %i1 = add i32 %i0, 1
  %j1 = add i32 %j, 1
  %px1 = getelementptr i16, ptr addrspace(1) %x, i32 %i1
  %v1 = load i16, ptr addrspace(1) %px1, align 2
  %pb1 = getelementptr i16, ptr addrspace(1) %b, i32 %j1
  %w1 = load i16, ptr addrspace(1) %pb1, align 2
  %s1 = add i16 %v1, %w1
  %py1 = getelementptr i16, ptr addrspace(1) %y, i32 %i1
  store i16 %s1, ptr addrspace(1) %py1, align 2
  %i2 = add i32 %i0, 2
  %j2 = add i32 %j, 2
  %px2 = getelementptr i16, ptr addrspace(1) %x, i32 %i2
  %v2 = load i16, ptr addrspace(1) %px2, align 2
  %pb2 = getelementptr i16, ptr addrspace(1) %b, i32 %j2
  %w2 = load i16, ptr addrspace(1) %pb2, align 2
  %s2 = add i16 %v2, %w2
  %py2 = getelementptr i16, ptr addrspace(1) %y, i32 %i2
  store i16 %s2, ptr addrspace(1) %py2, align 2
  %i3 = add i32 %i0, 3
  %j3 = add i32 %j, 3
  %px3 = getelementptr i16, ptr addrspace(1) %x, i32 %i3
  %v3 = load i16, ptr addrspace(1) %px3, align 2
  %pb3 = getelementptr i16, ptr addrspace(1) %b, i32 %j3
  %w3 = load i16, ptr addrspace(1) %pb3, align 2
  %s3 = add i16 %v3, %w3
  %py3 = getelementptr i16, ptr addrspace(1) %y, i32 %i3
  store i16 %s3, ptr addrspace(1) %py3, align 2
  %i4 = add i32 %i0, 4
  %j4 = add i32 %j, 4
  %px4 = getelementptr i16, ptr addrspace(1) %x, i32 %i4
  %v4 = load i16, ptr addrspace(1) %px4, align 2
  %pb4 = getelementptr i16, ptr addrspace(1) %b, i32 %j4
  %w4 = load i16, ptr addrspace(1) %pb4, align 2
  %s4 = add i16 %v4, %w4
  %py4 = getelementptr i16, ptr addrspace(1) %y, i32 %i4
  store i16 %s4, ptr addrspace(1) %py4, align 2
  %i5 = add i32 %i0, 5
  %j5 = add i32 %j, 5
  %px5 = getelementptr i16, ptr addrspace(1) %x, i32 %i5
  %v5 = load i16, ptr addrspace(1) %px5, align 2
  %pb5 = getelementptr i16, ptr addrspace(1) %b, i32 %j5
  %w5 = load i16, ptr addrspace(1) %pb5, align 2
  %s5 = add i16 %v5, %w5
  %py5 = getelementptr i16, ptr addrspace(1) %y, i32 %i5
  store i16 %s5, ptr addrspace(1) %py5, align 2
  %i6 = add i32 %i0, 6
  %j6 = add i32 %j, 6
  %px6 = getelementptr i16, ptr addrspace(1) %x, i32 %i6
  %v6 = load i16, ptr addrspace(1) %px6, align 2
  %pb6 = getelementptr i16, ptr addrspace(1) %b, i32 %j6
  %w6 = load i16, ptr addrspace(1) %pb6, align 2
  %s6 = add i16 %v6, %w6
  %py6 = getelementptr i16, ptr addrspace(1) %y, i32 %i6
  store i16 %s6, ptr addrspace(1) %py6, align 2
  %i7 = add i32 %i0, 7
  %j7 = add i32 %j, 7
  %px7 = getelementptr i16, ptr addrspace(1) %x, i32 %i7
  %v7 = load i16, ptr addrspace(1) %px7, align 2
  %pb7 = getelementptr i16, ptr addrspace(1) %b, i32 %j7
  %w7 = load i16, ptr addrspace(1) %pb7, align 2
  %s7 = add i16 %v7, %w7
  %py7 = getelementptr i16, ptr addrspace(1) %y, i32 %i7
  store i16 %s7, ptr addrspace(1) %py7, align 2
  %i8 = add i32 %i0, 8
  %j8 = add i32 %j, 8
  %px8 = getelementptr i16, ptr addrspace(1) %x, i32 %i8
  %v8 = load i16, ptr addrspace(1) %px8, align 2
  %pb8 = getelementptr i16, ptr addrspace(1) %b, i32 %j8
  %w8 = load i16, ptr addrspace(1) %pb8, align 2
  %s8 = add i16 %v8, %w8
  %py8 = getelementptr i16, ptr addrspace(1) %y, i32 %i8
  store i16 %s8, ptr addrspace(1) %py8, align 2
  %i9 = add i32 %i0, 9
  %j9 = add i32 %j, 9
  %px9 = getelementptr i16, ptr addrspace(1) %x, i32 %i9
  %v9 = load i16, ptr addrspace(1) %px9, align 2
  %pb9 = getelementptr i16, ptr addrspace(1) %b, i32 %j9
  %w9 = load i16, ptr addrspace(1) %pb9, align 2
  %s9 = add i16 %v9, %w9
  %py9 = getelementptr i16, ptr addrspace(1) %y, i32 %i9
  store i16 %s9, ptr addrspace(1) %py9, align 2
  %i10 = add i32 %i0, 10
  %j10 = add i32 %j, 10
  %px10 = getelementptr i16, ptr addrspace(1) %x, i32 %i10
  %v10 = load i16, ptr addrspace(1) %px10, align 2
  %pb10 = getelementptr i16, ptr addrspace(1) %b, i32 %j10
  %w10 = load i16, ptr addrspace(1) %pb10, align 2
  %s10 = add i16 %v10, %w10
  %py10 = getelementptr i16, ptr addrspace(1) %y, i32 %i10
  store i16 %s10, ptr addrspace(1) %py10, align 2
  %i11 = add i32 %i0, 11
  %j11 = add i32 %j, 11
  %px11 = getelementptr i16, ptr addrspace(1) %x, i32 %i11
  %v11 = load i16, ptr addrspace(1) %px11, align 2
  %pb11 = getelementptr i16, ptr addrspace(1) %b, i32 %j11
  %w11 = load i16, ptr addrspace(1) %pb11, align 2
  %s11 = add i16 %v11, %w11
  %py11 = getelementptr i16, ptr addrspace(1) %y, i32 %i11
  store i16 %s11, ptr addrspace(1) %py11, align 2
  %i12 = add i32 %i0, 12
  %j12 = add i32 %j, 12
  %px12 = getelementptr i16, ptr addrspace(1) %x, i32 %i12
  %v12 = load i16, ptr addrspace(1) %px12, align 2
  %pb12 = getelementptr i16, ptr addrspace(1) %b, i32 %j12
  %w12 = load i16, ptr addrspace(1) %pb12, align 2
  %s12 = add i16 %v12, %w12
  %py12 = getelementptr i16, ptr addrspace(1) %y, i32 %i12
  store i16 %s12, ptr addrspace(1) %py12, align 2
  %i13 = add i32 %i0, 13
  %j13 = add i32 %j, 13
  %px13 = getelementptr i16, ptr addrspace(1) %x, i32 %i13
  %v13 = load i16, ptr addrspace(1) %px13, align 2
  %pb13 = getelementptr i16, ptr addrspace(1) %b, i32 %j13
  %w13 = load i16, ptr addrspace(1) %pb13, align 2
  %s13 = add i16 %v13, %w13
  %py13 = getelementptr i16, ptr addrspace(1) %y, i32 %i13
  store i16 %s13, ptr addrspace(1) %py13, align 2
  %i14 = add i32 %i0, 14
  %j14 = add i32 %j, 14
  %px14 = getelementptr i16, ptr addrspace(1) %x, i32 %i14
  %v14 = load i16, ptr addrspace(1) %px14, align 2
  %pb14 = getelementptr i16, ptr addrspace(1) %b, i32 %j14
  %w14 = load i16, ptr addrspace(1) %pb14, align 2
  %s14 = add i16 %v14, %w14
  %py14 = getelementptr i16, ptr addrspace(1) %y, i32 %i14
  store i16 %s14, ptr addrspace(1) %py14, align 2
  %i15 = add i32 %i0, 15
  %j15 = add i32 %j, 15
  %px15 = getelementptr i16, ptr addrspace(1) %x, i32 %i15
  %v15 = load i16, ptr addrspace(1) %px15, align 2
  %pb15 = getelementptr i16, ptr addrspace(1) %b, i32 %j15
  %w15 = load i16, ptr addrspace(1) %pb15, align 2
  %s15 = add i16 %v15, %w15
  %py15 = getelementptr i16, ptr addrspace(1) %y, i32 %i15
  store i16 %s15, ptr addrspace(1) %py15, align 2
  %i16 = add i32 %i0, 16
  %j16 = add i32 %j, 16
  %px16 = getelementptr i16, ptr addrspace(1) %x, i32 %i16
  %v16 = load i16, ptr addrspace(1) %px16, align 2
  %pb16 = getelementptr i16, ptr addrspace(1) %b, i32 %j16
  %w16 = load i16, ptr addrspace(1) %pb16, align 2
  %s16 = add i16 %v16, %w16
  %py16 = getelementptr i16, ptr addrspace(1) %y, i32 %i16
  store i16 %s16, ptr addrspace(1) %py16, align 2
  %i17 = add i32 %i0, 17
  %j17 = add i32 %j, 17
  %px17 = getelementptr i16, ptr addrspace(1) %x, i32 %i17
  %v17 = load i16, ptr addrspace(1) %px17, align 2
  %pb17 = getelementptr i16, ptr addrspace(1) %b, i32 %j17
  %w17 = load i16, ptr addrspace(1) %pb17, align 2
  %s17 = add i16 %v17, %w17
  %py17 = getelementptr i16, ptr addrspace(1) %y, i32 %i17
  store i16 %s17, ptr addrspace(1) %py17, align 2
  %i18 = add i32 %i0, 18
  %j18 = add i32 %j, 18
  %px18 = getelementptr i16, ptr addrspace(1) %x, i32 %i18
  %v18 = load i16, ptr addrspace(1) %px18, align 2
  %pb18 = getelementptr i16, ptr addrspace(1) %b, i32 %j18
  %w18 = load i16, ptr addrspace(1) %pb18, align 2
  %s18 = add i16 %v18, %w18
  %py18 = getelementptr i16, ptr addrspace(1) %y, i32 %i18
  store i16 %s18, ptr addrspace(1) %py18, align 2
  %i19 = add i32 %i0, 19
  %j19 = add i32 %j, 19
  %px19 = getelementptr i16, ptr addrspace(1) %x, i32 %i19
  %v19 = load i16, ptr addrspace(1) %px19, align 2
  %pb19 = getelementptr i16, ptr addrspace(1) %b, i32 %j19
  %w19 = load i16, ptr addrspace(1) %pb19, align 2
  %s19 = add i16 %v19, %w19
  %py19 = getelementptr i16, ptr addrspace(1) %y, i32 %i19
  store i16 %s19, ptr addrspace(1) %py19, align 2
  %i20 = add i32 %i0, 20
  %j20 = add i32 %j, 20
  %px20 = getelementptr i16, ptr addrspace(1) %x, i32 %i20
  %v20 = load i16, ptr addrspace(1) %px20, align 2
  %pb20 = getelementptr i16, ptr addrspace(1) %b, i32 %j20
  %w20 = load i16, ptr addrspace(1) %pb20, align 2
  %s20 = add i16 %v20, %w20
  %py20 = getelementptr i16, ptr addrspace(1) %y, i32 %i20
  store i16 %s20, ptr addrspace(1) %py20, align 2
  %i21 = add i32 %i0, 21
  %j21 = add i32 %j, 21
  %px21 = getelementptr i16, ptr addrspace(1) %x, i32 %i21
  %v21 = load i16, ptr addrspace(1) %px21, align 2
  %pb21 = getelementptr i16, ptr addrspace(1) %b, i32 %j21
  %w21 = load i16, ptr addrspace(1) %pb21, align 2
  %s21 = add i16 %v21, %w21
  %py21 = getelementptr i16, ptr addrspace(1) %y, i32 %i21
  store i16 %s21, ptr addrspace(1) %py21, align 2
  %i22 = add i32 %i0, 22
  %j22 = add i32 %j, 22
  %px22 = getelementptr i16, ptr addrspace(1) %x, i32 %i22
  %v22 = load i16, ptr addrspace(1) %px22, align 2
  %pb22 = getelementptr i16, ptr addrspace(1) %b, i32 %j22
  %w22 = load i16, ptr addrspace(1) %pb22, align 2
  %s22 = add i16 %v22, %w22
  %py22 = getelementptr i16, ptr addrspace(1) %y, i32 %i22
  store i16 %s22, ptr addrspace(1) %py22, align 2
  %i23 = add i32 %i0, 23
  %j23 = add i32 %j, 23
  %px23 = getelementptr i16, ptr addrspace(1) %x, i32 %i23
  %v23 = load i16, ptr addrspace(1) %px23, align 2
  %pb23 = getelementptr i16, ptr addrspace(1) %b, i32 %j23
  %w23 = load i16, ptr addrspace(1) %pb23, align 2
  %s23 = add i16 %v23, %w23
  %py23 = getelementptr i16, ptr addrspace(1) %y, i32 %i23
  store i16 %s23, ptr addrspace(1) %py23, align 2
  %i24 = add i32 %i0, 24
  %j24 = add i32 %j, 24
  %px24 = getelementptr i16, ptr addrspace(1) %x, i32 %i24
  %v24 = load i16, ptr addrspace(1) %px24, align 2
  %pb24 = getelementptr i16, ptr addrspace(1) %b, i32 %j24
  %w24 = load i16, ptr addrspace(1) %pb24, align 2
  %s24 = add i16 %v24, %w24
  %py24 = getelementptr i16, ptr addrspace(1) %y, i32 %i24
  store i16 %s24, ptr addrspace(1) %py24, align 2
  %i25 = add i32 %i0, 25
  %j25 = add i32 %j, 25
  %px25 = getelementptr i16, ptr addrspace(1) %x, i32 %i25
  %v25 = load i16, ptr addrspace(1) %px25, align 2
  %pb25 = getelementptr i16, ptr addrspace(1) %b, i32 %j25
  %w25 = load i16, ptr addrspace(1) %pb25, align 2
  %s25 = add i16 %v25, %w25
  %py25 = getelementptr i16, ptr addrspace(1) %y, i32 %i25
  store i16 %s25, ptr addrspace(1) %py25, align 2
  %jn = add nuw nsw i32 %j, 26
  %c = icmp ult i32 %jn, 8190
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
