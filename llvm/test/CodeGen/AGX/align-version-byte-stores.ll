; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=THREE
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-align-version-byte-stores=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-align-version-skip-alias-fallback=0 < %s | FileCheck %s --check-prefix=FOUR
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-truncated-compares=0 < %s | FileCheck %s --check-prefix=MOVE

; A row loop that loads words and stores bytes (int8 quantization, a mask):
; alias versioning's copy (L15-E546) is versioned again for a 4-byte aligned
; output row, whose bytes then go out four at a time (L15-E547). Alias
; versioning's copy for overlapping rows, which runs only when they overlap,
; is left: three copies, not four.

; CHECK-LABEL: quant:
; CHECK: device_store 0, i8, xyzw,
; THREE-LABEL: quant:
; THREE-COUNT-3: Loop Header
; THREE-NOT: Loop Header
; THREE-LABEL: mask:
; OFF-LABEL: quant:
; OFF-NOT: i8, xyzw
; OFF-LABEL: mask:
; OFF-NOT: i8, xyzw
; OFF: stop
; FOUR-LABEL: quant:
; FOUR-COUNT-4: Loop Header
; FOUR-NOT: Loop Header
; FOUR-LABEL: mask:

; A mask's truth values are written into the halves the store takes by the
; compare-selects themselves, not into 32-bit registers whose low halves
; moves then copy (L15-E548).
; CHECK-LABEL: mask:
; CHECK-NOT: bitop 0101, r{{[0-9]+}}h, r{{[0-9]+}}l, 0
; CHECK: fcmpsel gt, r{{[0-9]+}}h, r{{[0-9]+}}, u4, 1, 0
; CHECK-NOT: bitop 0101, r{{[0-9]+}}h, r{{[0-9]+}}l, 0
; CHECK: device_store 0, i8, xyzw,
; MOVE-LABEL: mask:
; MOVE: bitop 0101, r{{[0-9]+}}h, r{{[0-9]+}}l, 0
; MOVE: device_store 0, i8, xyzw,

define agx_kernel void @quant(ptr addrspace(1) %x, ptr addrspace(1) %q, float %s) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %jn, %loop ]
  %i0 = add i32 %row, %j
  %px0 = getelementptr float, ptr addrspace(1) %x, i32 %i0
  %v0 = load float, ptr addrspace(1) %px0, align 4
  %m0 = fmul float %v0, %s
  %b0 = fptosi float %m0 to i8
  %pq0 = getelementptr i8, ptr addrspace(1) %q, i32 %i0
  store i8 %b0, ptr addrspace(1) %pq0, align 1
  %i1 = add i32 %i0, 1
  %px1 = getelementptr float, ptr addrspace(1) %x, i32 %i1
  %v1 = load float, ptr addrspace(1) %px1, align 4
  %m1 = fmul float %v1, %s
  %b1 = fptosi float %m1 to i8
  %pq1 = getelementptr i8, ptr addrspace(1) %q, i32 %i1
  store i8 %b1, ptr addrspace(1) %pq1, align 1
  %i2 = add i32 %i0, 2
  %px2 = getelementptr float, ptr addrspace(1) %x, i32 %i2
  %v2 = load float, ptr addrspace(1) %px2, align 4
  %m2 = fmul float %v2, %s
  %b2 = fptosi float %m2 to i8
  %pq2 = getelementptr i8, ptr addrspace(1) %q, i32 %i2
  store i8 %b2, ptr addrspace(1) %pq2, align 1
  %i3 = add i32 %i0, 3
  %px3 = getelementptr float, ptr addrspace(1) %x, i32 %i3
  %v3 = load float, ptr addrspace(1) %px3, align 4
  %m3 = fmul float %v3, %s
  %b3 = fptosi float %m3 to i8
  %pq3 = getelementptr i8, ptr addrspace(1) %q, i32 %i3
  store i8 %b3, ptr addrspace(1) %pq3, align 1
  %i4 = add i32 %i0, 4
  %px4 = getelementptr float, ptr addrspace(1) %x, i32 %i4
  %v4 = load float, ptr addrspace(1) %px4, align 4
  %m4 = fmul float %v4, %s
  %b4 = fptosi float %m4 to i8
  %pq4 = getelementptr i8, ptr addrspace(1) %q, i32 %i4
  store i8 %b4, ptr addrspace(1) %pq4, align 1
  %i5 = add i32 %i0, 5
  %px5 = getelementptr float, ptr addrspace(1) %x, i32 %i5
  %v5 = load float, ptr addrspace(1) %px5, align 4
  %m5 = fmul float %v5, %s
  %b5 = fptosi float %m5 to i8
  %pq5 = getelementptr i8, ptr addrspace(1) %q, i32 %i5
  store i8 %b5, ptr addrspace(1) %pq5, align 1
  %i6 = add i32 %i0, 6
  %px6 = getelementptr float, ptr addrspace(1) %x, i32 %i6
  %v6 = load float, ptr addrspace(1) %px6, align 4
  %m6 = fmul float %v6, %s
  %b6 = fptosi float %m6 to i8
  %pq6 = getelementptr i8, ptr addrspace(1) %q, i32 %i6
  store i8 %b6, ptr addrspace(1) %pq6, align 1
  %i7 = add i32 %i0, 7
  %px7 = getelementptr float, ptr addrspace(1) %x, i32 %i7
  %v7 = load float, ptr addrspace(1) %px7, align 4
  %m7 = fmul float %v7, %s
  %b7 = fptosi float %m7 to i8
  %pq7 = getelementptr i8, ptr addrspace(1) %q, i32 %i7
  store i8 %b7, ptr addrspace(1) %pq7, align 1
  %jn = add nuw nsw i32 %j, 8
  %c = icmp ult i32 %jn, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define agx_kernel void @mask(ptr addrspace(1) %x, ptr addrspace(1) %q, float %s) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %jn, %loop ]
  %i0 = add i32 %row, %j
  %px0 = getelementptr float, ptr addrspace(1) %x, i32 %i0
  %v0 = load float, ptr addrspace(1) %px0, align 4
  %c0 = fcmp ogt float %v0, %s
  %b0 = zext i1 %c0 to i8
  %pq0 = getelementptr i8, ptr addrspace(1) %q, i32 %i0
  store i8 %b0, ptr addrspace(1) %pq0, align 1
  %i1 = add i32 %i0, 1
  %px1 = getelementptr float, ptr addrspace(1) %x, i32 %i1
  %v1 = load float, ptr addrspace(1) %px1, align 4
  %c1 = fcmp ogt float %v1, %s
  %b1 = zext i1 %c1 to i8
  %pq1 = getelementptr i8, ptr addrspace(1) %q, i32 %i1
  store i8 %b1, ptr addrspace(1) %pq1, align 1
  %i2 = add i32 %i0, 2
  %px2 = getelementptr float, ptr addrspace(1) %x, i32 %i2
  %v2 = load float, ptr addrspace(1) %px2, align 4
  %c2 = fcmp ogt float %v2, %s
  %b2 = zext i1 %c2 to i8
  %pq2 = getelementptr i8, ptr addrspace(1) %q, i32 %i2
  store i8 %b2, ptr addrspace(1) %pq2, align 1
  %i3 = add i32 %i0, 3
  %px3 = getelementptr float, ptr addrspace(1) %x, i32 %i3
  %v3 = load float, ptr addrspace(1) %px3, align 4
  %c3 = fcmp ogt float %v3, %s
  %b3 = zext i1 %c3 to i8
  %pq3 = getelementptr i8, ptr addrspace(1) %q, i32 %i3
  store i8 %b3, ptr addrspace(1) %pq3, align 1
  %i4 = add i32 %i0, 4
  %px4 = getelementptr float, ptr addrspace(1) %x, i32 %i4
  %v4 = load float, ptr addrspace(1) %px4, align 4
  %c4 = fcmp ogt float %v4, %s
  %b4 = zext i1 %c4 to i8
  %pq4 = getelementptr i8, ptr addrspace(1) %q, i32 %i4
  store i8 %b4, ptr addrspace(1) %pq4, align 1
  %i5 = add i32 %i0, 5
  %px5 = getelementptr float, ptr addrspace(1) %x, i32 %i5
  %v5 = load float, ptr addrspace(1) %px5, align 4
  %c5 = fcmp ogt float %v5, %s
  %b5 = zext i1 %c5 to i8
  %pq5 = getelementptr i8, ptr addrspace(1) %q, i32 %i5
  store i8 %b5, ptr addrspace(1) %pq5, align 1
  %i6 = add i32 %i0, 6
  %px6 = getelementptr float, ptr addrspace(1) %x, i32 %i6
  %v6 = load float, ptr addrspace(1) %px6, align 4
  %c6 = fcmp ogt float %v6, %s
  %b6 = zext i1 %c6 to i8
  %pq6 = getelementptr i8, ptr addrspace(1) %q, i32 %i6
  store i8 %b6, ptr addrspace(1) %pq6, align 1
  %i7 = add i32 %i0, 7
  %px7 = getelementptr float, ptr addrspace(1) %x, i32 %i7
  %v7 = load float, ptr addrspace(1) %px7, align 4
  %c7 = fcmp ogt float %v7, %s
  %b7 = zext i1 %c7 to i8
  %pq7 = getelementptr i8, ptr addrspace(1) %q, i32 %i7
  store i8 %b7, ptr addrspace(1) %pq7, align 1
  %jn = add nuw nsw i32 %j, 8
  %c = icmp ult i32 %jn, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
