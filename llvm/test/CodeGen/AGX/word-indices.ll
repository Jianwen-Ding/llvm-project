; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices -agx-loop-word-indices=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices -agx-loop-word-indices-exit=0 -agx-loop-word-indices-exit-count=0 < %s | FileCheck %s --check-prefix=EXIT
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices -agx-loop-word-indices-exit-count=0 < %s | FileCheck %s --check-prefix=COUNT
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices -agx-loop-exit-on-step-counter=0 < %s | FileCheck %s --check-prefix=STEP
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices -agx-loop-exit-count-uses-after=0 < %s | FileCheck %s --check-prefix=AFTER

; A loop's 16-byte loads of halves at a 32-bit counter of halves (LSR's,
; start 512 * row, the row's 16-byte steps): a counter of words beside it
; indexes them (L15-E594), the second load's 8 halves 4 words further on,
; where the half index took a 64-bit add for each address.
; CHECK-LABEL: define agx_kernel void @hsum(
; CHECK: %words.start = lshr exact i32 %start, 1
; The counter of halves, then read by its increment only, goes as well
; (L15-E623).
; CHECK: %words = phi i32 [ %words.start, %pre ], [ %words.next, %loop ]
; CHECK-NOT: %iv
; CHECK: getelementptr nuw i32, ptr addrspace(1) %x, i64
; CHECK: add nuw i32 %words, 4
; CHECK: %words.next = add nuw i32 %words, 8
; EXIT-LABEL: define agx_kernel void @hsum(
; EXIT: %iv = phi i32
; OFF-LABEL: define agx_kernel void @hsum(
; OFF-NOT: words
; OFF: getelementptr inbounds nuw [2 x i8]

; A counter whose 32-bit offsets could wrap (no bound on the row) is left.
; CHECK-LABEL: define agx_kernel void @unbounded(
; CHECK-NOT: words
; CHECK: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @hsum(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %start = shl i32 %t, 9
  %c0 = icmp sgt i32 %n, 0
  br i1 %c0, label %pre, label %out
pre:
  br label %loop
loop:
  %iv = phi i32 [ %start, %pre ], [ %iv.next, %loop ]
  %j = phi i32 [ 0, %pre ], [ %j.next, %loop ]
  %a = phi float [ 0.0, %pre ], [ %a2, %loop ]
  %z0 = zext i32 %iv to i64
  %p0 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %x, i64 %z0
  %v0 = load <8 x half>, ptr addrspace(1) %p0, align 4
  %i8 = add i32 %iv, 8
  %z8 = zext i32 %i8 to i64
  %p8 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %x, i64 %z8
  %v8 = load <8 x half>, ptr addrspace(1) %p8, align 4
  %e0 = fpext <8 x half> %v0 to <8 x float>
  %e8 = fpext <8 x half> %v8 to <8 x float>
  %s0 = call float @llvm.vector.reduce.fadd.v8f32(float %a, <8 x float> %e0)
  %a2 = call float @llvm.vector.reduce.fadd.v8f32(float %s0, <8 x float> %e8)
  %iv.next = add i32 %iv, 16
  %j.next = add nuw nsw i32 %j, 16
  %c = icmp slt i32 %j.next, %n
  br i1 %c, label %loop, label %out
out:
  %r = phi float [ 0.0, %entry ], [ %a2, %loop ]
  %tz = zext i32 %t to i64
  %q = getelementptr float, ptr addrspace(1) %y, i64 %tz
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

define agx_kernel void @unbounded(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n, i32 %row) {
entry:
  %start = shl i32 %row, 9
  %c0 = icmp sgt i32 %n, 0
  br i1 %c0, label %pre, label %out
pre:
  br label %loop
loop:
  %iv = phi i32 [ %start, %pre ], [ %iv.next, %loop ]
  %j = phi i32 [ 0, %pre ], [ %j.next, %loop ]
  %a = phi float [ 0.0, %pre ], [ %a1, %loop ]
  %z0 = zext i32 %iv to i64
  %p0 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %x, i64 %z0
  %v0 = load <8 x half>, ptr addrspace(1) %p0, align 4
  %e0 = fpext <8 x half> %v0 to <8 x float>
  %a1 = call float @llvm.vector.reduce.fadd.v8f32(float %a, <8 x float> %e0)
  %iv.next = add i32 %iv, 8
  %j.next = add nuw nsw i32 %j, 8
  %c = icmp slt i32 %j.next, %n
  br i1 %c, label %loop, label %out
out:
  %r = phi float [ 0.0, %entry ], [ %a1, %loop ]
  store float %r, ptr addrspace(1) %y, align 4
  ret void
}

declare float @llvm.vector.reduce.fadd.v8f32(float, <8 x float>)

attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" }

; The counter of bytes left with its increment and the exit compare only (a
; row of 512 floats in 16-byte steps): the compare is of the counter of words,
; 2048 / 4, and the counter of bytes goes (L15-E623).
; CHECK-LABEL: define agx_kernel void @exit(
; CHECK: %words.next = add nuw i32 %words, 4
; CHECK-NEXT: %c = icmp eq i32 %words.next, 512
; CHECK-NOT: %iv
; CHECK: ret void
; EXIT-LABEL: define agx_kernel void @exit(
; EXIT: %iv = phi i32
; EXIT: %c = icmp eq i64 %zn, 2048
; EXIT: %words.next = add nuw i32 %words, 4
define agx_kernel void @exit(ptr addrspace(1) %x, ptr addrspace(1) %y) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %tz = zext i32 %t to i64
  %off = shl nuw nsw i64 %tz, 11
  %xr = getelementptr i8, ptr addrspace(1) %x, i64 %off
  %yr = getelementptr i8, ptr addrspace(1) %y, i64 %off
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %z = zext i32 %iv to i64
  %p = getelementptr i8, ptr addrspace(1) %xr, i64 %z
  %v = load <4 x float>, ptr addrspace(1) %p, align 4
  %m = fmul <4 x float> %v, %v
  %q = getelementptr i8, ptr addrspace(1) %yr, i64 %z
  store <4 x float> %m, ptr addrspace(1) %q, align 4
  %iv.next = add nuw nsw i32 %iv, 16
  %zn = zext i32 %iv.next to i64
  %c = icmp eq i64 %zn, 2048
  br i1 %c, label %out, label %loop
out:
  ret void
}

; A count of iterations from 0 that only the exit compare with a runtime
; bound (a multiple of 16 below 2^31: an unrolled loop's) and a phi after the
; loop read: the compare is of the counter of words with its value at that
; bound, w.start + n / 16 * 8, and the phi takes n (L15-E624).
; CHECK-LABEL: define agx_kernel void @count(
; CHECK: %words.end = add i32 %words.start,
; CHECK: %words.next = add i32 %words, 8
; CHECK-NEXT: %c = icmp eq i32 %words.next, %words.end
; CHECK-NOT: %j
; CHECK: %done = phi i64 [ %nz, %loop ]
; COUNT-LABEL: define agx_kernel void @count(
; COUNT: %c = icmp eq i64 %jz, %nz
define agx_kernel void @count(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %start = shl i32 %t, 9
  %n16 = and i32 %n, 2147483632
  %nz = zext i32 %n16 to i64
  %c0 = icmp ugt i32 %n16, 0
  br i1 %c0, label %pre, label %out
pre:
  br label %loop
loop:
  %iv = phi i32 [ %start, %pre ], [ %iv.next, %loop ]
  %j = phi i32 [ 0, %pre ], [ %j.next, %loop ]
  %a = phi float [ 0.0, %pre ], [ %a2, %loop ]
  %z0 = zext i32 %iv to i64
  %p0 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %x, i64 %z0
  %v0 = load <8 x half>, ptr addrspace(1) %p0, align 4
  %i8 = add i32 %iv, 8
  %z8 = zext i32 %i8 to i64
  %p8 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %x, i64 %z8
  %v8 = load <8 x half>, ptr addrspace(1) %p8, align 4
  %e0 = fpext <8 x half> %v0 to <8 x float>
  %e8 = fpext <8 x half> %v8 to <8 x float>
  %s0 = call float @llvm.vector.reduce.fadd.v8f32(float %a, <8 x float> %e0)
  %a2 = call float @llvm.vector.reduce.fadd.v8f32(float %s0, <8 x float> %e8)
  %iv.next = add i32 %iv, 16
  %j.next = add nuw nsw i32 %j, 16
  %jz = zext i32 %j.next to i64
  %c = icmp eq i64 %jz, %nz
  br i1 %c, label %exit, label %loop
exit:
  %done = phi i64 [ %jz, %loop ]
  %r0 = phi float [ %a2, %loop ]
  %d = trunc i64 %done to i32
  %df = uitofp i32 %d to float
  %r1 = fadd float %r0, %df
  br label %out
out:
  %r = phi float [ 0.0, %entry ], [ %r1, %exit ]
  %tz = zext i32 %t to i64
  %q = getelementptr float, ptr addrspace(1) %y, i64 %tz
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; Likewise against a loop's other counter by a power of two (here of floats,
; which E594 leaves), the bound a 64-bit value below 2^32 by its known bits
; (L15-E625): the index's flags go, as it is now compared on the last
; iteration too.
; CHECK-LABEL: define agx_kernel void @step(
; CHECK: %[[N32:.*]] = trunc i64 %n8 to i32
; CHECK: %[[Q:.*]] = lshr exact i32 %[[N32]], 3
; CHECK: %[[S:.*]] = shl i32 %[[Q]], 3
; CHECK: %[[END:.*]] = add i32 %start, %[[S]]
; CHECK: %iv.next = add i32 %iv, 8
; CHECK-NEXT: %c = icmp eq i32 %iv.next, %[[END]]
; CHECK-NOT: %j
; CHECK: ret void
; STEP-LABEL: define agx_kernel void @step(
; STEP: %c = icmp eq i64 %jz, %n8
define agx_kernel void @step(ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %n) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %start = shl i32 %t, 9
  %n8 = and i64 %n, 2147483640
  %c0 = icmp ugt i64 %n8, 0
  br i1 %c0, label %pre, label %out
pre:
  br label %loop
loop:
  %iv = phi i32 [ %start, %pre ], [ %iv.next, %loop ]
  %j = phi i32 [ 0, %pre ], [ %j.next, %loop ]
  %z = zext i32 %iv to i64
  %p = getelementptr float, ptr addrspace(1) %x, i64 %z
  %v = load <8 x float>, ptr addrspace(1) %p, align 4
  %m = fmul <8 x float> %v, %v
  %q = getelementptr float, ptr addrspace(1) %y, i64 %z
  store <8 x float> %m, ptr addrspace(1) %q, align 4
  %iv.next = add nuw i32 %iv, 8
  %j.next = add nuw nsw i32 %j, 8
  %jz = zext i32 %j.next to i64
  %c = icmp eq i64 %jz, %n8
  br i1 %c, label %out, label %loop
out:
  ret void
}

; The count read after the loop beyond its exit block too (an unrolled
; loop's remainder starting at it, past the exit block that only the latch
; enters): there it is the bound (L15-E626).
; CHECK-LABEL: define agx_kernel void @after(
; CHECK: %c = icmp eq i32 %iv.next,
; CHECK-NOT: %j
; CHECK: %rest = phi i64 [ 0, %entry ], [ %n8, %exit ]
; AFTER-LABEL: define agx_kernel void @after(
; AFTER: %c = icmp eq i64 %jz, %n8
define agx_kernel void @after(ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %n) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %start = shl i32 %t, 9
  %n8 = and i64 %n, 2147483640
  %c0 = icmp ugt i64 %n8, 0
  br i1 %c0, label %pre, label %tail
pre:
  br label %loop
loop:
  %iv = phi i32 [ %start, %pre ], [ %iv.next, %loop ]
  %j = phi i32 [ 0, %pre ], [ %j.next, %loop ]
  %z = zext i32 %iv to i64
  %p = getelementptr float, ptr addrspace(1) %x, i64 %z
  %v = load <8 x float>, ptr addrspace(1) %p, align 4
  %m = fmul <8 x float> %v, %v
  %q = getelementptr float, ptr addrspace(1) %y, i64 %z
  store <8 x float> %m, ptr addrspace(1) %q, align 4
  %iv.next = add nuw i32 %iv, 8
  %j.next = add nuw nsw i32 %j, 8
  %jz = zext i32 %j.next to i64
  %c = icmp eq i64 %jz, %n8
  br i1 %c, label %exit, label %loop
exit:
  br label %tail
tail:
  %rest = phi i64 [ 0, %entry ], [ %jz, %exit ]
  %r = trunc i64 %rest to i32
  %tz = zext i32 %t to i64
  %o = getelementptr i32, ptr addrspace(1) %y, i64 %tz
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
}
