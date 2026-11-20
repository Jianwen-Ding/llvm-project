; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -start-before=agx-word-indices -stop-after=agx-word-indices -agx-loop-word-indices=0 < %s | FileCheck %s --check-prefix=OFF

; A loop's 16-byte loads of halves at a 32-bit counter of halves (LSR's,
; start 512 * row, the row's 16-byte steps): a counter of words beside it
; indexes them (L15-E594), the second load's 8 halves 4 words further on,
; where the half index took a 64-bit add for each address.
; CHECK-LABEL: define agx_kernel void @hsum(
; CHECK: %words.start = lshr exact i32 %start, 1
; CHECK: %words = phi i32 [ %words.start, %pre ], [ %words.next, %loop ]
; CHECK: getelementptr nuw i32, ptr addrspace(1) %x, i64
; CHECK: add nuw i32 %words, 4
; CHECK: %words.next = add nuw i32 %words, 8
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
