; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-loop-ranges=0 < %s | FileCheck %s --check-prefix=OFF

; A stencil over a thread's row (L15-E553): its loads' i32 indices,
; zext(258 * r + j + k), are no add recurrences LAA can bound without the
; thread position's bound, which the kernel's agx-upper-bounds gives. Without
; it the loop needed predicates or too many checks and stayed as it was;
; analyzed again with the bound, it is versioned. The bound is for the
; analysis only: taken off again.

; CHECK-LABEL: define agx_kernel void @stencil(
; CHECK: .lver.check:
; CHECK-NEXT: call i32 @llvm.agx.thread.position.in.grid.x(){{$}}
; CHECK: load float, {{.*}} !alias.scope
; CHECK: store float {{.*}} !alias.scope
; OFF-LABEL: define agx_kernel void @stencil(
; OFF-NOT: lver
; OFF: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.fma.f32(float, float, float)

define agx_kernel void @stencil(ptr addrspace(1) %x, ptr addrspace(1) %y) #0 {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 258
  %orow = shl i32 %r, 8
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %jn, %loop ]
  %b = add i32 %row, %j
  %ob = add i32 %orow, %j
  %i0 = add i32 %b, 0
  %z0 = zext i32 %i0 to i64
  %p0 = getelementptr float, ptr addrspace(1) %x, i64 %z0
  %v0 = load float, ptr addrspace(1) %p0, align 4
  %i1 = add i32 %b, 1
  %z1 = zext i32 %i1 to i64
  %p1 = getelementptr float, ptr addrspace(1) %x, i64 %z1
  %v1 = load float, ptr addrspace(1) %p1, align 4
  %i2 = add i32 %b, 2
  %z2 = zext i32 %i2 to i64
  %p2 = getelementptr float, ptr addrspace(1) %x, i64 %z2
  %v2 = load float, ptr addrspace(1) %p2, align 4
  %i3 = add i32 %b, 3
  %z3 = zext i32 %i3 to i64
  %p3 = getelementptr float, ptr addrspace(1) %x, i64 %z3
  %v3 = load float, ptr addrspace(1) %p3, align 4
  %i4 = add i32 %b, 4
  %z4 = zext i32 %i4 to i64
  %p4 = getelementptr float, ptr addrspace(1) %x, i64 %z4
  %v4 = load float, ptr addrspace(1) %p4, align 4
  %i5 = add i32 %b, 5
  %z5 = zext i32 %i5 to i64
  %p5 = getelementptr float, ptr addrspace(1) %x, i64 %z5
  %v5 = load float, ptr addrspace(1) %p5, align 4
  %s0a = fmul float %v0, 0.25
  %s0b = call float @llvm.fma.f32(float %v1, float 0.5, float %s0a)
  %s0 = call float @llvm.fma.f32(float %v2, float 0.25, float %s0b)
  %o0 = add i32 %ob, 0
  %oz0 = zext i32 %o0 to i64
  %q0 = getelementptr float, ptr addrspace(1) %y, i64 %oz0
  store float %s0, ptr addrspace(1) %q0, align 4
  %s1a = fmul float %v1, 0.25
  %s1b = call float @llvm.fma.f32(float %v2, float 0.5, float %s1a)
  %s1 = call float @llvm.fma.f32(float %v3, float 0.25, float %s1b)
  %o1 = add i32 %ob, 1
  %oz1 = zext i32 %o1 to i64
  %q1 = getelementptr float, ptr addrspace(1) %y, i64 %oz1
  store float %s1, ptr addrspace(1) %q1, align 4
  %s2a = fmul float %v2, 0.25
  %s2b = call float @llvm.fma.f32(float %v3, float 0.5, float %s2a)
  %s2 = call float @llvm.fma.f32(float %v4, float 0.25, float %s2b)
  %o2 = add i32 %ob, 2
  %oz2 = zext i32 %o2 to i64
  %q2 = getelementptr float, ptr addrspace(1) %y, i64 %oz2
  store float %s2, ptr addrspace(1) %q2, align 4
  %s3a = fmul float %v3, 0.25
  %s3b = call float @llvm.fma.f32(float %v4, float 0.5, float %s3a)
  %s3 = call float @llvm.fma.f32(float %v5, float 0.25, float %s3b)
  %o3 = add i32 %ob, 3
  %oz3 = zext i32 %o3 to i64
  %q3 = getelementptr float, ptr addrspace(1) %y, i64 %oz3
  store float %s3, ptr addrspace(1) %q3, align 4
  %jn = add nuw nsw i32 %j, 4
  %c = icmp ult i32 %jn, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
attributes #0 = { "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" }
