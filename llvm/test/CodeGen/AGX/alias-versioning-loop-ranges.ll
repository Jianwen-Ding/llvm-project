; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-loop-ranges=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-loop-ranges-runtime=0 < %s | FileCheck %s --check-prefix=NORT

; A stencil over a thread's row (L15-E553): its loads' i32 indices,
; zext(258 * r + j + k), are no add recurrences LAA can bound without the
; thread position's bound, which the kernel's agx-upper-bounds gives. Without
; it the loop needed predicates or too many checks and stayed as it was;
; analyzed again with the bound, it is versioned. The bound is for the
; analysis only: taken off again.

; CHECK-LABEL: define agx_kernel void @stencil(
; CHECK: .lver.check:
; CHECK: call i32 @llvm.agx.thread.position.in.grid.x(){{$}}
; CHECK: load float, {{.*}} !alias.scope
; CHECK: store float {{.*}} !alias.scope
; OFF-LABEL: define agx_kernel void @stencil(
; OFF-NOT: lver
; OFF: ret void

; A row of a runtime length, unrolled with a remainder (L15-E565): its
; accesses step 16 bytes a step; analyzed again with the bounds too.
; CHECK-LABEL: define agx_kernel void @rowdyn(
; CHECK: .lver.check:
; CHECK: store float {{.*}} !alias.scope
; NORT-LABEL: define agx_kernel void @rowdyn(
; NORT-NOT: lver
; NORT: ret void

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
define agx_kernel void @rowdyn(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree writeonly captures(none) %2, ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 captures(none) dereferenceable(4) %3) #0 {
entry:
  %.val = load i32, ptr addrspace(4) %3, align 4
  %4 = icmp sgt i32 %.val, 0
  br i1 %4, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %entry
  %5 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %6 = shl i32 %5, 9
  %wide.trip.count = zext nneg i32 %.val to i64
  %xtraiter = and i64 %wide.trip.count, 3
  %7 = icmp ult i32 %.val, 4
  br i1 %7, label %.epil.preheader, label %.lr.ph.new

.lr.ph.new:                                       ; preds = %.lr.ph
  %unroll_iter = and i64 %wide.trip.count, 2147483644
  br label %8

8:                                                ; preds = %8, %.lr.ph.new
  %indvars.iv = phi i64 [ 0, %.lr.ph.new ], [ %indvars.iv.next.3, %8 ]
  %niter = phi i64 [ 0, %.lr.ph.new ], [ %niter.next.3, %8 ]
  %9 = trunc nuw nsw i64 %indvars.iv to i32
  %10 = add i32 %6, %9
  %11 = zext i32 %10 to i64
  %12 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %11
  %13 = load float, ptr addrspace(1) %12, align 4
  %14 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv
  %15 = load float, ptr addrspace(1) %14, align 4
  %16 = fmul float %13, %15
  %17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %11
  store float %16, ptr addrspace(1) %17, align 4
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %18 = trunc nuw nsw i64 %indvars.iv.next to i32
  %19 = add i32 %6, %18
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %20
  %22 = load float, ptr addrspace(1) %21, align 4
  %23 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next
  %24 = load float, ptr addrspace(1) %23, align 4
  %25 = fmul float %22, %24
  %26 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %20
  store float %25, ptr addrspace(1) %26, align 4
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %27 = trunc nuw nsw i64 %indvars.iv.next.1 to i32
  %28 = add i32 %6, %27
  %29 = zext i32 %28 to i64
  %30 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %29
  %31 = load float, ptr addrspace(1) %30, align 4
  %32 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.1
  %33 = load float, ptr addrspace(1) %32, align 4
  %34 = fmul float %31, %33
  %35 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %29
  store float %34, ptr addrspace(1) %35, align 4
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %36 = trunc nuw nsw i64 %indvars.iv.next.2 to i32
  %37 = add i32 %6, %36
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %38
  %40 = load float, ptr addrspace(1) %39, align 4
  %41 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.2
  %42 = load float, ptr addrspace(1) %41, align 4
  %43 = fmul float %40, %42
  %44 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %38
  store float %43, ptr addrspace(1) %44, align 4
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 4
  %niter.next.3 = add nuw i64 %niter, 4
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, %unroll_iter
  br i1 %niter.ncmp.3, label %._crit_edge.loopexit.unr-lcssa, label %8

._crit_edge.loopexit.unr-lcssa:                   ; preds = %8
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %._crit_edge, label %.epil.preheader

.epil.preheader:                                  ; preds = %._crit_edge.loopexit.unr-lcssa, %.lr.ph
  %indvars.iv.epil.init = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next.3, %._crit_edge.loopexit.unr-lcssa ]
  %lcmp.mod1 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod1)
  br label %45

45:                                               ; preds = %45, %.epil.preheader
  %indvars.iv.epil = phi i64 [ %indvars.iv.epil.init, %.epil.preheader ], [ %indvars.iv.next.epil, %45 ]
  %epil.iter = phi i64 [ 0, %.epil.preheader ], [ %epil.iter.next, %45 ]
  %46 = trunc nuw nsw i64 %indvars.iv.epil to i32
  %47 = add i32 %6, %46
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %48
  %50 = load float, ptr addrspace(1) %49, align 4
  %51 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv.epil
  %52 = load float, ptr addrspace(1) %51, align 4
  %53 = fmul float %50, %52
  %54 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %48
  store float %53, ptr addrspace(1) %54, align 4
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %._crit_edge, label %45

._crit_edge:                                      ; preds = %._crit_edge.loopexit.unr-lcssa, %45, %entry
  ret void
}

attributes #0 = { "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" }
