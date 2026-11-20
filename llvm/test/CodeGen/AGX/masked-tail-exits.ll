; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-masked-head-starts=0 < %s | FileCheck %s --check-prefix=NOHEAD
;
; A loop whose reductions take their identity once a mask on an increasing
; index holds (a causal mask: j + lane > row) ends there: its bound becomes
; umin(n, max(row - lane, 0) + 1) (L15-E476). Not with a store in the loop,
; a +0 identity of a sum that may start at -0, maxnum's -inf with a NaN start,
; a value of the loop used after it, or a mask that turns off.

declare float @llvm.maxnum.f32(float, float)

; CHECK-LABEL: @causal(
; CHECK: %[[L:.*]] = call i32 @llvm.usub.sat.i32(i32 %row32, i32 %lane32)
; CHECK: %[[E:.*]] = zext i32 %[[L]] to i64
; CHECK: %[[P:.*]] = call i64 @llvm.uadd.sat.i64(i64 %[[E]], i64 1)
; CHECK: %unmasked.bound = call i64 @llvm.umin.i64(i64 %n, i64 %[[P]])
; With the loop skipped where its first iteration is masked, the mask is
; gone from it (L15-E518):
; CHECK: %first.masked = icmp ugt i64 {{.*}}, %row
; CHECK: br i1 %first.masked, label %done.loopexit, label %loop
; CHECK-NOT: select
; CHECK: %r.n = fadd float %r, %x
; CHECK: icmp ult i64 %j.n, %unmasked.bound
; CHECK: phi float [ %r.n, %loop ], [ 0.000000e+00, %loop.preheader ]
define agx_kernel void @causal(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m = icmp ugt i64 %a, %row
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0.0, float %x
  %r.n = fadd float %r, %s
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}

; maxnum with -inf from a start that is no NaN; >= without the + 1
; CHECK-LABEL: @causal_max(
; CHECK: %[[L2:.*]] = call i32 @llvm.usub.sat.i32(i32 %row32, i32 %lane32)
; CHECK-NOT: uadd.sat
; CHECK: %unmasked.bound = call i64 @llvm.umin.i64
define agx_kernel void @causal_max(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0xFFF0000000000000, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m = icmp uge i64 %a, %row
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0xFFF0000000000000, float %x
  %r.n = call float @llvm.maxnum.f32(float %r, float %s)
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0xFFF0000000000000, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @store(
; CHECK-NOT: unmasked.bound
; CHECK: ret void
define agx_kernel void @store(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m = icmp ugt i64 %a, %row
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0.0, float %x
  %r.n = fadd float %r, %s
  store float %x, ptr addrspace(1) %o
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @negzero_start(
; CHECK-NOT: unmasked.bound
; CHECK: ret void
define agx_kernel void @negzero_start(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ -0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m = icmp ugt i64 %a, %row
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0.0, float %x
  %r.n = fadd float %r, %s
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ -0.0, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @nan_start(
; CHECK-NOT: unmasked.bound
; CHECK: ret void
define agx_kernel void @nan_start(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0x7FF8000000000000, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m = icmp ugt i64 %a, %row
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0xFFF0000000000000, float %x
  %r.n = call float @llvm.maxnum.f32(float %r, float %s)
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0x7FF8000000000000, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @used_after(
; CHECK-NOT: unmasked.bound
; CHECK: ret void
define agx_kernel void @used_after(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m = icmp ugt i64 %a, %row
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0.0, float %x
  %r.n = fadd float %r, %s
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %xa = phi float [ 0.0, %entry ], [ %x, %loop ]
  %f2 = fadd float %f, %xa
  store float %f2, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @turns_off(
; CHECK-NOT: unmasked.bound
; CHECK: ret void
define agx_kernel void @turns_off(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m = icmp ult i64 %a, %row
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0.0, float %x
  %r.n = fadd float %r, %s
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}

; A sliding window's j <= row && j + 128 >= row, the unmasked values'
; condition (an and with the inverse compare), or its masked form (an or
; with the compare): the tail compare implies the mask (L15-E480). An and of
; the compare with something else does not.
; Its lower edge j + 128 >= row, a mask true up to an iteration, moves the
; start past those iterations (at most to the last one), after the bound
; (L15-E481).
; CHECK-LABEL: @window(
; CHECK: %unmasked.bound = call i64 @llvm.umin.i64
; CHECK: %unmasked.start = add nuw{{.*}} i64
; CHECK: %unmasked.start.b = call i64 @llvm.umin.i64(i64 %unmasked.start, i64
; Both masks are then gone from the loop: it is skipped where its first
; iteration is masked under either (L15-E519).
; CHECK: %first.masked = icmp ugt i64
; CHECK: %first.masked1 = icmp ult i64
; CHECK: or i1 %first.masked, %first.masked1
; CHECK: phi i64 [{{.*}}%unmasked.start.b, %loop.preheader ]
; CHECK-NOT: select
; CHECK: %r.n = fadd float %r, %x
; NOHEAD-LABEL: @window(
; NOHEAD-NOT: unmasked.start
; NOHEAD: ret void
define agx_kernel void @window(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %u1 = icmp ule i64 %a, %row
  %aw = add nuw i64 %a, 128
  %u2 = icmp uge i64 %aw, %row
  %ok = and i1 %u1, %u2
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %ok, float %x, float 0.0
  %r.n = fadd float %r, %s
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @masked_and(
; CHECK-NOT: unmasked.bound
; CHECK: ret void
define agx_kernel void @masked_and(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %row32, i32 %lane32, i32 %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %m1 = icmp ugt i64 %a, %row
  %m2 = icmp ult i64 %a, 500
  %m = and i1 %m1, %m2
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %m, float 0.0, float %x
  %r.n = fadd float %r, %s
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}
