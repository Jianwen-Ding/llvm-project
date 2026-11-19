; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
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
; CHECK: icmp ult i64 %j.n, %unmasked.bound
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
