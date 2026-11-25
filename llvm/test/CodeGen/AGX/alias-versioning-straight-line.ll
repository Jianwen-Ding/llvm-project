; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes=agx-alias-version -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes=agx-alias-version -agx-alias-versioning-straight-line=0 -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes=agx-alias-version -agx-alias-versioning-diff-checks=0 -S < %s | FileCheck %s --check-prefix=BOUNDS

; A straight-line run of stores and loads (L15-E528): an unrolled pass over a
; row that stores y and reloads x, as softmax's last pass after its sum. A copy
; for y and x not overlapping (the thread's own ranges) takes the values x and
; exp(x - m) had before the run: no load or exp in it. The run as it was
; otherwise.

; The run starts at the first store: the load before it, its exp too, take
; the earlier values in both copies.
; CHECK-LABEL: define agx_kernel void @normalize(
; CHECK-COUNT-4: call float @llvm.exp.f32
; y's [y, y + 16) and x's [x + 4, x + 16) apart where x - y + 15 >=u 27, one
; compare of the difference (L15-E636), instead of four ends compared.
; CHECK: %[[D:.*]] = sub i64 %{{.*}}, %{{.*}}
; CHECK: %[[K:.*]] = sub i64 %[[D]], -15
; CHECK: %agx.alias.diff = icmp uge i64 %[[K]], 27
; CHECK: br i1 %{{.*}}, label %[[FAST:.*]], label %[[SLOW:.*]]
; CHECK: [[SLOW]]:
; CHECK-COUNT-3: call float @llvm.exp.f32
; CHECK: [[FAST]]:
; CHECK-NOT: load
; CHECK-NOT: call float @llvm.exp.f32
; CHECK: store float {{.*}} !alias.scope
; CHECK-NOT: load
; CHECK-NOT: call float @llvm.exp.f32
; CHECK: br label %entry.noalias.end
; BOUNDS-LABEL: define agx_kernel void @normalize(
; BOUNDS-COUNT-2: icmp ule i64
; BOUNDS-NEXT: or i1
; BOUNDS-NEXT: br i1
; OFF-LABEL: define agx_kernel void @normalize(
; OFF-NOT: noalias
; OFF: ret void

; A run whose copy would not be smaller (nothing it loads was loaded before)
; is left as one block.
; CHECK-LABEL: define agx_kernel void @scale(
; CHECK-NOT: br i1
; CHECK-NOT: noalias
; CHECK: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp.f32(float)

define agx_kernel void @normalize(ptr addrspace(1) %x, ptr addrspace(1) %y, float %m) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %r = shl i32 %t, 2
  %i0 = zext i32 %r to i64
  %r1 = add i32 %r, 1
  %i1 = zext i32 %r1 to i64
  %r2 = add i32 %r, 2
  %i2 = zext i32 %r2 to i64
  %r3 = add i32 %r, 3
  %i3 = zext i32 %r3 to i64
  %p0 = getelementptr float, ptr addrspace(1) %x, i64 %i0
  %p1 = getelementptr float, ptr addrspace(1) %x, i64 %i1
  %p2 = getelementptr float, ptr addrspace(1) %x, i64 %i2
  %p3 = getelementptr float, ptr addrspace(1) %x, i64 %i3
  %a0 = load float, ptr addrspace(1) %p0, align 4
  %a1 = load float, ptr addrspace(1) %p1, align 4
  %a2 = load float, ptr addrspace(1) %p2, align 4
  %a3 = load float, ptr addrspace(1) %p3, align 4
  %d0 = fsub float %a0, %m
  %e0 = call float @llvm.exp.f32(float %d0)
  %d1 = fsub float %a1, %m
  %e1 = call float @llvm.exp.f32(float %d1)
  %d2 = fsub float %a2, %m
  %e2 = call float @llvm.exp.f32(float %d2)
  %d3 = fsub float %a3, %m
  %e3 = call float @llvm.exp.f32(float %d3)
  %s0 = fadd float %e0, %e1
  %s1 = fadd float %s0, %e2
  %s2 = fadd float %s1, %e3
  %inv = fdiv float 1.0, %s2
  %b0 = load float, ptr addrspace(1) %p0, align 4
  %f0 = fsub float %b0, %m
  %g0 = call float @llvm.exp.f32(float %f0)
  %o0 = fmul float %g0, %inv
  %q0 = getelementptr float, ptr addrspace(1) %y, i64 %i0
  store float %o0, ptr addrspace(1) %q0, align 4
  %b1 = load float, ptr addrspace(1) %p1, align 4
  %f1 = fsub float %b1, %m
  %g1 = call float @llvm.exp.f32(float %f1)
  %o1 = fmul float %g1, %inv
  %q1 = getelementptr float, ptr addrspace(1) %y, i64 %i1
  store float %o1, ptr addrspace(1) %q1, align 4
  %b2 = load float, ptr addrspace(1) %p2, align 4
  %f2 = fsub float %b2, %m
  %g2 = call float @llvm.exp.f32(float %f2)
  %o2 = fmul float %g2, %inv
  %q2 = getelementptr float, ptr addrspace(1) %y, i64 %i2
  store float %o2, ptr addrspace(1) %q2, align 4
  %b3 = load float, ptr addrspace(1) %p3, align 4
  %f3 = fsub float %b3, %m
  %g3 = call float @llvm.exp.f32(float %f3)
  %o3 = fmul float %g3, %inv
  %q3 = getelementptr float, ptr addrspace(1) %y, i64 %i3
  store float %o3, ptr addrspace(1) %q3, align 4
  ret void
}

define agx_kernel void @scale(ptr addrspace(1) %x, ptr addrspace(1) %y, float %m) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %r = shl i32 %t, 2
  %i0 = zext i32 %r to i64
  %r1 = add i32 %r, 1
  %i1 = zext i32 %r1 to i64
  %r2 = add i32 %r, 2
  %i2 = zext i32 %r2 to i64
  %r3 = add i32 %r, 3
  %i3 = zext i32 %r3 to i64
  %p0 = getelementptr float, ptr addrspace(1) %x, i64 %i0
  %p1 = getelementptr float, ptr addrspace(1) %x, i64 %i1
  %p2 = getelementptr float, ptr addrspace(1) %x, i64 %i2
  %p3 = getelementptr float, ptr addrspace(1) %x, i64 %i3
  %q0 = getelementptr float, ptr addrspace(1) %y, i64 %i0
  %q1 = getelementptr float, ptr addrspace(1) %y, i64 %i1
  %q2 = getelementptr float, ptr addrspace(1) %y, i64 %i2
  %q3 = getelementptr float, ptr addrspace(1) %y, i64 %i3
  %b0 = load float, ptr addrspace(1) %p0, align 4
  %o0 = fmul float %b0, %m
  store float %o0, ptr addrspace(1) %q0, align 4
  %b1 = load float, ptr addrspace(1) %p1, align 4
  %o1 = fmul float %b1, %m
  store float %o1, ptr addrspace(1) %q1, align 4
  %b2 = load float, ptr addrspace(1) %p2, align 4
  %o2 = fmul float %b2, %m
  store float %o2, ptr addrspace(1) %q2, align 4
  %b3 = load float, ptr addrspace(1) %p3, align 4
  %o3 = fmul float %b3, %m
  store float %o3, ptr addrspace(1) %q3, align 4
  ret void
}
