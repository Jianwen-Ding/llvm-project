; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes=agx-mark-exps-below-max -S < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=LLC

; A row maximum over an i32 counter and the sum of its exponentials over an
; i64 one: their trip counts compared at the wider type (both unsigned), not
; an assertion in SCEV's compare of APInts of different widths (L15-E743).
; CHECK: call float @llvm.exp.f32(float %d), !agx.exp.below.max
; LLC-LABEL: mixed:
; LLC: stop

define agx_kernel void @mixed(ptr addrspace(1) %x, ptr addrspace(1) %o) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 6
  %rz = zext i32 %row to i64
  %base = getelementptr inbounds float, ptr addrspace(1) %x, i64 %rz
  br label %max
max:
  %i = phi i32 [ 0, %entry ], [ %i1, %max ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %max ]
  %iz = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %base, i64 %iz
  %v = load float, ptr addrspace(1) %p, align 4
  %m1 = call float @llvm.maximum.f32(float %m, float %v)
  %i1 = add nuw nsw i32 %i, 1
  %c1 = icmp ult i32 %i1, 64
  br i1 %c1, label %max, label %sum
sum:
  %j = phi i64 [ 0, %max ], [ %j1, %sum ]
  %s = phi float [ 0.0, %max ], [ %s1, %sum ]
  %p2 = getelementptr inbounds float, ptr addrspace(1) %base, i64 %j
  %w = load float, ptr addrspace(1) %p2, align 4
  %d = fsub float %w, %m1
  %e = call float @llvm.exp.f32(float %d)
  %s1 = fadd float %s, %e
  %j1 = add nuw nsw i64 %j, 1
  %c2 = icmp ult i64 %j1, 64
  br i1 %c2, label %sum, label %done
done:
  %tz = zext i32 %t to i64
  %q = getelementptr inbounds float, ptr addrspace(1) %o, i64 %tz
  store float %s1, ptr addrspace(1) %q, align 4
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.maximum.f32(float, float)
declare float @llvm.exp.f32(float)
