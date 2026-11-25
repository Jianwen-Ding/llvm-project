; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; A running maximum whose sign of zero reaches only exponentials, also through
; SIMD-group shuffles of its bits and through threadgroup memory whose every
; load that may read it has such uses, is nsz (L15-E415). The threadgroup
; accesses are bounded within their globals (lane-0 stores at (tid >> 5) * 4,
; under the special registers' ranges), or a load may read any of them.

@g = internal addrspace(3) global [32 x float] poison, align 4
@h = internal addrspace(3) global [32 x float] poison, align 4

declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare i32 @llvm.agx.simd.shuffle.xor.i32(i32, i32)
declare void @llvm.agx.threadgroup.barrier()
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.exp2.f32(float)

; CHECK-LABEL: @shuffled(
; CHECK: call nsz float @llvm.maxnum.f32
; CHECK: call nsz float @llvm.maxnum.f32
define agx_kernel void @shuffled(ptr addrspace(1) %p, float %a) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %m = call float @llvm.maxnum.f32(float %x, float %a)
  %b = bitcast float %m to i32
  %s = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %b, i32 16)
  %sf = bitcast i32 %s to float
  %m2 = call float @llvm.maxnum.f32(float %m, float %sf)
  %d = fsub float %x, %m2
  %e = call float @llvm.exp2.f32(float %d)
  store float %e, ptr addrspace(1) %q, align 4
  ret void
}

; CHECK-LABEL: @through_threadgroup(
; CHECK: call nsz float @llvm.maxnum.f32
define agx_kernel void @through_threadgroup(ptr addrspace(1) %p, float %a) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %m = call float @llvm.maxnum.f32(float %x, float %a)
  %w = lshr i32 %t, 5
  %l = and i32 %t, 31
  %first = icmp eq i32 %l, 0
  br i1 %first, label %st, label %join
st:
  %gp = getelementptr float, ptr addrspace(3) @g, i32 %w
  store float %m, ptr addrspace(3) %gp, align 4
  br label %join
join:
  call void @llvm.agx.threadgroup.barrier()
  %lp = getelementptr float, ptr addrspace(3) @g, i32 %l
  %v = load float, ptr addrspace(3) %lp, align 4
  %d = fsub float %x, %v
  %e = call float @llvm.exp2.f32(float %d)
  store float %e, ptr addrspace(1) %q, align 4
  ret void
}

; A load of the stored maximum whose sign reaches a division keeps it.
; CHECK-LABEL: @sign_read(
; CHECK: call float @llvm.maxnum.f32
define agx_kernel void @sign_read(ptr addrspace(1) %p, float %a) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %m = call float @llvm.maxnum.f32(float %x, float %a)
  %l = and i32 %t, 31
  %gp = getelementptr float, ptr addrspace(3) @g, i32 %l
  store float %m, ptr addrspace(3) %gp, align 4
  call void @llvm.agx.threadgroup.barrier()
  %v = load float, ptr addrspace(3) %gp, align 4
  %r = fdiv float 1.0, %v
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; A load of another global at an offset that may leave it (it may read @g)
; whose value reaches a division keeps the sign; at a bounded offset it
; cannot read @g.
; CHECK-LABEL: @other_unbounded(
; CHECK: call float @llvm.maxnum.f32
; CHECK-LABEL: @other_bounded(
; CHECK: call nsz float @llvm.maxnum.f32
define agx_kernel void @other_unbounded(ptr addrspace(1) %p, float %a, i64 %k) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %m = call float @llvm.maxnum.f32(float %x, float %a)
  %l = and i32 %t, 31
  %gp = getelementptr float, ptr addrspace(3) @g, i32 %l
  store float %m, ptr addrspace(3) %gp, align 4
  call void @llvm.agx.threadgroup.barrier()
  %v = load float, ptr addrspace(3) %gp, align 4
  %ha = add i64 %k, ptrtoint (ptr addrspace(3) @h to i64)
  %hp = inttoptr i64 %ha to ptr addrspace(3)
  %u = load float, ptr addrspace(3) %hp, align 4
  %r = fdiv float 1.0, %u
  %d = fsub float %x, %v
  %e = call float @llvm.exp2.f32(float %d)
  %o = fadd float %e, %r
  store float %o, ptr addrspace(1) %q, align 4
  ret void
}
define agx_kernel void @other_bounded(ptr addrspace(1) %p, float %a, i64 %k) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %m = call float @llvm.maxnum.f32(float %x, float %a)
  %l = and i32 %t, 31
  %gp = getelementptr float, ptr addrspace(3) @g, i32 %l
  store float %m, ptr addrspace(3) %gp, align 4
  call void @llvm.agx.threadgroup.barrier()
  %v = load float, ptr addrspace(3) %gp, align 4
  %k31 = and i64 %k, 124
  %ha = add i64 %k31, ptrtoint (ptr addrspace(3) @h to i64)
  %hp = inttoptr i64 %ha to ptr addrspace(3)
  %u = load float, ptr addrspace(3) %hp, align 4
  %r = fdiv float 1.0, %u
  %d = fsub float %x, %v
  %e = call float @llvm.exp2.f32(float %d)
  %o = fadd float %e, %r
  store float %o, ptr addrspace(1) %q, align 4
  ret void
}
