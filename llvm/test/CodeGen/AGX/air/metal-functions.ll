; AIR functions MSL code calls that the portable passes do not make:
; ldexp and frexp (its exponent through thread memory), and the SIMD-group's
; active threads (simd_active_threads_mask, simd_is_first).
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o /dev/null

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %l = call float @llvm.ldexp.f32.i32(float %x, i32 %e)
; CHECK: %lh = call half @llvm.ldexp.f16.i32(half %h, i32 %e)
; CHECK: [[F:%.*]] = call { float, i32 } @llvm.frexp.f32.i32(float %x)
; CHECK: [[E:%.*]] = extractvalue { float, i32 } [[F]], 1
; CHECK: store i32 [[E]], ptr addrspace(5) %exp
; CHECK: %m = extractvalue { float, i32 } [[F]], 0
; CHECK: [[B:%.*]] = call i32 @llvm.agx.simd.ballot(i1 true)
; CHECK: %mask = zext i32 [[B]] to i64
; CHECK: [[LANE:%.*]] = call i32 @llvm.agx.thread.index.in.simdgroup()
; CHECK: [[B2:%.*]] = call i32 @llvm.agx.simd.ballot(i1 true)
; CHECK: [[FIRST:%.*]] = call i32 @llvm.cttz.i32(i32 [[B2]], i1 false)
; CHECK: %first = icmp eq i32 [[LANE]], [[FIRST]]
; CHECK-NOT: @air.

target triple = "air64-apple-macosx14.0.0"

declare float @air.ldexp.f32(float, i32)
declare half @air.ldexp.f16(half, i32)
declare float @air.frexp.f32(float, ptr)
declare i64 @air.simd_active_threads_mask.i64()
declare i1 @air.simd_is_first()

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %args) {
  %x = load float, ptr addrspace(2) %args
  %ep = getelementptr i8, ptr addrspace(2) %args, i64 4
  %e = load i32, ptr addrspace(2) %ep
  %hp = getelementptr i8, ptr addrspace(2) %args, i64 8
  %h = load half, ptr addrspace(2) %hp
  %l = call float @air.ldexp.f32(float %x, i32 %e)
  store volatile float %l, ptr addrspace(1) %out
  %lh = call half @air.ldexp.f16(half %h, i32 %e)
  store volatile half %lh, ptr addrspace(1) %out
  %exp = alloca i32
  %m = call float @air.frexp.f32(float %x, ptr %exp)
  %n = load i32, ptr %exp
  store volatile float %m, ptr addrspace(1) %out
  store volatile i32 %n, ptr addrspace(1) %out
  %mask = call i64 @air.simd_active_threads_mask.i64()
  store volatile i64 %mask, ptr addrspace(1) %out
  %first = call i1 @air.simd_is_first()
  %z = zext i1 %first to i32
  store volatile i32 %z, ptr addrspace(1) %out
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_name", !"args"}
