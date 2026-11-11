; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; lround / llround / lrint / llrint (no libcalls on AGX: "no libcall
; available", and half could not be selected) are the integral value
; converted and frozen (an out-of-range operand gives an unspecified value,
; not poison); double ones in AGXSoftF64, before its lowering.

; CHECK-LABEL: @lround_f32(
; CHECK: [[R:%.*]] = call float @llvm.round.f32(float %x)
; CHECK-NEXT: [[I:%.*]] = fptosi float [[R]] to i32
; CHECK-NEXT: freeze i32 [[I]]
define agx_kernel void @lround_f32(ptr addrspace(1) %p, float %x) {
  %r = call i32 @llvm.lround.i32.f32(float %x)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @llrint_f16(
; CHECK: call half @llvm.rint.f16(half %x)
; CHECK-NEXT: fptosi half {{.*}} to i64
; CHECK-NEXT: freeze i64
define agx_kernel void @llrint_f16(ptr addrspace(1) %p, half %x) {
  %r = call i64 @llvm.llrint.i64.f16(half %x)
  store i64 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @llround_f64(
; CHECK-NOT: llround
; CHECK: freeze i64
define agx_kernel void @llround_f64(ptr addrspace(1) %p, double %x) {
  %r = call i64 @llvm.llround.i64.f64(double %x)
  store i64 %r, ptr addrspace(1) %p
  ret void
}
