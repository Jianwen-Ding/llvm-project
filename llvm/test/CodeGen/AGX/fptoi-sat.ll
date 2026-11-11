; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -verify-machineinstrs < %s | FileCheck %s

; convert saturates and converts NaN to 0, so saturating conversions to 32 or
; 16 bits are one convert. One that saturates to 8 bits is the 16-bit
; convert and an integer clamp (an unsigned one a single icmpsel).

; CHECK-LABEL: sat32:
; CHECK-NOT:     fcmpsel
; CHECK:         convert s32_to_f
; CHECK-NOT:     fcmpsel
; CHECK:         stop
define agx_kernel void @sat32(ptr addrspace(1) %out, float %a, half %h) {
  %s = call i32 @llvm.fptosi.sat.i32.f32(float %a)
  %u = call i32 @llvm.fptoui.sat.i32.f32(float %a)
  %hs = call i32 @llvm.fptosi.sat.i32.f16(half %h)
  %hu = call i32 @llvm.fptoui.sat.i32.f16(half %h)
  %x = add i32 %s, %u
  %y = add i32 %hs, %hu
  %z = xor i32 %x, %y
  %f = sitofp i32 %z to float
  store float %f, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: sat16:
; CHECK-NOT:     fcmpsel
; CHECK:         stop
define agx_kernel void @sat16(ptr addrspace(1) %out, float %a, half %h) {
  %s = call i16 @llvm.fptosi.sat.i16.f32(float %a)
  %u = call i16 @llvm.fptoui.sat.i16.f16(half %h)
  %x = xor i16 %s, %u
  store i16 %x, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: sat8:
; CHECK-NOT:     fcmpsel
; The convert's 8-bit mode saturates to the byte (mode 1, printed by its
; integer-to-float name, with the float-to-integer direction rtz); the byte
; store needs no sign extension.
; CHECK:         convert s8_to_f, {{r[0-9]+l}}, {{u[0-9]+}}, rtz
; CHECK-NOT:     icmpsel
; CHECK:         stop
define agx_kernel void @sat8(ptr addrspace(1) %out, float %a) {
  %s = call i8 @llvm.fptosi.sat.i8.f32(float %a)
  store i8 %s, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: usat8:
; CHECK-NOT:     fcmpsel
; CHECK:         convert u8_to_f, {{r[0-9]+l}}, {{u[0-9]+l}}, rtz
; CHECK-NOT:     icmpsel
; CHECK:         stop
define agx_kernel void @usat8(ptr addrspace(1) %out, half %a) {
  %s = call i8 @llvm.fptoui.sat.i8.f16(half %a)
  store i8 %s, ptr addrspace(1) %out
  ret void
}
declare i8 @llvm.fptoui.sat.i8.f16(half)
