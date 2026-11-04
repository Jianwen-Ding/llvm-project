; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -stop-after=agx-strict-fp < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -verify-machineinstrs < %s -o /dev/null

; A float rounded to half in a constant direction: the hardware's conversion
; to nearest, then one unit toward the direction when it went past the float
; (a float denormal, which the hardware flushes, becomes a signed zero or the
; smallest half denormal). No integer conversion of the float is needed.

; CHECK-LABEL: @rz(
; CHECK: fptrunc float %a to half
; CHECK: fpext half
; CHECK: sub i16
; CHECK: fcmp ogt float
; CHECK: icmp ult i32 {{.*}}, 8388608
; CHECK-NOT: lshr
; CHECK: ret void
define agx_kernel void @rz(ptr addrspace(1) %p, float %a) #0 {
  %r = call half @llvm.experimental.constrained.fptrunc.f16.f32(float %a, metadata !"round.towardzero", metadata !"fpexcept.ignore") #0
  store half %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @ru(
; CHECK: fptrunc float %a to half
; CHECK: fcmp olt float
; CHECK-NOT: lshr
; CHECK: ret void
define agx_kernel void @ru(ptr addrspace(1) %p, float %a) #0 {
  %r = call half @llvm.experimental.constrained.fptrunc.f16.f32(float %a, metadata !"round.upward", metadata !"fpexcept.ignore") #0
  store half %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @rd(
; CHECK: fptrunc float %a to half
; CHECK: fcmp ogt float
; CHECK-NOT: lshr
; CHECK: ret void
define agx_kernel void @rd(ptr addrspace(1) %p, float %a) #0 {
  %r = call half @llvm.experimental.constrained.fptrunc.f16.f32(float %a, metadata !"round.downward", metadata !"fpexcept.ignore") #0
  store half %r, ptr addrspace(1) %p
  ret void
}

declare half @llvm.experimental.constrained.fptrunc.f16.f32(float, metadata, metadata)
attributes #0 = { strictfp }
