; A call of a function AIR does not have is an error in either form.
; RUN: not opt -mtriple=air -passes=air-verify -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -mtriple=air -passes=air-verify-metal -disable-output %s 2>&1 | FileCheck %s
; CHECK: a call of 'air.copysign.f32', which AIR does not have
; CHECK-NOT: air.sin.f32
target triple = "air"
declare float @air.copysign.f32(float, float)
declare float @air.sin.f32(float)
define float @f(float %x) {
  %y = call float @air.copysign.f32(float %x, float %x)
  %z = call float @air.sin.f32(float %y)
  ret float %z
}
