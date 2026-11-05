; A floating point function of AIR declared with another type than its name says.
; RUN: not opt -mtriple=air -passes=air-verify -disable-output %s 2>&1 | FileCheck %s
; CHECK: a call of 'air.sin.f32' with parameters or a result of another type
; CHECK-NOT: air.fast_cos.f32
target triple = "air"
declare i32 @air.sin.f32(i32)
declare float @air.fast_cos.f32(float)
define float @f(i32 %x, float %f) {
  %y = call i32 @air.sin.f32(i32 %x)
  %z = call float @air.fast_cos.f32(float %f)
  ret float %z
}
