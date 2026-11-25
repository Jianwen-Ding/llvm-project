; RUN: llc -mtriple=agx -mcpu=agx-g14c -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -o /dev/null -print-after=agx-soft-f64 -agx-soft-f64-mul-direct=0 -agx-soft-f64-add-frame3=0 -agx-soft-f64-div-direct=0 < %s 2>&1 | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; The double short paths' rounding: the exponent kept as its field in the
; high word (masks with 0x7ff00000, a normal result's (E - 1) << 20 below
; 2045 << 20), a product's significand from two funnel shifts of its words
; (L15-E664), a sum's significands 3 places up with the implicit bit ORed
; (L15-E666), a quotient's significand from its digits, (q1 << 26) +
; (low >> 2) (L15-E665).

; CHECK-LABEL: @mul(
; CHECK: call i32 @llvm.fshr.i32
; CHECK: call i32 @llvm.fshr.i32
; CHECK: and i32 {{%.*}}, 2146435072
; CHECK: icmp ult i32 {{%.*}}, 2144337920
; CHECK: br i1
; OFF-LABEL: @mul(
; OFF-NOT: @llvm.fshr.i32
; OFF: br i1
define agx_kernel void @mul(ptr addrspace(1) %p, double %x, double %y) {
  %r = fmul double %x, %y
  store double %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @add(
; CHECK: [[A:%.*]] = shl i64 {{%.*}}, 3
; CHECK-NEXT: or i64 [[A]], 36028797018963968
; CHECK: icmp ult i32 {{%.*}}, 2144337920
; CHECK: br i1
; OFF-LABEL: @add(
; OFF-NOT: 36028797018963968
; OFF: br i1
define agx_kernel void @add(ptr addrspace(1) %p, double %x, double %y) {
  %r = fadd double %x, %y
  store double %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @quot(
; CHECK: shl i64 {{%.*}}, 26
; CHECK: lshr i32 {{%.*}}, 2
; CHECK: icmp ult i32 {{%.*}}, 2144337920
; CHECK: br i1
; OFF-LABEL: @quot(
; OFF-NOT: icmp ult i32 {{%.*}}, 2144337920
; OFF: br i1
define agx_kernel void @quot(ptr addrspace(1) %p, double %x, double %y) {
  %r = fdiv double %x, %y
  store double %r, ptr addrspace(1) %p
  ret void
}
