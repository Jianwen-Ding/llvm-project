; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-codegenprepare -agx-newton-div64=false < %s \
; RUN:   | FileCheck %s --check-prefix=LOOP
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; 64-bit division and remainder whose operands do not fit in 32 bits: a
; reciprocal estimate of 2^64 / y below the exact value, two Newton-Raphson
; steps with 64-bit multiply-highs, the quotient and two corrections, without
; control flow, instead of the 64-iteration bit loop.

; CHECK-LABEL: define agx_kernel void @udiv64(
; CHECK:         fmul float %{{.*}}, f0x5F7FFFE0
; CHECK-COUNT-3: mul i128
; CHECK-COUNT-2: icmp uge i64
; CHECK-NOT:     br i1
; CHECK-NOT:     udiv-do-while
; LOOP-LABEL:  define agx_kernel void @udiv64(
; LOOP:        udiv-do-while:
define agx_kernel void @udiv64(ptr addrspace(1) %p, i64 %a, i64 %b) {
  %q = udiv i64 %a, %b
  store i64 %q, ptr addrspace(1) %p
  ret void
}

; Signed remainder: the magnitudes divided, the sign of the dividend; by zero
; the dividend.
; CHECK-LABEL: define agx_kernel void @srem64(
; CHECK:         call float @llvm.agx.rcp.f32(
; CHECK-COUNT-2: icmp uge i64
; CHECK:         select i1 %{{.*}}, i64 %a,
define agx_kernel void @srem64(ptr addrspace(1) %p, i64 %a, i64 %b) {
  %r = srem i64 %a, %b
  store i64 %r, ptr addrspace(1) %p
  ret void
}
