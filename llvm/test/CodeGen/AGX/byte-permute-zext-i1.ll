; RUN: llc -mtriple=agx -mcpu=agx-g14c -O0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
;
; A byte tree's zero extension of a value narrower than a byte (an i1) is not
; a zero: (x & 255) | zext(i1 true) keeps its or (agx-cf-fuzz seed 6210,
; L15-E467: byteSources gave the i1 no bytes, and the or lost it).

; CHECK-LABEL: k:
; CHECK:       device_load 0, i8,
; CHECK:       or {{r[0-9]+}}, {{r[0-9]+}}, 1
; CHECK:       device_store
define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {
entry:
  %x0 = load i32, ptr addrspace(1) %p, align 4
  %x0.masked = and i32 %x0, 255
  %t4 = zext i1 true to i32
  %t = or i32 %x0.masked, %t4
  store i32 %t, ptr addrspace(1) %q, align 4
  ret void
}
