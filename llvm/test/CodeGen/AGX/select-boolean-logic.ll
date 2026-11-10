; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs \
; RUN:   -agx-select-boolean-logic=0 < %s | FileCheck %s --check-prefix=OFF

; and / or of a comparison and a 0/1 value: the comparison's select of the
; value and 0 (or of 1 and the value), not its 0/1 and then the and / or.

; CHECK-LABEL: and_cmp:
; CHECK: fcmpsel eq, [[B:r[0-9]+]], {{.*}}, 1, 0
; CHECK-NEXT: icmpsel ult, {{r[0-9]+}}, {{.*}}, [[B]], 0
; CHECK-NEXT: device_store
; OFF-LABEL: and_cmp:
; OFF: and	r
define agx_kernel void @and_cmp(ptr addrspace(1) %p, i32 %a, i32 %b, float %x, float %y) {
  %c1 = icmp ult i32 %a, %b
  %c2 = fcmp oeq float %x, %y
  %c = and i1 %c1, %c2
  %z = zext i1 %c to i32
  store i32 %z, ptr addrspace(1) %p, align 4
  ret void
}

; CHECK-LABEL: or_cmp:
; CHECK: icmpsel ueq, [[B:r[0-9]+]], {{.*}}, 7, 1, 0
; CHECK-NEXT: icmpsel slt, {{r[0-9]+}}, {{.*}}, 1, [[B]]
; CHECK-NEXT: device_store
; OFF-LABEL: or_cmp:
; OFF: or	r
define agx_kernel void @or_cmp(ptr addrspace(1) %p, i32 %a, i32 %b, i32 %c) {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp eq i32 %c, 7
  %o = or i1 %c1, %c2
  %z = zext i1 %o to i32
  store i32 %z, ptr addrspace(1) %p, align 4
  ret void
}
