; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-high-half-compares=false < %s | FileCheck %s --check-prefix=OFF

; An ordered 64-bit comparison with a constant whose low half fixes the low
; comparison to what the condition gives for equal values (x < c with c's
; low half 0, x <= c with it all ones, > and >= alike, signed or unsigned)
; is the comparison of the high halves alone: one icmpsel, not two.

; CHECK-LABEL: sgtm1:
; CHECK: icmpsel sgt
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: sgtm1:
; OFF-COUNT-2: icmpsel
define agx_kernel void @sgtm1(ptr addrspace(1) %p, i64 %x) {
  %c = icmp sgt i64 %x, -1
  %r = zext i1 %c to i32
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: slt:
; CHECK: icmpsel slt, r{{[0-9]+}}, u{{[0-9]+}}, 5, 1, 0
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: slt:
; OFF-COUNT-2: icmpsel
define agx_kernel void @slt(ptr addrspace(1) %p, i64 %x) {
  %c = icmp slt i64 %x, 21474836480
  %r = zext i1 %c to i32
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: ule:
; CHECK: icmpsel ult, r{{[0-9]+}}, u{{[0-9]+}}, 4, 1, 0
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: ule:
; OFF-COUNT-2: icmpsel
define agx_kernel void @ule(ptr addrspace(1) %p, i64 %x) {
  %c = icmp ule i64 %x, 17179869183
  %r = zext i1 %c to i32
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: uge:
; CHECK: icmpsel ugt
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: uge:
; OFF-COUNT-2: icmpsel
define agx_kernel void @uge(ptr addrspace(1) %p, i64 %x) {
  %c = icmp uge i64 %x, -8589934592
  %r = zext i1 %c to i32
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: sge:
; CHECK: icmpsel sgt
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: sge:
; OFF-COUNT-2: icmpsel
define agx_kernel void @sge(ptr addrspace(1) %p, i64 %x) {
  %c = icmp sge i64 %x, -4294967296
  %r = zext i1 %c to i32
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: ugt:
; CHECK: icmpsel ugt, r{{[0-9]+}}, u{{[0-9]+}}, 6, 1, 0
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: ugt:
; OFF-COUNT-2: icmpsel
define agx_kernel void @ugt(ptr addrspace(1) %p, i64 %x) {
  %c = icmp ugt i64 %x, 30064771071
  %r = zext i1 %c to i32
  store i32 %r, ptr addrspace(1) %p
  ret void
}
