; In a function that flushes half denormals, the encodings a minimumnum combines for equal operands
; are of the flushed operands: the OR of a zero and a subnormal must not keep the subnormal.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @ftz(
; CHECK: %[[A:.*]] = and i16 %{{.*}}, 32767
; CHECK: icmp ult i16 %[[A]], 1024
; CHECK: and i16 %{{.*}}, -32768
; CHECK: or i16
define void @ftz(ptr addrspace(1) %o, half %x, half %y) "air-kernel" denormal_fpenv(preservesign) {
  %r = call half @llvm.minimumnum.f16(half %x, half %y)
  store half %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: define void @keep(
; CHECK-NOT: icmp ult i16
; CHECK: or i16
define void @keep(ptr addrspace(1) %o, half %x, half %y) "air-kernel" {
  %r = call half @llvm.minimumnum.f16(half %x, half %y)
  store half %r, ptr addrspace(1) %o
  ret void
}

declare half @llvm.minimumnum.f16(half, half)
