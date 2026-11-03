; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-sink-remat -verify-machineinstrs < %s -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o - | FileCheck %s --check-prefix=OFF
; A large constant used in a later block gets its own move there, not in the
; block before the branch. It stays out of a loop deeper than its definition.

target triple = "agx-unknown-unknown"
declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: side:
; OFF-LABEL: side:
define agx_kernel void @side(ptr addrspace(1) %p) {
  %id = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp eq i32 %id, 7
  br i1 %c, label %then, label %end
then:
  %v = mul i32 %id, 1234567
  store volatile i32 %v, ptr addrspace(1) %p
  br label %end
end:
  ret void
}
; CHECK-COUNT-1: 1234567
