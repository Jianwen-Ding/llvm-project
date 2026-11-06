; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-callee-saved=48 -agx-call-tree-max-gprs=0 -verify-machineinstrs < %s | FileCheck %s
; Called functions keep r80 and up. What is live across a call stays in such a
; register, with no spill in the caller. A callee that uses them saves them on
; entry and restores them before it returns, and a kernel with calls declares
; all registers, as its callees may use any.

target triple = "agx-unknown-unknown"
declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: kernel:
; CHECK: general purpose registers: 128 x 32-bit
; CHECK-NOT: stack_store
; CHECK: get_sr r[[KEPT:[0-9]+]], sr80
; CHECK-NOT: stack_store
; CHECK: call leaf
; CHECK-NEXT: pop_exec r0l, 2
; CHECK-NOT: stack_load
; CHECK: iadd {{.*}}r[[KEPT]]
; CHECK: stop
define agx_kernel void @kernel(ptr addrspace(1) %p) {
  %id = call i32 @llvm.agx.thread.position.in.grid.x()
  %called = call i32 @leaf(i32 3)
  %sum = add i32 %called, %id
  store volatile i32 %sum, ptr addrspace(1) %p
  ret void
}

; A leaf with few registers saves nothing.
; CHECK-LABEL: leaf:
; CHECK-NOT: stack_store
; CHECK: ret r1
define internal i32 @leaf(i32 %x) noinline {
  %r = add i32 %x, 5
  ret i32 %r
}

attributes #0 = { noinline }
