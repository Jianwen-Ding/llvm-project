; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-call-tree-max-gprs=0 -agx-declare-tree-gprs -verify-machineinstrs < %s -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-call-tree-max-gprs=0 -verify-machineinstrs < %s -o - | FileCheck %s --check-prefix=ALL
; With -agx-declare-tree-gprs a kernel's record has the registers its call tree
; uses, even with its callee emitted after it; otherwise every register.

target triple = "agx-unknown-unknown"
declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @kernel(ptr addrspace(1) %p) {
  %id = call i32 @llvm.agx.thread.position.in.grid.x()
  %called = call i32 @leaf(i32 %id)
  store volatile i32 %called, ptr addrspace(1) %p
  ret void
}

define internal i32 @leaf(i32 %x) noinline {
  %r = add i32 %x, 5
  ret i32 %r
}

; The word after name, uniforms and two halves: small, not 128.
; CHECK: .section .agx.kernels
; CHECK-NOT: .long 128
; ALL: .section .agx.kernels
; ALL: .long 128
