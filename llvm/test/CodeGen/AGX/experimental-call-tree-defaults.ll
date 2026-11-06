; A kernel with real calls and what it calls are capped at 72 registers by default
; (-agx-call-tree-max-gprs), which the kernel declares instead of all 128.
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-call-tree-max-gprs=0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NOCAP

target triple = "agx-unknown-unknown"
declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: kernel:
; CHECK: general purpose registers: 72 x 32-bit
; NOCAP-LABEL: kernel:
; NOCAP: general purpose registers: 128 x 32-bit
define agx_kernel void @kernel(ptr addrspace(1) %p) {
  %id = call i32 @llvm.agx.thread.position.in.grid.x()
  %called = call i32 @leaf(i32 %id)
  store volatile i32 %called, ptr addrspace(1) %p
  ret void
}

define internal i32 @leaf(i32 %x) noinline {
  %y = mul i32 %x, %x
  %z = add i32 %y, 7
  ret i32 %z
}
