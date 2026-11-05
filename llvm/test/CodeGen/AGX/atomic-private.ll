; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=atomic-expand < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; Atomic accesses of private memory (ld.volatile.local, st.volatile.local) belong to one
; thread: plain accesses that stay volatile. As atomics no instruction could be selected.

; CHECK-LABEL: @private32(
; CHECK: store volatile i32 %v, ptr addrspace(5) %p, align 4
; CHECK: load volatile i32, ptr addrspace(5) %p, align 4
define agx_kernel void @private32(i32 %v, ptr addrspace(1) %o) {
  %p = alloca i32, align 4, addrspace(5)
  store atomic volatile i32 %v, ptr addrspace(5) %p syncscope("one-as") monotonic, align 4
  %r = load atomic volatile i32, ptr addrspace(5) %p syncscope("one-as") monotonic, align 4
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @private_widths(
; CHECK-NOT: atomic
; CHECK: ret void
define agx_kernel void @private_widths(i64 %v, i16 %h, i8 %b, ptr addrspace(1) %o) {
  %p = alloca i64, align 8, addrspace(5)
  store atomic volatile i64 %v, ptr addrspace(5) %p monotonic, align 8
  %r = load atomic volatile i64, ptr addrspace(5) %p acquire, align 8
  store i64 %r, ptr addrspace(1) %o
  store atomic volatile i16 %h, ptr addrspace(5) %p release, align 2
  %s = load atomic volatile i16, ptr addrspace(5) %p monotonic, align 2
  store i16 %s, ptr addrspace(1) %o
  store atomic volatile i8 %b, ptr addrspace(5) %p monotonic, align 1
  %t = load atomic volatile i8, ptr addrspace(5) %p monotonic, align 1
  store i8 %t, ptr addrspace(1) %o
  ret void
}
