; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-expand-wide-atomics < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; A 32-bit uinc_wrap / udec_wrap whose SIMD group's active threads all access
; one address: the lowest thread reads, the group applies the operations in
; lane order, the lowest thread compare-and-swaps the result (the group
; retrying together); other groups keep the instruction (AtomicExpand's loop).

; CHECK-LABEL: @inc(
; CHECK: call i32 @llvm.agx.simd.ballot(i1 true)
; CHECK: inc.dec.batch.op:
; CHECK: icmp uge i32
; CHECK: cmpxchg ptr addrspace(1) %p
; CHECK: inc.dec.single:
; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v monotonic
define agx_kernel void @inc(ptr addrspace(1) %p, i32 %v, ptr addrspace(1) %out) {
  %o = atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v monotonic, align 4
  store i32 %o, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @dec(
; CHECK: inc.dec.batch.op:
; CHECK: atomicrmw udec_wrap ptr addrspace(1) %p, i32 %v monotonic
define agx_kernel void @dec(ptr addrspace(1) %p, i32 %v, ptr addrspace(1) %out) {
  %o = atomicrmw udec_wrap ptr addrspace(1) %p, i32 %v monotonic, align 4
  store i32 %o, ptr addrspace(1) %out
  ret void
}

; stronger orderings keep the instruction
; CHECK-LABEL: @inc_seq(
; CHECK-NOT: inc.dec.batch
; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v seq_cst
define agx_kernel void @inc_seq(ptr addrspace(1) %p, i32 %v) {
  %o = atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v seq_cst, align 4
  ret void
}
