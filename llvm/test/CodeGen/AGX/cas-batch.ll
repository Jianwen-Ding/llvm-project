; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-expand-wide-atomics < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; 32-bit read-modify-writes that are compare-and-swap loops: a SIMD group
; whose active threads all access one address forms a batch (the lowest
; reads, the group applies the operations in lane order, the lowest
; compare-and-swaps the result, the batch retrying together); groups on
; different addresses keep the instruction (AtomicExpand's loop).

; CHECK-LABEL: @inc(
; CHECK: cas.batch.failed:
; CHECK: call i32 @llvm.agx.simd.ballot(i1 true)
; CHECK: cas.batch.op:
; CHECK: icmp uge i32
; CHECK: cas.batch.swap:
; CHECK: cmpxchg ptr addrspace(1) %p
; CHECK: cas.batch.single:
; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v monotonic
define agx_kernel void @inc(ptr addrspace(1) %p, i32 %v, ptr addrspace(1) %out) {
  %o = atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v monotonic, align 4
  store i32 %o, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @add_f16x2(
; CHECK: cas.batch.op:
; CHECK: fadd <2 x half>
; CHECK: atomicrmw fadd ptr addrspace(1) %p, <2 x half> %v monotonic
define agx_kernel void @add_f16x2(ptr addrspace(1) %p, <2 x half> %v, ptr addrspace(1) %out) {
  %o = atomicrmw fadd ptr addrspace(1) %p, <2 x half> %v monotonic, align 4
  store <2 x half> %o, ptr addrspace(1) %out
  ret void
}

; a native atomic and stronger orderings keep the instruction
; CHECK-LABEL: @add_i32(
; CHECK-NOT: cas.batch
; CHECK: atomicrmw add
define agx_kernel void @add_i32(ptr addrspace(1) %p, i32 %v) {
  %o = atomicrmw add ptr addrspace(1) %p, i32 %v monotonic, align 4
  ret void
}
; CHECK-LABEL: @inc_seq(
; CHECK-NOT: cas.batch
; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v seq_cst
define agx_kernel void @inc_seq(ptr addrspace(1) %p, i32 %v) {
  %o = atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v seq_cst, align 4
  ret void
}
