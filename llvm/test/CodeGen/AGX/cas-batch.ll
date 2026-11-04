; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-expand-wide-atomics < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; 32-bit read-modify-writes that are compare-and-swap loops: a SIMD group
; whose active threads all access one address forms a batch (the lowest
; reads, the group applies the operations in lane order, the lowest
; compare-and-swaps the result, the batch retrying together). Only for a
; uniform address outside divergent branches: a divergent address, or an
; atomic under a divergent branch (a reduction's one atomic), keeps the
; instruction (AtomicExpand's loop).

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

; a 16-bit operation batches on its aligned word, each thread's operation on
; its half
; CHECK-LABEL: @add_f16(
; CHECK: cas.batch.op:
; CHECK: lshr i32
; CHECK: fadd half
; CHECK: shl i32 65535
; CHECK: cas.batch.swap:
; CHECK: cmpxchg ptr addrspace(1) %{{.*}}, i32
; CHECK: atomicrmw fadd ptr addrspace(1) %p, half %v monotonic
define agx_kernel void @add_f16(ptr addrspace(1) %p, half %v, ptr addrspace(1) %out) {
  %o = atomicrmw fadd ptr addrspace(1) %p, half %v monotonic, align 2
  store half %o, ptr addrspace(1) %out
  ret void
}

; a 16-bit compare-and-swap (a partword loop) batches on its aligned word too:
; each thread's half compared with what it expects
; CHECK-LABEL: @cas16(
; CHECK: cas.batch.op:
; CHECK: icmp eq i16
; CHECK: cas.batch.swap:
; CHECK: cmpxchg ptr addrspace(1) %{{.*}}, i32
; CHECK: cmpxchg ptr addrspace(1) %p, i16 %a, i16 %b monotonic monotonic
define agx_kernel void @cas16(ptr addrspace(1) %p, i16 %a, i16 %b, ptr addrspace(1) %out) {
  %r = cmpxchg ptr addrspace(1) %p, i16 %a, i16 %b monotonic monotonic, align 2
  %v = extractvalue { i16, i1 } %r, 0
  store i16 %v, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @divergent_address(
; CHECK-NOT: cas.batch
; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %q, i32 %v monotonic
; CHECK: ret void
define agx_kernel void @divergent_address(ptr addrspace(1) %p, i32 %v) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %o = atomicrmw uinc_wrap ptr addrspace(1) %q, i32 %v monotonic, align 4
  ret void
}

; CHECK-LABEL: @one_thread(
; CHECK-NOT: cas.batch
; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v monotonic
; CHECK: ret void
define agx_kernel void @one_thread(ptr addrspace(1) %p, i32 %v) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %c = icmp eq i32 %t, 0
  br i1 %c, label %a, label %e
a:
  %o = atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v monotonic, align 4
  br label %e
e:
  ret void
}
