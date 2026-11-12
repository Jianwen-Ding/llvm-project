; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-aggregate-atomics=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=IEEE

; Atomic additions on device memory: the threads of a SIMD group on the address
; of the lowest pending thread add their sum with one atomic (Cycles' film
; accumulation: many threads of a group on one pixel), a few addresses at a
; time, the rest one by one.

; CHECK-LABEL: film:
; CHECK:         icmp_ballot
; CHECK:         simd_shuffle
; CHECK:         popcount
; CHECK:         simd_fadd
; CHECK:         atomic sub, 0, _,
; CHECK:         jmp_exec_any
; CHECK:         atomic sub, 0, _,
; OFF-LABEL: film:
; OFF-NOT:       simd_fadd
; OFF:           atomic sub, 0, _,
; OFF-NOT:       atomic sub
; IEEE-LABEL: film:
; IEEE-NOT:      simd_fadd
define agx_kernel void @film(ptr addrspace(1) %buf, ptr addrspace(1) %x) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pix = lshr i32 %i, 4
  %p = getelementptr float, ptr addrspace(1) %buf, i32 %pix
  %q = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %q
  %r = atomicrmw fadd ptr addrspace(1) %p, float %v monotonic
  ret void
}

; A result: the leader's old value plus the exclusive scan of the group.
; CHECK-LABEL: slot:
; CHECK:         simd_iadd
; CHECK:         simd_prefix_iadd
; CHECK:         atomic add, 0,
; CHECK:         simd_shuffle
; IEEE-LABEL: slot:
; IEEE:          simd_iadd
define agx_kernel void @slot(ptr addrspace(1) %cnt, ptr addrspace(1) %out) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %k = and i32 %i, 3
  %p = getelementptr i32, ptr addrspace(1) %cnt, i32 %k
  %s = atomicrmw add ptr addrspace(1) %p, i32 1 monotonic
  %o = getelementptr i32, ptr addrspace(1) %out, i32 %s
  store i32 %i, ptr addrspace(1) %o
  ret void
}

; A uniform address: one round, no address test.
; CHECK-LABEL: total:
; CHECK-NOT:     popcount
; CHECK:         simd_iadd
; CHECK:         atomic add, 0, _,
; CHECK-NOT:     atomic add
define agx_kernel void @total(ptr addrspace(1) %t, ptr addrspace(1) %x) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %x, i32 %i
  %v = load i32, ptr addrspace(1) %q
  %r = atomicrmw add ptr addrspace(1) %t, i32 %v monotonic
  ret void
}

; Threadgroup memory, other operations and stronger orderings stay.
; CHECK-LABEL: kept:
; CHECK-NOT:     simd_
; CHECK:         stop
define agx_kernel void @kept(ptr addrspace(3) %s, ptr addrspace(1) %d, i32 %v) {
  %a = atomicrmw add ptr addrspace(3) %s, i32 %v monotonic
  %b = atomicrmw max ptr addrspace(1) %d, i32 %v monotonic
  %c = atomicrmw add ptr addrspace(1) %d, i32 %v seq_cst
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
