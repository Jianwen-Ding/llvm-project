; The 64-bit atomics the Metal pipeline emulates under a spin lock
; (AIRLowerAtomicRMW: Metal's atomics are 32-bit) are atomicrmw of i64 again,
; which the AGX backend compiles with its own lock inside the loop that takes
; it, or natively (G14's u64 min and max without a result): the optimizer
; would otherwise move the emulation's critical section after its loop, where
; the threads of a SIMD-group waiting for a lock one of them holds never let
; it run.
; RUN: llc -mtriple=air %s -o %t.metal.ll
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %t.metal.ll | FileCheck %s
; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: llvm-air2agx -mcpu=agx-g14g --emit=asm %t.metallib -o - | FileCheck %s --check-prefix=ASM

; CHECK-LABEL: define agx_kernel void @add64(
; CHECK: %[[R:.*]] = atomicrmw volatile add ptr addrspace(1) %{{.*}}, i64 %{{.*}} syncscope("device") monotonic, align 8
; CHECK-NOT: cmpxchg
; CHECK: store i64 %[[R]]
; CHECK-LABEL: define agx_kernel void @umax64(
; CHECK: atomicrmw volatile umax ptr addrspace(1) %{{.*}}, i64 %{{.*}} syncscope("device") monotonic, align 8
; CHECK-NOT: cmpxchg
; CHECK-LABEL: define agx_kernel void @xchg64(
; CHECK: atomicrmw volatile xchg ptr addrspace(1) %{{.*}}, i64 %{{.*}} syncscope("device") monotonic, align 8
; CHECK-NOT: cmpxchg

; The native u64 max: no lock.
; ASM-LABEL: umax64:
; ASM-NOT: atomic cmpxchg, 0, {{.*}}, 0, signed, r{{[0-9]+}}_r{{[0-9]+}}, 0
; ASM: stop

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @add64(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %i = extractvalue { i32, i32, i32 } %t, 0
  %p = getelementptr i64, ptr addrspace(1) %in, i32 %i
  %v = load i64, ptr addrspace(1) %p
  %s = and i32 %i, 7
  %q = getelementptr i64, ptr addrspace(1) %out, i32 %s
  %old = atomicrmw add ptr addrspace(1) %q, i64 %v monotonic
  store i64 %old, ptr addrspace(1) %p
  ret void
}

define void @umax64(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %i = extractvalue { i32, i32, i32 } %t, 0
  %p = getelementptr i64, ptr addrspace(1) %in, i32 %i
  %v = load i64, ptr addrspace(1) %p
  %q = getelementptr i64, ptr addrspace(1) %out, i32 0
  %old = atomicrmw umax ptr addrspace(1) %q, i64 %v monotonic
  ret void
}

define void @xchg64(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %i = extractvalue { i32, i32, i32 } %t, 0
  %p = getelementptr i64, ptr addrspace(1) %in, i32 %i
  %v = load i64, ptr addrspace(1) %p
  %old = atomicrmw xchg ptr addrspace(1) %out, i64 %v monotonic
  ret void
}
