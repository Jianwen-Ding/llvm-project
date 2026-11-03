; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-expand-wide-atomics < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-expand-wide-atomics -agx-wide-atomics-batch=false < %s \
; RUN:   | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; A 64-bit read-modify-write: when the SIMD group's active threads all access
; one address, the lowest takes the lock once and every thread applies the
; operations in lane order; otherwise each thread takes the lock.

; CHECK-LABEL: define agx_kernel void @add64(
; CHECK:         call i32 @llvm.agx.simd.ballot(i1 true)
; CHECK:         br i1 %{{.*}}, label %wide.atomic.batch.check, label %wide.atomic
; CHECK:       wide.atomic.batch.lock:
; CHECK:         cmpxchg ptr addrspace(1)
; CHECK:       wide.atomic.batch.op:
; CHECK:         call i32 @llvm.cttz.i32
; CHECK:         call i32 @llvm.agx.simd.shuffle.i32
; CHECK:         add i64
; CHECK:       wide.atomic.batch.write:
; OFF-NOT:     wide.atomic.batch
define agx_kernel void @add64(ptr addrspace(1) %p, i64 %v, ptr addrspace(1) %out) {
  %old = atomicrmw add ptr addrspace(1) %p, i64 %v monotonic, align 8
  store i64 %old, ptr addrspace(1) %out
  ret void
}

; A compare-and-swap: each thread's comparison against the value the
; previous ones left.
; CHECK-LABEL: define agx_kernel void @cas64(
; CHECK:       wide.atomic.batch.op:
; CHECK:         icmp eq i64
; CHECK:         select i1
define agx_kernel void @cas64(ptr addrspace(1) %p, i64 %c, i64 %n, ptr addrspace(1) %out) {
  %r = cmpxchg ptr addrspace(1) %p, i64 %c, i64 %n monotonic monotonic, align 8
  %old = extractvalue { i64, i1 } %r, 0
  store i64 %old, ptr addrspace(1) %out
  ret void
}
