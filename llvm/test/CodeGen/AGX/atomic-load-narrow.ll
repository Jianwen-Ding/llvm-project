; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-expand-wide-atomics < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; 8- and 16-bit atomic loads read their aligned 32-bit word atomically (the
; compare-and-swap AtomicExpand makes of an atomic load is 32-bit then; an 8-
; or 16-bit one could not be selected).

; CHECK-LABEL: @ld16(
; CHECK: load atomic volatile i32, ptr addrspace(1) %{{.*}} monotonic, align 4
; CHECK: lshr i32
; CHECK: trunc i32 %{{.*}} to i16
define agx_kernel void @ld16(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %v = load atomic volatile i16, ptr addrspace(1) %p monotonic, align 2
  store i16 %v, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @ld8(
; CHECK: load atomic i32, ptr addrspace(3) %{{.*}} acquire, align 4
; CHECK: trunc i32 %{{.*}} to i8
define agx_kernel void @ld8(ptr addrspace(3) %p, ptr addrspace(1) %o) {
  %v = load atomic i8, ptr addrspace(3) %p acquire, align 1
  store i8 %v, ptr addrspace(1) %o
  ret void
}
