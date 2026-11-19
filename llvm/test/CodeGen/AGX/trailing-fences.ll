; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-remove-trailing-fences=0 < %s | FileCheck %s --check-prefix=KEEP

; A kernel's fence (and an acquire's wait) that no memory access, barrier or
; call follows on any path to its end orders nothing (L15-E438): an acq_rel
; atomic's acquire at the end of a histogram goes, its release before stays.

; CHECK-LABEL: hist:
; CHECK:       memory_barrier 0, 3, 3
; CHECK:       atomic add
; CHECK-NEXT:  stop
; KEEP-LABEL:  hist:
; KEEP:        atomic add
; KEEP-NEXT:   wait 0
; KEEP-NEXT:   memory_barrier 3, 1, 9
define agx_kernel void @hist(ptr addrspace(1) %h, i32 %b) {
  %p = getelementptr inbounds i32, ptr addrspace(1) %h, i32 %b
  %o = atomicrmw add ptr addrspace(1) %p, i32 1 acq_rel, align 4
  ret void
}

; A store after the join, on the atomic's path: the acquire stays.
; CHECK-LABEL: store_after:
; CHECK:       atomic add
; CHECK:       memory_barrier 3, 1, 9
; CHECK:       device_store
define agx_kernel void @store_after(ptr addrspace(1) %h, ptr addrspace(1) %o, i32 %b) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %t, %b
  br i1 %c, label %then, label %join
then:
  %p = getelementptr inbounds i32, ptr addrspace(1) %h, i32 %t
  %a = atomicrmw add ptr addrspace(1) %p, i32 1 acq_rel, align 4
  br label %join
join:
  %q = getelementptr inbounds i32, ptr addrspace(1) %o, i32 %t
  store i32 %t, ptr addrspace(1) %q, align 4
  ret void
}

; A barrier after the fence: the fence orders the store for the threads
; past the barrier, and stays.
; CHECK-LABEL: barrier_after:
; CHECK:       device_store
; CHECK:       memory_barrier
; CHECK:       threadgroup_barrier
define agx_kernel void @barrier_after(ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr inbounds i32, ptr addrspace(1) %o, i32 %t
  store i32 %t, ptr addrspace(1) %q, align 4
  fence release
  call void @llvm.agx.threadgroup.barrier()
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare void @llvm.agx.threadgroup.barrier()
