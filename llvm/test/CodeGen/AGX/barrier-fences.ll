; A threadgroup barrier between two threadgroup fences (AIR's air.wg.barrier
; and the gpu dialect's gpu.barrier, as agx-lower-air writes them) keeps the
; fence before it, as CUDA's __syncthreads has it; -agx-merge-barrier-fences=0
; keeps both.
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-merge-barrier-fences=0 < %s | FileCheck %s --check-prefix=OFF

; CHECK-LABEL: exchange:
; CHECK:      threadgroup_store
; CHECK:      memory_barrier 1, 2, 9
; CHECK-NEXT: threadgroup_barrier
; CHECK-NOT:  memory_barrier
; CHECK:      threadgroup_load
; OFF-LABEL:  exchange:
; OFF:        memory_barrier 1, 2, 9
; OFF-NEXT:   threadgroup_barrier
; OFF-NEXT:   memory_barrier 1, 2, 9

; A fence after the barrier without one before it stays.
; CHECK-LABEL: after_only:
; CHECK:      threadgroup_barrier
; CHECK-NEXT: memory_barrier 1, 2, 9

target triple = "agx"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare void @llvm.agx.threadgroup.barrier()

define agx_kernel void @exchange(ptr addrspace(1) %out, i32 %v) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %m = and i32 %t, 63
  %p = getelementptr i32, ptr addrspace(3) @tg, i32 %m
  store i32 %v, ptr addrspace(3) %p
  fence syncscope("workgroup") release
  call void @llvm.agx.threadgroup.barrier()
  fence syncscope("workgroup") acquire
  %n = xor i32 %m, 1
  %q = getelementptr i32, ptr addrspace(3) @tg, i32 %n
  %x = load i32, ptr addrspace(3) %q
  %o = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %x, ptr addrspace(1) %o
  ret void
}

define agx_kernel void @after_only(ptr addrspace(1) %out, i32 %v) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %m = and i32 %t, 63
  %p = getelementptr i32, ptr addrspace(3) @tg, i32 %m
  store i32 %v, ptr addrspace(3) %p
  call void @llvm.agx.threadgroup.barrier()
  fence syncscope("workgroup") acquire
  %n = xor i32 %m, 1
  %q = getelementptr i32, ptr addrspace(3) @tg, i32 %n
  %x = load i32, ptr addrspace(3) %q
  %o = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %x, ptr addrspace(1) %o
  ret void
}
