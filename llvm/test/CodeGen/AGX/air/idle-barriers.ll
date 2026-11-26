; A kernel's barrier of threadgroup memory alone orders nothing when no
; threadgroup access has happened since the last barrier (or the kernel's
; entry) on every path to it, or none happens before the next one (or the
; kernel's end) on every path from it (MSL orders device memory by a barrier
; only with mem_device): agx-lower-air drops it, the forward rule first and
; the backward one on what is left. The Metal pipeline's
; air-tg-barrier-insert puts barriers around every threadgroup access.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: opt -mtriple=agx -passes=agx-lower-air -agx-air-idle-barriers=false -S %s | FileCheck %s --check-prefix=OFF

target triple = "air64-apple-macosx14.0.0"

declare void @air.wg.barrier(i32, i32)

@tg = internal addrspace(3) global [64 x i32] undef, align 4

; The barriers before the store and after the load go; the one between stays.
; CHECK-LABEL: define agx_kernel void @k(
; CHECK-NOT: barrier
; CHECK: store i32 %v, ptr addrspace(3)
; CHECK-NEXT: fence syncscope("workgroup") release
; CHECK-NEXT: call void @llvm.agx.threadgroup.barrier()
; CHECK-NEXT: fence syncscope("workgroup") acquire
; CHECK: load i32, ptr addrspace(3)
; CHECK-NOT: barrier
; CHECK: ret void
; OFF-LABEL: define agx_kernel void @k(
; OFF-COUNT-3: call void @llvm.agx.threadgroup.barrier()
define void @k(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p
  %i = and i32 %v, 63
  call void @air.wg.barrier(i32 2, i32 1)
  %s = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %s
  call void @air.wg.barrier(i32 2, i32 1)
  %j = sub i32 63, %i
  %t = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %j
  %r = load i32, ptr addrspace(3) %t
  call void @air.wg.barrier(i32 2, i32 1)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; Two barriers between two accesses: one stays.
; CHECK-LABEL: define agx_kernel void @pair(
; CHECK: store i32 %v, ptr addrspace(3)
; CHECK-COUNT-1: call void @llvm.agx.threadgroup.barrier()
; CHECK-NOT: call void @llvm.agx.threadgroup.barrier()
; CHECK: load i32, ptr addrspace(3)
define void @pair(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p
  %s = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %v
  store i32 %v, ptr addrspace(3) %s
  call void @air.wg.barrier(i32 2, i32 1)
  call void @air.wg.barrier(i32 2, i32 1)
  %r = load i32, ptr addrspace(3) @tg
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; A loop of a reduction step: the program's barrier between the store and
; the next iteration's load stays, the inserted ones next to it go.
; CHECK-LABEL: define agx_kernel void @tree(
; CHECK: loop:
; CHECK: load i32, ptr addrspace(3)
; CHECK: store i32
; CHECK-NEXT: fence syncscope("workgroup") release
; CHECK-NEXT: call void @llvm.agx.threadgroup.barrier()
; CHECK-NEXT: fence syncscope("workgroup") acquire
; CHECK-NOT: call void @llvm.agx.threadgroup.barrier()
; CHECK: ret void
define void @tree(ptr addrspace(1) %p) {
entry:
  %v = load i32, ptr addrspace(1) %p
  br label %loop
loop:
  %n = phi i32 [ 32, %entry ], [ %n1, %loop ]
  call void @air.wg.barrier(i32 2, i32 1)
  %a = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %n
  %x = load i32, ptr addrspace(3) %a
  call void @air.wg.barrier(i32 2, i32 1)
  %s = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %v
  store i32 %x, ptr addrspace(3) %s
  call void @air.wg.barrier(i32 2, i32 1)
  %n1 = lshr i32 %n, 1
  %c = icmp ne i32 %n1, 0
  br i1 %c, label %loop, label %done
done:
  ret void
}

; In a loop that accesses threadgroup memory every barrier is behind and
; ahead of an access; one that orders device memory stays anyway; and so does
; one after a call that may access threadgroup memory; and the callee's,
; inlined into the kernel, which another caller could call after accesses.
; CHECK-LABEL: define agx_kernel void @kept(
; CHECK-COUNT-4: call void @llvm.agx.threadgroup.barrier()
; CHECK-NOT: call void @llvm.agx.threadgroup.barrier()
define void @kept(ptr addrspace(1) %p) {
entry:
  %i = load i32, ptr addrspace(1) %p
  call void @air.wg.barrier(i32 3, i32 1)
  call void @helper()
  call void @air.wg.barrier(i32 2, i32 1)
  br label %loop
loop:
  %n = phi i32 [ 0, %entry ], [ %n1, %loop ]
  %s = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %i
  store i32 %n, ptr addrspace(3) %s
  call void @air.wg.barrier(i32 2, i32 1)
  %n1 = add i32 %n, 1
  %c = icmp ult i32 %n1, 4
  br i1 %c, label %loop, label %done
done:
  ret void
}

; Not in a function a kernel calls: it may be called after accesses.
define internal void @helper() noinline {
  call void @air.wg.barrier(i32 2, i32 1)
  ret void
}

!air.kernel = !{!0, !2, !3, !4}
!3 = !{ptr @pair, !{}, !{!1}}
!4 = !{ptr @tree, !{}, !{!1}}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"p"}
!2 = !{ptr @kept, !{}, !{!1}}
