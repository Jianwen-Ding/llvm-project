; A barrier only some threads reach desynchronizes the threadgroup: threads that
; did not reach it pass the next barrier early and read what another thread has not
; written yet (seen on an Apple M2 Max, PTX tcgen05.alloc by the first SIMD-group), or
; never finish. The thread's index may be a kernel parameter that !air.kernel
; describes, and a condition may come from memory.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @by_index(
; CHECK: icmp ult
; CHECK-NEXT: br i1
; CHECK-NOT: call void @air.wg.barrier
; CHECK: store i32 1, ptr addrspace(3)

target triple = "air"

@tg = internal addrspace(3) global [8 x i32] undef, align 4

define void @by_index(<3 x i32> %tid) "air-kernel" {
entry:
  %x = extractelement <3 x i32> %tid, i64 0
  %c = icmp ult i32 %x, 32
  br i1 %c, label %first, label %join
first:
  br label %first2
first2:
  store i32 1, ptr addrspace(3) @tg
  br label %join
join:
  ret void
}

; CHECK-LABEL: define void @by_memory(
; CHECK: load i32, ptr addrspace(1)
; CHECK: br i1
; CHECK-NOT: call void @air.wg.barrier
; CHECK: store i32 2, ptr addrspace(3)
define void @by_memory(ptr addrspace(1) %p) "air-kernel" {
entry:
  %v = load i32, ptr addrspace(1) %p
  %c = icmp eq i32 %v, 0
  br i1 %c, label %a, label %join
a:
  br label %b
b:
  store i32 2, ptr addrspace(3) @tg
  br label %join
join:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @by_index, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tidtg"}
