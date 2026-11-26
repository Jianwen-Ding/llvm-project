; MSL's atomic_load_explicit and atomic_store_explicit, as AIR calls them
; (air.atomic.<space>.load / store: pointer, [value,] order, scope, volatile),
; are atomic loads and stores of their order and scope. Device scope is
; syncscope("device"): coherent among the GPU's threads, not with the host
; (Metal makes nothing of a dispatch host-coherent before it completes), so the
; volatile ones are device atomics, without the host's coherent accesses and
; their barriers.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o - | FileCheck %s --check-prefix=ASM

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %a = load atomic volatile i32, ptr addrspace(1) %in syncscope("device") monotonic, align 4
; CHECK: store atomic volatile i32 %a, ptr addrspace(1) %out syncscope("device") release, align 4
; CHECK: %b = load atomic float, ptr addrspace(1) %in syncscope("device") acquire, align 4
; CHECK: store atomic float %b, ptr addrspace(1) %out syncscope("device") monotonic, align 4
; CHECK: %c = load atomic volatile i32, ptr addrspace(3) @k.tg syncscope("workgroup") acquire, align 4
; CHECK: store atomic volatile i32 %c, ptr addrspace(3) @k.tg syncscope("workgroup") monotonic, align 4
; CHECK-NOT: @air.atomic

; ASM-LABEL: k:
; ASM-NOT: memory_barrier 3, 1, 9
; ASM: atomic cmpxchg
; ASM-NOT: device_load {{.*}}, 3{{$}}

target triple = "air64-apple-macosx14.0.0"

declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
declare void @air.atomic.global.store.i32(ptr addrspace(1), i32, i32, i32, i1)
declare float @air.atomic.global.load.f32(ptr addrspace(1), i32, i32, i1)
declare void @air.atomic.global.store.f32(ptr addrspace(1), float, i32, i32, i1)
declare i32 @air.atomic.local.load.i32(ptr addrspace(3), i32, i32, i1)
declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)

define void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(3) %tg) {
  %a = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %in, i32 0, i32 2, i1 true)
  call void @air.atomic.global.store.i32(ptr addrspace(1) %out, i32 %a, i32 3, i32 2, i1 true)
  ; acq_rel: a load acquires, a store (of acquire) does neither.
  %b = call float @air.atomic.global.load.f32(ptr addrspace(1) %in, i32 4, i32 2, i1 false)
  call void @air.atomic.global.store.f32(ptr addrspace(1) %out, float %b, i32 2, i32 2, i1 false)
  %c = call i32 @air.atomic.local.load.i32(ptr addrspace(3) %tg, i32 2, i32 1, i1 true)
  call void @air.atomic.local.store.i32(ptr addrspace(3) %tg, i32 %c, i32 0, i32 1, i1 true)
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2, !3}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"in"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"tg"}
