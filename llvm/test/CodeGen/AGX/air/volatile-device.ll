; AIR's volatile 32-bit device accesses are volatile monotonic atomics of
; device scope (device atomics, coherent across the GPU's cores) rather than
; host-coherent accesses: Metal makes nothing of a dispatch host-coherent
; before it ends. Other widths, misaligned ones and threadgroup memory stay.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o - | FileCheck %s --check-prefix=ASM

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %a = load atomic volatile i32, ptr addrspace(1) %p syncscope("device") monotonic, align 4
; CHECK: %f = load atomic volatile float, ptr addrspace(1) %p syncscope("device") monotonic, align 4
; CHECK: store atomic volatile i32 %a, ptr addrspace(1) %p syncscope("device") monotonic, align 4
; CHECK: %w = load volatile i64, ptr addrspace(1) %p, align 8
; CHECK: %u = load volatile i32, ptr addrspace(1) %p, align 2
; CHECK: %t = load volatile i32, ptr addrspace(3) @k.tg, align 4

; The 32-bit ones are device atomics, without a barrier; the 64-bit and the
; misaligned one stay host-coherent (a barrier, a coherent load).
; ASM-LABEL: k:
; ASM-NOT: memory_barrier
; ASM: atomic cmpxchg, 0,
; ASM-NOT: memory_barrier
; ASM: atomic cmpxchg, 1,
; ASM-NOT: memory_barrier
; ASM: atomic xchg,
; ASM: memory_barrier 3, 1, 9
; ASM-NEXT: device_load 0, i32, xy, {{.*}}, 3
; ASM: memory_barrier 3, 1, 9
; ASM-NEXT: device_load 0, i32, x, {{.*}}, 3

target triple = "air64-apple-macosx14.0.0"

define void @k(ptr addrspace(1) %p, ptr addrspace(3) %tg) {
  %a = load volatile i32, ptr addrspace(1) %p, align 4
  %f = load volatile float, ptr addrspace(1) %p, align 4
  store volatile i32 %a, ptr addrspace(1) %p, align 4
  %w = load volatile i64, ptr addrspace(1) %p, align 8
  %u = load volatile i32, ptr addrspace(1) %p, align 2
  %t = load volatile i32, ptr addrspace(3) %tg, align 4
  %s = add i32 %u, %t
  %fi = bitcast float %f to i32
  %s2 = add i32 %s, %fi
  %wl = trunc i64 %w to i32
  %s3 = add i32 %s2, %wl
  store i32 %s3, ptr addrspace(1) %p, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"p"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"tg"}
