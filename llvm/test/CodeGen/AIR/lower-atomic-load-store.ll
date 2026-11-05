; An atomic load or store of 32 bits is AIR's function for Metal: the writer has no atomic
; form, and a plain load polling for another thread's write may never see it (a kernel
; waiting on a mbarrier in another CTA hung an M2 Max). Other widths stay volatile.
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE

; PORTABLE: load atomic i32

; CHECK-LABEL: define {{.*}}@k(
; CHECK: call i32 @air.atomic.local.load.i32(ptr addrspace(3) {{.*}}, i32 0, i32 1, i1 true)
; CHECK: call void @air.atomic.global.store.i32(ptr addrspace(1) {{.*}}, i32 {{.*}}, i32 0, i32 2, i1 true)
; CHECK: load volatile i64

target triple = "air"
@t = internal addrspace(3) global i32 undef, align 4

define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load atomic i32, ptr addrspace(3) @t syncscope("workgroup") monotonic, align 4
  store atomic i32 %v, ptr addrspace(1) %o monotonic, align 4
  %w = load atomic i64, ptr addrspace(1) %o monotonic, align 8
  store i64 %w, ptr addrspace(1) %o
  ret void
}
