; An unknown argument role, a location outside 0..30, and a type AIR does not have. The location is outside
; Metal's 31 buffers, which only the AIR given to Metal must respect: portable AIR also feeds the AGX backend.
; RUN: not opt -mtriple=air -passes=air-verify -disable-output %s 2>&1 | FileCheck %s --implicit-check-not="outside 0..30"
; RUN: not opt -mtriple=air -passes=air-verify-metal -disable-output %s 2>&1 | FileCheck %s --check-prefixes=CHECK,METAL
; CHECK-DAG: has the role 'air.nonsense', which AIR does not have
; METAL-DAG: a buffer argument at location 4000, outside 0..30
; CHECK-DAG: in function 'f': a type AIR does not have
target triple = "air"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b) "air-kernel" { ret void }
define void @f() {
  %x = fadd fp128 0xL0, 0xL0
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.nonsense"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 4000, i32 1, !"air.read_write"}
