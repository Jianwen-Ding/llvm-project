; An unknown argument role, a location outside 0..30, and a type AIR does not have.
; RUN: not opt -mtriple=air -passes=air-verify -disable-output %s 2>&1 | FileCheck %s
; CHECK-DAG: has the role 'air.nonsense', which AIR does not have
; CHECK-DAG: a buffer argument at location 4000, outside 0..30
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
