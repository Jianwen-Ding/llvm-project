; A byte array retyped by its accesses stays one array: a ptrtoint of the global moves
; with the getelementptrs. Left on the old global, it was a second allocation whose
; address named other memory (PTX shared addresses held as integers).
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s

; CHECK-COUNT-1: {{^}}@{{.*}}addrspace(3) global
; CHECK-NOT: {{^}}@{{.*}}addrspace(3) global
; CHECK-LABEL: define {{.*}}@k(

target triple = "air"

@b = internal addrspace(3) global [64 x i8] undef, align 16

define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i32 1, ptr addrspace(3) @b
  %a = ptrtoint ptr addrspace(3) @b to i32
  %s = add i32 %a, 4
  %z = zext i32 %s to i64
  %p = inttoptr i64 %z to ptr addrspace(3)
  store i32 2, ptr addrspace(3) %p
  %v = load i32, ptr addrspace(3) @b
  store i32 %v, ptr addrspace(1) %o
  ret void
}
