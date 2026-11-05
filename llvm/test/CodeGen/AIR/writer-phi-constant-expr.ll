; A constant expression of a threadgroup global as a phi's operand is expanded in
; the incoming block: before the phi, the library is rejected when its pipeline is
; built ("Failed to materializeAll").
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: opt -passes=verify -disable-output %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s

; CHECK-LABEL: define {{.*}}@k(
; CHECK: ptrtoint
; CHECK: br label %[[LOOP:[0-9]+]]
; CHECK: [[LOOP]]:
; CHECK-NEXT: phi i32

target triple = "air"

@buf = internal addrspace(3) global [64 x i32] undef, align 4

define void @k(ptr addrspace(1) %o) "air-kernel" {
entry:
  br label %loop
loop:
  %a = phi i32 [ ptrtoint (ptr addrspace(3) @buf to i32), %entry ], [ %a.next, %loop ]
  %z = zext i32 %a to i64
  %p = inttoptr i64 %z to ptr addrspace(3)
  %v = load i32, ptr addrspace(3) %p
  store i32 %v, ptr addrspace(1) %o
  %a.next = add i32 %a, 16
  %c = icmp eq i32 %a.next, 256
  br i1 %c, label %done, label %loop
done:
  ret void
}
