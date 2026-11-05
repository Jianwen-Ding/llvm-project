; A call with another function type than its callee's declaration.
; RUN: not opt -mtriple=air -passes=air-verify -disable-output %s 2>&1 | FileCheck %s
; CHECK: a call of 'air.wg.barrier' with another function type than its declaration
target triple = "air"
declare void @air.wg.barrier(i32)
define void @f() {
  call void @air.wg.barrier(i32 2, i32 1)
  ret void
}
