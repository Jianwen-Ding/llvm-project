; Names of the SIMD-group and atomic families are checked exactly.
; RUN: not opt -mtriple=air -passes=air-verify -disable-output %s 2>&1 | FileCheck %s
; CHECK-DAG: a call of 'air.simd_nonsense.i32', which AIR does not have
; CHECK-DAG: a call of 'air.atomic.global.nonsense.i32', which AIR does not have
; CHECK-DAG: a call of 'air.simd_shuffle.u.i32' without operands
target triple = "air"
declare i32 @air.simd_nonsense.i32(i32)
declare i32 @air.atomic.global.nonsense.i32(ptr addrspace(1))
declare i32 @air.simd_shuffle.u.i32()
declare i32 @air.simd_sum.s.i32(i32)
declare i32 @air.atomic.global.add.s.i32(ptr addrspace(1), i32, i32, i32, i1)
define i32 @f(ptr addrspace(1) %p) {
  %a = call i32 @air.simd_nonsense.i32(i32 1)
  %b = call i32 @air.atomic.global.nonsense.i32(ptr addrspace(1) %p)
  %c = call i32 @air.simd_shuffle.u.i32()
  %d = call i32 @air.simd_sum.s.i32(i32 1)
  %e = call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %p, i32 1, i32 0, i32 2, i1 false)
  ret i32 %e
}

; RUN: not opt -mtriple=air -passes=air-verify-metal -disable-output %s 2>&1 | FileCheck %s --check-prefix=METAL
; METAL-DAG: a call of 'llvm.stacksave.p0', which Metal does not take
; METAL-DAG: a call through an address made from an integer
; METAL-DAG: the global 'q' has a type AIR does not have
@q = addrspace(2) constant fp128 0xL0
declare ptr @llvm.stacksave.p0()
define void @g() {
  %s = call ptr @llvm.stacksave.p0()
  %f = inttoptr i64 8 to ptr
  call void %f()
  ret void
}
