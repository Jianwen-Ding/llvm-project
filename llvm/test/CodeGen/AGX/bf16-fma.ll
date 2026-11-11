; RUN: llc -mtriple=agx -mcpu=agx-g13g -stop-after=agx-strict-fp < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g13g -agx-bf16-fma=false -stop-after=agx-strict-fp < %s | FileCheck %s --check-prefix=OFF

; bfloat fma without ieee-fp32 too is rounded once: the float fma and a
; rounding to bfloat would round twice (fma(0x4704, 0xcab0, 0x436e) is
; 0xd235, the float fma rounded 0xd236). In float where that is exact (TwoSum,
; the sum rounded to odd, then to nearest bfloat), the exact software fma
; behind an unlikely branch otherwise. fmuladd, which may round the product,
; is the bfloat multiplication and addition.

; CHECK-LABEL: @fma(
; CHECK: fmul float
; CHECK: fadd float
; CHECK: fsub float
; CHECK: br i1 {{.*}}, !prof
; CHECK-NOT: call bfloat @llvm.fma
; OFF-LABEL: @fma(
; OFF: call bfloat @llvm.fma.bf16
define agx_kernel void @fma(ptr addrspace(1) %p, bfloat %a, bfloat %b, bfloat %c) {
  %r = call bfloat @llvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @fmuladd(
; CHECK: [[M:%.*]] = fmul bfloat %a, %b
; CHECK-NEXT: fadd bfloat [[M]], %c
; OFF-LABEL: @fmuladd(
; OFF: call bfloat @llvm.fmuladd.bf16
define agx_kernel void @fmuladd(ptr addrspace(1) %p, bfloat %a, bfloat %b, bfloat %c) {
  %r = call bfloat @llvm.fmuladd.bf16(bfloat %a, bfloat %b, bfloat %c)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}
