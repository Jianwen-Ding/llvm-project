; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -stop-after=agx-strict-fp < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -agx-strict-bf16-fast-fma=false -stop-after=agx-strict-fp < %s | FileCheck %s --check-prefix=SOFT

; Strict bfloat fma: in float where that is exact (the product of normal
; bfloats, TwoSum's error, the sum rounded to odd, then to nearest bfloat),
; the exact software fma behind an unlikely branch otherwise.

; CHECK-LABEL: @f(
; CHECK: fmul float
; CHECK: fadd float
; CHECK: br i1 {{.*}}, !prof
; CHECK: ret void
; SOFT-LABEL: @f(
; SOFT-NOT: br i1
; SOFT: ret void
define agx_kernel void @f(ptr addrspace(1) %p, bfloat %a, bfloat %b, bfloat %c) #0 {
  %r = call bfloat @llvm.experimental.constrained.fma.bf16(bfloat %a, bfloat %b, bfloat %c, metadata !"round.tonearest", metadata !"fpexcept.ignore") #0
  store bfloat %r, ptr addrspace(1) %p
  ret void
}
declare bfloat @llvm.experimental.constrained.fma.bf16(bfloat, bfloat, bfloat, metadata, metadata)
attributes #0 = { strictfp }
