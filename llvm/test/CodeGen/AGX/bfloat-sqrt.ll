; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-bfloat-sqrt-product=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -mattr=+ieee-fp32 < %s | FileCheck %s --check-prefix=OFF

; A bfloat square root as the float product x rsqrt_special(x) rounded to
; bfloat: the same as the correctly rounded float square root rounded to
; bfloat for every bfloat where float arithmetic flushes (L15-E335).

; CHECK-LABEL: bsqrt:
; CHECK: rsqrt_special [[Y:r[0-9]+]], [[X:r[0-9]+]]
; CHECK-NEXT: fmul32 {{r[0-9]+}}, [[X]], [[Y]]
; CHECK-NOT: fmadd32
; CHECK: stop
; OFF-LABEL: bsqrt:
; OFF: fmadd32
define agx_kernel void @bsqrt(ptr addrspace(1) %p) {
  %x = load bfloat, ptr addrspace(1) %p
  %r = call bfloat @llvm.sqrt.bf16(bfloat %x)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}
