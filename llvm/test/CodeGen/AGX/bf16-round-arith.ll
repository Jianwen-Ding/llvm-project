; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+canonical-nan < %s | FileCheck %s --check-prefix=CANON

; Rounding a float to bfloat: a result of the float arithmetic, whose NaN is
; 0x7fc00000 (still a NaN once rounded), needs no NaN test; any other float
; does, and so does arithmetic whose NaNs are PTX's canonical 0x7fffffff.

; CHECK-LABEL: add:
; CHECK: fadd32
; CHECK-NOT: fcmpsel eq
; CHECK: stop
; CANON-LABEL: add:
; CANON: fcmpsel eq
define agx_kernel void @add(ptr addrspace(1) %p, float %a, float %b) {
  %s = fadd float %a, %b
  %r = fptrunc float %s to bfloat
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: cvt:
; CHECK: fcmpsel eq
define agx_kernel void @cvt(ptr addrspace(1) %p, float %a) {
  %r = fptrunc float %a to bfloat
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; The larger of two bfloats is already a bfloat: its float needs no rounding,
; only the NaN replacement.
; CHECK-LABEL: maxbf:
; CHECK-NOT: {{iadd|icmpsel}}
; CHECK: stop
define agx_kernel void @maxbf(ptr addrspace(1) %p, <2 x bfloat> %a, <2 x bfloat> %b) {
  %r = call <2 x bfloat> @llvm.maximumnum.v2bf16(<2 x bfloat> %a, <2 x bfloat> %b)
  %n = fcmp uno <2 x bfloat> %r, zeroinitializer
  %s = select <2 x i1> %n, <2 x bfloat> splat (bfloat 0xR7FFF), <2 x bfloat> %r
  store <2 x bfloat> %s, ptr addrspace(1) %p
  ret void
}
declare <2 x bfloat> @llvm.maximumnum.v2bf16(<2 x bfloat>, <2 x bfloat>)
