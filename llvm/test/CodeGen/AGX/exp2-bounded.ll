; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-exp2-bounds-by-arg=0 < %s | FileCheck %s --check-prefix=CLAMP

; 2^x and 10^x clamp x itself (to [-151, 128] and [-46, 38.6], past the
; overflow, so that the scale 2^n has n <= 128), so that no infinity makes
; the low part of x ln b a NaN: two selects, instead of a clamp of x ln b and
; +inf / +0 selects after (the same result for every float, checked
; exhaustively, flushing and not). The scale, an exponent addition, ends in
; a NaN select (fcmpsel eq) (CLAMP).
;
; Now (L15-E485) +inf and +0 by compares of x itself, at the exact bounds
; (2^x: +inf from 128, +0 at or below -126 - 2^-17; 10^x: from 38.5318413,
; at or below -37.9297829; every float checked), and no clamps and no NaN
; select: a NaN fails both compares and stays the NaN of e^r.

; CHECK-LABEL: e2:
; CHECK: fcmpsel gte
; CHECK: fcmpsel lte
; CHECK-NOT: {{[fi]cmpsel}}
; CHECK: stop
; CLAMP-LABEL: e2:
; CLAMP-COUNT-2: fcmpsel {{lt|gt}}
; CLAMP: fcmpsel eq
; CLAMP-NOT: fcmpsel
; CLAMP: stop
define agx_kernel void @e2(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.exp2.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

; CHECK-LABEL: e10:
; CHECK: fcmpsel gte
; CHECK: fcmpsel lte
; CHECK-NOT: {{[fi]cmpsel}}
; CHECK: stop
; CLAMP-LABEL: e10:
; CLAMP-COUNT-2: fcmpsel {{lt|gt}}
; CLAMP: fcmpsel eq
; CLAMP-NOT: fcmpsel
; CLAMP: stop
define agx_kernel void @e10(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.exp10.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}
; 2^-y: the compares of y, mirrored.
; CHECK-LABEL: e2neg:
; CHECK: fcmpsel lte
; CHECK: fcmpsel gte
; CHECK-NOT: {{[fi]cmpsel}}
; CHECK: stop
define agx_kernel void @e2neg(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %n = fneg float %x
  %y = call float @llvm.exp2.f32(float %n)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp2.f32(float)
declare float @llvm.exp10.f32(float)
