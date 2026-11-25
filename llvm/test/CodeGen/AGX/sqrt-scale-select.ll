; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -agx-sqrt-scale-select=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=OFF
;
; float sqrt's rescale of a value below 2^-80 (scaled by 2^64 first): the
; result times (x < 2^-80 ? 2^-32 : 1), with the constants in uniforms, the
; select reading x alone, instead of a select between s 2^-32 and s (three
; registers). Without uniform constants, the select of two mov_imm values
; would read three registers too: the select of the products (L15-E698).

; CHECK-LABEL: sqrt:
; CHECK:      fcmpsel lt, [[SC:r[0-9]+]], [[X:r[0-9]+]], [[K:u[0-9]+]], u{{[0-9]+}}, u{{[0-9]+}}
; CHECK:      fcmpsel eq, [[S:r[0-9]+]], [[S]], [[S]], [[S]], r{{[0-9]+}}
; CHECK-NEXT: fmul32 r{{[0-9]+}}, [[S]], [[SC]]
; CHECK-NEXT: device_store
; OFF-LABEL: sqrt:
; OFF:      fcmpsel eq, [[S:r[0-9]+]], [[S]], [[S]], [[S]], r{{[0-9]+}}
; OFF-NEXT: fmul32 [[P:r[0-9]+]], [[S]], {{[ru][0-9]+}}
; OFF-NEXT: fcmpsel lt, r{{[0-9]+}}, {{[ru][0-9]+}}, {{[ru][0-9]+}}, [[P]], [[S]]
; OFF-NEXT: device_store
define agx_kernel void @sqrt(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %x = load float, ptr addrspace(1) %q
  %r = call float @llvm.sqrt.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}

declare float @llvm.sqrt.f32(float)
