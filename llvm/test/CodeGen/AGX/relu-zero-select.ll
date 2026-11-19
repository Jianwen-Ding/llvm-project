; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-relu-zero-selects=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's arith.maximumf(x, 0) through AIR, which has no maximum: maxnum(x, +0),
; select(x == 0, +0, ...) for -0, and the NaN select. maxnum(x, +0) as
; x > 0 ? x : +0 (frozen x) is +0 wherever x == 0, and so is minnum of it
; with 6 (relu6): the zero select goes (L15-E399; alive2, IEEE and
; preserve-sign denormals). A minnum with a negative constant keeps it.

declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)

; CHECK-LABEL: relu:
; CHECK-COUNT-2: fcmpsel
; CHECK-NOT:   fcmpsel
; CHECK:       stop
; OFF-LABEL:   relu:
; OFF-COUNT-3: fcmpsel
define agx_kernel void @relu(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p, align 4
  %m = call float @llvm.maxnum.f32(float %x, float 0.0)
  %z = fcmp oeq float %x, 0.0
  %r = select i1 %z, float 0.0, float %m
  %u = fcmp uno float %x, 0.0
  %n = select i1 %u, float 0x7FF8000000000000, float %r
  store float %n, ptr addrspace(1) %p, align 4
  ret void
}

; CHECK-LABEL: relu6:
; CHECK-COUNT-3: fcmpsel
; CHECK-NOT:   fcmpsel
; CHECK:       stop
define agx_kernel void @relu6(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p, align 4
  %m = call float @llvm.maxnum.f32(float %x, float 0.0)
  %z = fcmp oeq float %x, 0.0
  %c = call float @llvm.minnum.f32(float %m, float 6.0)
  %r = select i1 %z, float 0.0, float %c
  %u = fcmp uno float %x, 0.0
  %n = select i1 %u, float 0x7FF8000000000000, float %r
  store float %n, ptr addrspace(1) %p, align 4
  ret void
}

; CHECK-LABEL: negmin:
; CHECK-COUNT-4: fcmpsel
; CHECK:       stop
define agx_kernel void @negmin(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p, align 4
  %m = call float @llvm.maxnum.f32(float %x, float 0.0)
  %z = fcmp oeq float %x, 0.0
  %c = call float @llvm.minnum.f32(float %m, float -1.0)
  %r = select i1 %z, float 0.0, float %c
  %u = fcmp uno float %x, 0.0
  %n = select i1 %u, float 0x7FF8000000000000, float %r
  store float %n, ptr addrspace(1) %p, align 4
  ret void
}
