; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-converted-minmax-num=0 < %s | FileCheck %s --check-prefix=OFF

; A maximum / minimum whose only use is a float-to-integer conversion is the
; maxnum / minnum (a NaN made the conversion poison; a zero's sign converts
; to 0 either way), and a select of a NaN constant into the conversion is the
; other arm (MLIR's maximumf): no NaN select (fcmpsel eq, or the maximum's
; NaN-keeping order), and the clamp to [-128, 127] of an i8 conversion is the
; saturating conversion's: one select (L15-E281; alive2:
; tests/alive-tv/agx/converted-*).

; CHECK-LABEL: clamp:
; CHECK: fcmpsel gtn
; CHECK-NOT: fcmpsel
; CHECK: convert
; CHECK: stop
; OFF-LABEL: clamp:
; OFF-COUNT-2: fcmpsel
; OFF: convert
define agx_kernel void @clamp(ptr addrspace(1) %p, float %x) {
  %m = call float @llvm.maximum.f32(float %x, float -128.0)
  %n = call float @llvm.minimum.f32(float %m, float 127.0)
  %r = fptosi float %n to i8
  store i8 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: mlir:
; CHECK: fcmpsel gtn
; CHECK-NOT: fcmpsel
; CHECK: convert
; CHECK: stop
; OFF-LABEL: mlir:
; OFF: fcmpsel eq
define agx_kernel void @mlir(ptr addrspace(1) %p, float %x) {
  %m = call float @llvm.maxnum.f32(float %x, float -128.0)
  %c = fcmp uno float %x, 0.0
  %n = call float @llvm.minnum.f32(float %m, float 127.0)
  %s = select i1 %c, float 0x7FF8000000000000, float %n
  %r = fptosi float %s to i8
  store i8 %r, ptr addrspace(1) %p
  ret void
}

declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
