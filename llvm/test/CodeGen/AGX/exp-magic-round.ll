; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-exp-magic-round=0 -agx-sincos-magic-round=0 < %s | FileCheck %s --check-prefix=OFF
;
; The precise exponential's n = round(x / ln 2) is t + 1.5 2^23 - 1.5 2^23 on
; the FP pipe, not a rint and a convert on the shared units; the scale's
; integer is the sum's bits (0x4b400000 + n, which shifted by 23 is n << 23),
; and the subtraction reads the constant's register negated (L15-E469).

; CHECK-LABEL: e:
; CHECK: mov_imm [[C:r[0-9]+]], 1262485504
; CHECK: fadd32 [[M:r[0-9]+]], {{r[0-9]+}}, [[C]]
; CHECK-NEXT: fadd32 {{r[0-9]+}}, [[M]], [[C]].neg
; CHECK-NOT: rint
; CHECK-NOT: convert
; CHECK: imadd {{r[0-9]+}}, [[M]], {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: stop
; OFF-LABEL: e:
; OFF: rint
; OFF: convert f_to_s32
define agx_kernel void @e(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.exp.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

; sin's j = round(2 x / pi) likewise, its quadrant the sum's low two bits
; (L15-E470).
; CHECK-LABEL: sin_q:
; CHECK: mov_imm [[C:r[0-9]+]], 1262485504
; CHECK: fadd32 [[M:r[0-9]+]], {{r[0-9]+}}, [[C]]
; CHECK-NEXT: fadd32 {{r[0-9]+}}, [[M]], [[C]].neg
; CHECK-NOT: rint
; CHECK-NOT: convert
; CHECK: and {{r[0-9]+}}, [[M]], 1
; CHECK: stop
; OFF-LABEL: sin_q:
; OFF: rint
; OFF: convert f_to_s32
define agx_kernel void @sin_q(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.sin.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

; A negative constant alone stays a register of its own.
; CHECK-LABEL: neg_alone:
; CHECK: mov_imm [[K:r[0-9]+]], -1027080192
; CHECK: fmul32 {{r[0-9]+}}, {{r[0-9]+}}, [[K]]
define agx_kernel void @neg_alone(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = fmul float %x, -100.0
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp.f32(float)
declare float @llvm.sin.f32(float)
