; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32 -o /dev/null -print-after=agx-strict-fp < %s 2>&1 | FileCheck %s --check-prefix=STRICT
; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-lower-fp-env < %s 2>&1 | FileCheck %s --check-prefix=ENV

; A negation or absolute value of its own is a bit operation, which keeps
; denormals.
; CHECK-LABEL: neg:
; CHECK: xor
; CHECK-NOT: fadd
define agx_kernel void @neg(ptr addrspace(1) %o, float %a) #0 {
  %n = fneg float %a
  store float %n, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: abs:
; CHECK: and
; CHECK-NOT: fadd
define agx_kernel void @abs(ptr addrspace(1) %o, float %a) #0 {
  %n = call float @llvm.fabs.f32(float %a)
  store float %n, ptr addrspace(1) %o
  ret void
}

; The hardware minimum keeps a denormal operand: its result is flushed in a
; function that flushes float denormals.
; CHECK-LABEL: min_ftz:
; CHECK: fcmpsel
; CHECK-NEXT: fmul32 {{.*}}, 1.0
define agx_kernel void @min_ftz(ptr addrspace(1) %o, float %a, float %b) #0 {
  %m = call float @llvm.minnum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %o
  ret void
}

; float to bfloat rounds to nearest even.
; CHECK-LABEL: to_bf16:
; CHECK: iadd
; CHECK: 32767
define agx_kernel void @to_bf16(ptr addrspace(1) %o, float %a) {
  %b = fptrunc float %a to bfloat
  store bfloat %b, ptr addrspace(1) %o
  ret void
}

; With ieee-fp32, bfloat arithmetic is float arithmetic on the extended
; operands, tagged for the IEEE lowering, rounded back once; fma is computed
; exactly with integer arithmetic.
; STRICT-LABEL: @bf16_add(
; STRICT: fpext bfloat %a to float
; STRICT: fptrunc float {{.*}} to bfloat
; STRICT-NOT: fadd bfloat
define agx_kernel void @bf16_add(ptr addrspace(1) %o, bfloat %a, bfloat %b) {
  %s = fadd bfloat %a, %b
  store bfloat %s, ptr addrspace(1) %o
  ret void
}

; STRICT-LABEL: @bf16_fma(
; STRICT-NOT: @llvm.fma
; STRICT: mul i128
define agx_kernel void @bf16_fma(ptr addrspace(1) %o, bfloat %a, bfloat %b, bfloat %c) {
  %s = call bfloat @llvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c)
  store bfloat %s, ptr addrspace(1) %o
  ret void
}

; Vector constrained operations are rounded lane by lane.
; ENV-LABEL: @vector_rz(
; ENV-NOT: call {{.*}}constrained
; ENV: insertelement <2 x float>
define agx_kernel void @vector_rz(ptr addrspace(1) %o, <2 x float> %a, <2 x float> %b) #1 {
  %s = call <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float> %a, <2 x float> %b, metadata !"round.towardzero", metadata !"fpexcept.ignore") #1
  store <2 x float> %s, ptr addrspace(1) %o
  ret void
}

; round(x), half away from zero: trunc(x + copysign(0x3effffff, x)), the
; float just below 0.5 (0.5 itself would round 0.49999997 up).
; CHECK-LABEL: round_f32:
; CHECK: mov_imm [[B:r[0-9]+]], 1056964607
; CHECK-NEXT: bfi [[B]], [[X:u[0-9]+]], [[B]], 0, mask 0x7FFFFFFF
; CHECK-NEXT: fadd32 [[S:r[0-9]+]], [[X]], [[B]]
; CHECK-NEXT: trunc {{r[0-9]+}}, [[S]]
define agx_kernel void @round_f32(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.round.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: round_f16:
; CHECK: mov_imm [[B:r[0-9]+l]], 14335
; CHECK-NEXT: bfi [[B]], [[X:u[0-9]+l]], [[B]], 0, mask 0x7FFF
; CHECK-NEXT: fadd16 [[S:r[0-9]+l]], [[X]], [[B]]
; CHECK-NEXT: trunc {{r[0-9]+l}}, [[S]]
define agx_kernel void @round_f16(ptr addrspace(1) %p, half %x) {
  %r = call half @llvm.round.f16(half %x)
  store half %r, ptr addrspace(1) %p
  ret void
}

; CUDA's roundf (add.rz.ftz.f32 of copysign(0.5, x), then cvt.rzi) is round(x).
; ENV-LABEL: @roundf_copysign(
; ENV: %s = call float @llvm.round.f32(float %x)
; ENV-NOT: constrained
; ENV: call float @llvm.trunc.f32(float %s)
define agx_kernel void @roundf_copysign(ptr addrspace(1) %p, float %x) #99 {
  %h = call float @llvm.copysign.f32(float 5.000000e-01, float %x)
  %s = call float @llvm.experimental.constrained.fadd.f32(float %x, float %h, metadata !"round.towardzero", metadata !"fpexcept.ignore") #99
  %t = call float @llvm.experimental.constrained.trunc.f32(float %s, metadata !"fpexcept.ignore") #99
  store float %t, ptr addrspace(1) %p
  ret void
}

; With the copysign folded for a clamped x, -0 + 0.5 truncates to +0: round(x)
; + 0, unless the sign bit of x is known clear (maximumnum with 0).
; ENV-LABEL: @roundf_nonneg(
; ENV: [[R:%.*]] = call float @llvm.round.f32(float %x)
; ENV-NEXT: %s = fadd float [[R]], 0.000000e+00
; ENV-LABEL: @roundf_clamped(
; ENV: %s = call float @llvm.round.f32(float %c)
; ENV-NEXT: call float @llvm.trunc.f32(float %s)
define agx_kernel void @roundf_nonneg(ptr addrspace(1) %p, float nofpclass(ninf nsub nnorm) %x) #99 {
  %s = call float @llvm.experimental.constrained.fadd.f32(float %x, float 5.000000e-01, metadata !"round.towardzero", metadata !"fpexcept.ignore") #99
  %t = call float @llvm.experimental.constrained.trunc.f32(float %s, metadata !"fpexcept.ignore") #99
  store float %t, ptr addrspace(1) %p
  ret void
}
define agx_kernel void @roundf_clamped(ptr addrspace(1) %p, float %y) #99 {
  %c = call float @llvm.maximumnum.f32(float %y, float 0.000000e+00)
  %s = call float @llvm.experimental.constrained.fadd.f32(float %c, float 5.000000e-01, metadata !"round.towardzero", metadata !"fpexcept.ignore") #99
  %t = call float @llvm.experimental.constrained.trunc.f32(float %s, metadata !"fpexcept.ignore") #99
  store float %t, ptr addrspace(1) %p
  ret void
}

; Without the truncation the addition stays a rounded one.
; ENV-LABEL: @rz_add_kept(
; ENV-NOT: llvm.round
; ENV: ret void
define agx_kernel void @rz_add_kept(ptr addrspace(1) %p, float %x) #99 {
  %h = call float @llvm.copysign.f32(float 5.000000e-01, float %x)
  %s = call float @llvm.experimental.constrained.fadd.f32(float %x, float %h, metadata !"round.towardzero", metadata !"fpexcept.ignore") #99
  store float %s, ptr addrspace(1) %p
  ret void
}

declare float @llvm.fabs.f32(float)
declare float @llvm.minnum.f32(float, float)
declare bfloat @llvm.fma.bf16(bfloat, bfloat, bfloat)
declare <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float>, <2 x float>, metadata, metadata)

attributes #0 = { denormal_fpenv(preservesign) }
attributes #1 = { strictfp }

; With ieee-fp32, 2^x is 2^f * 2^n with n = floor(x), scaled exactly, and
; sin x is x for tiny x.
; STRICT-LABEL: @exp2_sin(
; STRICT: call float @llvm.floor.f32
; STRICT: call float @llvm.exp2.f32
; STRICT: fcmp olt float {{.*}}, f0x39800000
define agx_kernel void @exp2_sin(ptr addrspace(1) %o, float %a) {
  %e = call float @llvm.exp2.f32(float %a)
  %s = call float @llvm.sin.f32(float %e)
  store float %s, ptr addrspace(1) %o
  ret void
}
declare float @llvm.round.f32(float)
declare float @llvm.copysign.f32(float, float)
declare float @llvm.maximumnum.f32(float, float)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.trunc.f32(float, metadata)
attributes #99 = { strictfp }

; -|x| is an or of the sign bit.
; CHECK-LABEL: fnabs_f32:
; CHECK: mov_imm [[M:r[0-9]+]], -2147483648
; CHECK-NEXT: or [[M]], {{u[0-9]+}}, [[M]]
; CHECK-NEXT: device_store
; CHECK-LABEL: fnabs_f16:
; CHECK: mov_imm [[H:r[0-9]+l]], 32768
; CHECK-NEXT: or [[H]], {{u[0-9]+l}}, [[H]]
; CHECK-NEXT: device_store
define agx_kernel void @fnabs_f32(ptr addrspace(1) %p, float %x) {
  %a = call float @llvm.fabs.f32(float %x)
  %n = fneg float %a
  store float %n, ptr addrspace(1) %p
  ret void
}
define agx_kernel void @fnabs_f16(ptr addrspace(1) %p, half %x) {
  %a = call half @llvm.fabs.f16(half %x)
  %n = fneg half %a
  store half %n, ptr addrspace(1) %p
  ret void
}
