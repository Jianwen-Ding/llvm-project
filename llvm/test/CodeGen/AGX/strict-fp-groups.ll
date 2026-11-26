; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+canonical-nan -agx-ieee-group-limit=8 -stop-after=agx-strict-fp < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+canonical-nan -agx-ieee-group-limit=8 -verify-machineinstrs < %s -o /dev/null

; IEEE binary32 operations (!agx.keep-denormals) of a block checked together:
; the hardware results with one branch to the exact path, which is the run
; again with each operation's own check. Within the run, the members' results
; (hardware results, never denormal where the check passes) get no denormal
; tests: the fma tests only its other operands, and the reciprocal square
; root needs no scaling. A result whose every use canonicalizes its own NaNs
; gets no canonical NaN of its own. Denormal and zero tests are on
; 0xfffffe - 2|x|.

; CHECK-LABEL: define agx_kernel void @run(
; CHECK:         %m = fmul float %x, %y
; CHECK-NOT:     fcmp uno float %m
; CHECK:         sub i32 16777214,
; CHECK:         %s = call float @llvm.fma.f32(float %m, float %z, float %w)
; CHECK-NEXT:    bitcast float %m to i32
; CHECK-NEXT:    bitcast float %z to i32
; CHECK-NEXT:    bitcast float %w to i32
; CHECK-NEXT:    shl i32
; CHECK-NEXT:    sub i32 16777214,
; CHECK-NEXT:    icmp ult i32 {{.*}}, 16777213
; CHECK-NEXT:    shl i32
; CHECK-NEXT:    sub i32 16777214,
; CHECK-NEXT:    icmp ult i32 {{.*}}, 16777213
; CHECK-NEXT:    or i1
; CHECK:         %r = call float @llvm.agx.rsqrt.f32(float %s)
; CHECK-NEXT:    %t = fadd float %r, %s
; CHECK:         br i1 %{{[0-9]+}}, label %[[EXACT:[0-9]+]], label %[[JOIN:[0-9]+]], !prof
; CHECK:       [[EXACT]]:
; CHECK:         fmul float %x, %y
; CHECK:         br i1
; CHECK:       [[JOIN]]:
; CHECK-NEXT:    %[[T:[0-9]+]] = phi float [ %t, %0 ], [ %{{[0-9]+}}, %{{[0-9]+}} ]
; CHECK-NEXT:    fcmp uno float %[[T]], %[[T]]
define agx_kernel void @run(ptr addrspace(1) %out, float %x, float %y, float %z, float %w) {
  %m = fmul float %x, %y, !agx.keep-denormals !0
  %s = call float @llvm.fma.f32(float %m, float %z, float %w), !agx.keep-denormals !0
  %r = call float @llvm.agx.rsqrt.f32(float %s), !agx.keep-denormals !0
  %t = fadd float %r, %s, !agx.keep-denormals !0
  store float %t, ptr addrspace(1) %out
  ret void
}

; An integer converted to float is never denormal: its zero test is a float
; comparison, and an fma has no denormal test of it.
; CHECK-LABEL: define agx_kernel void @int_operand(
; CHECK:         %c = sitofp i32 %i to float
; CHECK:         %p = fmul float %c, %x
; CHECK:         fcmp oeq float %c, 0.000000e+00
; CHECK:         %q = call float @llvm.fma.f32(float %c, float %y, float %x)
; CHECK-NEXT:    bitcast float %c to i32
; CHECK-NEXT:    bitcast float %y to i32
; CHECK-NEXT:    bitcast float %x to i32
; CHECK-NEXT:    shl i32
; CHECK-NEXT:    sub i32 16777214,
; CHECK-NEXT:    icmp ult i32 {{.*}}, 16777213
; CHECK-NEXT:    shl i32
; CHECK-NEXT:    sub i32 16777214,
; CHECK-NEXT:    icmp ult i32 {{.*}}, 16777213
; CHECK-NEXT:    or i1
define agx_kernel void @int_operand(ptr addrspace(1) %out, i32 %i, float %x, float %y) {
  %c = sitofp i32 %i to float
  %p = fmul float %c, %x, !agx.keep-denormals !0
  store float %p, ptr addrspace(1) %out
  %q = call float @llvm.fma.f32(float %c, float %y, float %x), !agx.keep-denormals !0
  %g = getelementptr float, ptr addrspace(1) %out, i32 1
  store float %q, ptr addrspace(1) %g
  ret void
}

; A run none of whose members needs a check (x + 0 is a select, an addition
; of a constant of at least 2^-101 is always right): no branch.
; CHECK-LABEL: define agx_kernel void @unchecked(
; CHECK:         %a = select i1 %{{[0-9]+}}, float 0.000000e+00, float %x
; CHECK-NEXT:    %b = fadd float %a, 1.000000e+00
; CHECK-NOT:     br i1
; CHECK:         ret void
define agx_kernel void @unchecked(ptr addrspace(1) %out, float %x) {
  %a = fadd float %x, 0.0, !agx.keep-denormals !0
  %b = fadd float %a, 1.0, !agx.keep-denormals !0
  store float %b, ptr addrspace(1) %out
  ret void
}

declare float @llvm.fma.f32(float, float, float)
declare float @llvm.agx.rsqrt.f32(float)
!0 = !{}
