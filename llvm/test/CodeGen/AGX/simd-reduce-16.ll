; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; 16-bit SIMD-group and quad-group reductions and scans are the 16-bit forms
; of the instructions (half registers: fp16 for the float operations,
; recorded on the M2 Max as simd_fadd r14l, r6l and the like); they had no
; patterns (an f16 gpu.subgroup_reduce failed to select).

declare half @llvm.agx.simd.reduce.f16(half, i32 immarg)
declare i16 @llvm.agx.quad.reduce.i16(i16, i32 immarg)
declare half @llvm.agx.simd.prefix.f16(half, i32 immarg)

; CHECK-LABEL: reduce16:
; CHECK: simd_fadd r{{[0-9]+}}{{[lh]}}, {{[ru][0-9]+[lh]}}
; CHECK: quad_max.u r{{[0-9]+}}{{[lh]}}, {{[ru][0-9]+[lh]}}
; CHECK: simd_prefix_fadd r{{[0-9]+}}{{[lh]}}, {{[ru][0-9]+[lh]}}
define agx_kernel void @reduce16(ptr addrspace(1) %p, ptr addrspace(1) %q, half %h, i16 %i) {
  %s = call half @llvm.agx.simd.reduce.f16(half %h, i32 1)
  %m = call i16 @llvm.agx.quad.reduce.i16(i16 %i, i32 30)
  %x = call half @llvm.agx.simd.prefix.f16(half %h, i32 1)
  %t = fadd half %s, %x
  store half %t, ptr addrspace(1) %p, align 2
  store i16 %m, ptr addrspace(1) %q, align 2
  ret void
}
