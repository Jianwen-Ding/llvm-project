; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-quad-xor-shuffles=0 < %s | FileCheck %s --check-prefix=OFF

; A SIMD-group xor shuffle by 1, 2 or 3 stays in the lane's quad: the
; quad-group's, which issues in half the cycles (the M2 Max coverage
; recordings: 1.04 against 2.0); by 4 or more it stays (L15-E304).

; CHECK-LABEL: butterfly:
; CHECK: simd_shuffle_xor {{r[0-9]+}}, {{r[0-9]+}}, 4
; CHECK: quad_shuffle_xor {{r[0-9]+}}, {{r[0-9]+}}, 2
; CHECK: quad_shuffle_xor {{r[0-9]+}}, {{r[0-9]+}}, 1
; OFF-LABEL: butterfly:
; OFF-NOT: quad_shuffle_xor
define agx_kernel void @butterfly(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x0 = load float, ptr addrspace(1) %q
  %s4 = call float @llvm.agx.simd.shuffle.xor.f32(float %x0, i32 4)
  %x1 = fadd float %x0, %s4
  %s2 = call float @llvm.agx.simd.shuffle.xor.f32(float %x1, i32 2)
  %x2 = fadd float %x1, %s2
  %s1 = call float @llvm.agx.simd.shuffle.xor.f32(float %x2, i32 1)
  %x3 = fadd float %x2, %s1
  store float %x3, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.agx.simd.shuffle.xor.f32(float, i32)
