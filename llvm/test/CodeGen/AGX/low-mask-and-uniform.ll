; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=IMM
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -agx-low-mask-and-uniform=0 < %s | FileCheck %s --check-prefix=IMM

; x & (2^k - 1) beyond the 8-bit immediates: a bfeil (the shared unit's)
; without a mov_imm for the mask; with the constants in uniforms, an and
; with the mask in one (the ALU's, L15-E557).

; CHECK-LABEL: m7f:
; CHECK-NOT: bfeil
; CHECK: and r{{[0-9]+}}, r{{[0-9]+}}, u{{[0-9]+}}
; IMM-LABEL: m7f:
; IMM: bfeil r{{[0-9]+}}, 0, r{{[0-9]+}}, 0, mask 0x7FFFFFFF
; CHECK-LABEL: m3ff:
; CHECK-NOT: bfeil
; CHECK: and r{{[0-9]+}}, r{{[0-9]+}}, u{{[0-9]+}}
; IMM-LABEL: m3ff:
; IMM: bfeil r{{[0-9]+}}, 0, r{{[0-9]+}}, 0, mask 0x3FF

define agx_kernel void @m7f(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %i
  %x = load i32, ptr addrspace(1) %a
  %r = and i32 %x, 2147483647
  store i32 %r, ptr addrspace(1) %a
  ret void
}
define agx_kernel void @m3ff(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %i
  %x = load i32, ptr addrspace(1) %a
  %r = and i32 %x, 1023
  store i32 %r, ptr addrspace(1) %a
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
