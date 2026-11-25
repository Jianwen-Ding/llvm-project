; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=ASR
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -agx-sign-splat-compares=0 < %s | FileCheck %s --check-prefix=ASR

; x >> 31 (arithmetic), the sign splat: with the constants in uniforms, an
; icmpsel of x < 0 with -1 a uniform (the ALU's), not an asr (the shared
; unit's, L15-E562). Without them the asr stays (the -1 would hold a GPR).

; CHECK-LABEL: k:
; CHECK-NOT: asr
; CHECK: icmpsel slt, r{{[0-9]+}}, r{{[0-9]+}}, 0, u{{[0-9]+}}, 0
; CHECK-NOT: asr
; ASR-LABEL: k:
; ASR: asr r{{[0-9]+}}, r{{[0-9]+}}, 31
define agx_kernel void @k(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %i
  %x = load i32, ptr addrspace(1) %a
  %s = ashr i32 %x, 31
  %t = xor i32 %s, %x
  store i32 %t, ptr addrspace(1) %a
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
