; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-log-exponent-by-add=0 < %s | FileCheck %s --check-prefix=CONVERT

; log's exponent e: the biased field into the bits of 2^23 that it leaves
; clear (one bfeil with that base), less 2^23 + 127 (L15-E614), not a
; subtract and a convert.
; CHECK-LABEL: lone:
; CHECK-NOT: convert
; CHECK: bfeil {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, 23, mask 0xFF
; CHECK-NOT: convert
; CHECK: device_store
; CONVERT-LABEL: lone:
; CONVERT: convert s16_to_f
define agx_kernel void @lone(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %t to i64
  %pp = getelementptr float, ptr addrspace(1) %p, i64 %z
  %x = load float, ptr addrspace(1) %pp
  %l = call float @llvm.log.f32(float %x)
  %qq = getelementptr float, ptr addrspace(1) %q, i64 %z
  store float %l, ptr addrspace(1) %qq
  ret void
}

declare float @llvm.log.f32(float)
declare i32 @llvm.agx.thread.position.in.grid.x()
