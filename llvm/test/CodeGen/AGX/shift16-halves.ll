; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; x >> 16 on its own, arithmetic or logical: the high half extended by an
; iadd's source modifier (the ALU's), not an asr / bfeil on the shared unit
; (L15-E563).

; CHECK-LABEL: k:
; CHECK-NOT: asr
; CHECK-NOT: bfeil
; CHECK-DAG: iadd r{{[0-9]+}}, r{{[0-9]+}}h.sx, 0
; CHECK-DAG: iadd r{{[0-9]+}}, r{{[0-9]+}}h, 0
; CHECK: device_store
define agx_kernel void @k(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %i
  %x = load i32, ptr addrspace(1) %a
  %s = ashr i32 %x, 16
  %u = lshr i32 %x, 16
  %s3 = mul i32 %s, 3
  %u5 = mul i32 %u, 5
  %t = xor i32 %s3, %u5
  store i32 %t, ptr addrspace(1) %a
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
