; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; x + (a << s) and (a << s) - x for s of 5 to 7 (past iadd's 4-place shift),
; the shift used only there: one imadd / imsub by the 8-bit immediate 2^s
; instead of a bfi and an iadd on two units (L15-E318). Shifts of 8 or more
; (2^s past the immediate) and shared shifts stay.

; CHECK-LABEL: k:
; CHECK-DAG: imadd {{r[0-9]+}}, {{r[0-9]+}}, 64, u2
; CHECK-DAG: imsub {{r[0-9]+}}, {{r[0-9]+}}, 32, u3
; CHECK-DAG: bfi {{r[0-9]+}}, 0, {{[ru][0-9]+}}, 9
define agx_kernel void @k(ptr addrspace(1) %p, i32 %x, i32 %y) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = shl i32 %t, 6
  %s = add i32 %a, %x
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  store i32 %s, ptr addrspace(1) %q
  %b = shl i32 %t, 5
  %u = sub i32 %b, %y
  %q2 = getelementptr i32, ptr addrspace(1) %q, i32 1
  store i32 %u, ptr addrspace(1) %q2
  %c = shl i32 %t, 9
  %w = add i32 %c, %y
  %q3 = getelementptr i32, ptr addrspace(1) %q, i32 2
  store i32 %w, ptr addrspace(1) %q3
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
