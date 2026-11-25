; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-nonneg-xor-divisions=0 < %s | FileCheck %s --check-prefix=OFF

; sdiv (x ^ (x >>s 31)), c > 0, a floor division's dividend (never
; negative), is the udiv of its low 31 bits: the unsigned magic multiply and
; shift without the signed quotient's rounding correction (L15-E292; alive2:
; tests/alive-tv/agx/nonneg-xor-sdiv).

; CHECK-LABEL: floor7:
; CHECK: bfeil {{r[0-9]+}}, 0, {{r[0-9]+}}, 0, mask 0x7FFFFFFF
; CHECK-NOT: .sx
; CHECK: stop
; OFF-LABEL: floor7:
; OFF: imadd {{r[0-9]+}}_{{r[0-9]+}}, {{r[0-9]+}}.sx
define agx_kernel void @floor7(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %x = load i32, ptr addrspace(1) %q
  %s = ashr i32 %x, 31
  %a = xor i32 %x, %s
  %d = sdiv i32 %a, 7
  %r = xor i32 %d, %s
  store i32 %r, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
