; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-floor-modulos=0 < %s | FileCheck %s --check-prefix=OFF

; The floor modulo r < 0 ? r + c : r of r = srem x, c (MLIR's affine mod) is
; x - c * floor(x / c), the floor quotient through the unsigned division of
; x ^ (x >>s 31): no signed magic multiply (.sx) and no select (L15-E294;
; alive2: tests/alive-tv/agx/floor-mod).

; CHECK-LABEL: fmod7:
; CHECK-NOT: .sx
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: fmod7:
; OFF: imadd {{r[0-9]+}}_{{r[0-9]+}}, {{r[0-9]+}}.sx
; OFF: icmpsel slt
define agx_kernel void @fmod7(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %x = load i32, ptr addrspace(1) %q
  %r = srem i32 %x, 7
  %n = icmp slt i32 %r, 0
  %a = add nsw i32 %r, 7
  %m = select i1 %n, i32 %a, i32 %r
  store i32 %m, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
