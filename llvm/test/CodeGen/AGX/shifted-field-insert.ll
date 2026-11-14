; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-shifted-field-insert=0 < %s | FileCheck %s --check-prefix=OFF

; (x << s) | q with s above iadd's 4 places, x within its field and q zero
; there: the shift is a bfi already, so one bfi inserting x into q saves the
; or (L15-E317): a 3-D index (z << 12) | (y << 6) | x.

; CHECK-LABEL: idx3:
; CHECK: bfi {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, 6, mask 0x3F
; CHECK: bfi {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, 12, mask 0xF
; CHECK-NOT: {{^	}}or	
; OFF-LABEL: idx3:
; OFF: {{^	}}or	
define agx_kernel void @idx3(ptr addrspace(1) %p) {
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %y = call i32 @llvm.agx.thread.position.in.grid.y()
  %z = call i32 @llvm.agx.thread.position.in.grid.z()
  %xm = and i32 %x, 63
  %ym = and i32 %y, 63
  %zm = and i32 %z, 15
  %a = shl i32 %ym, 6
  %b = shl i32 %zm, 12
  %c = or disjoint i32 %a, %xm
  %i = or disjoint i32 %c, %b
  %q = getelementptr float, ptr addrspace(1) %p, i32 %i
  store float 1.0, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.grid.y()
declare i32 @llvm.agx.thread.position.in.grid.z()
