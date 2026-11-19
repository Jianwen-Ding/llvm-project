; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
;
; bits + (n << 23), n a float's conversion (the exp expansion's ldexp): one
; imadd by 2^23 from a register instead of the shift's bfi and an iadd. Other
; shifts past 7 keep the bfi (an address's shift is often loop-invariant)
; (L15-E457).

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: ldexp:
; CHECK:       convert f_to_s32, [[N:r[0-9]+]], u2, rtz
; CHECK:       mov_imm [[P:r[0-9]+]], 8388608
; CHECK:       imadd {{r[0-9]+}}, [[N]], [[P]], u3
define agx_kernel void @ldexp(ptr addrspace(1) %p, float %x, i32 %b) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %n = fptosi float %x to i32
  %fr = freeze i32 %n
  %s = shl i32 %fr, 23
  %r = add i32 %b, %s
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  store i32 %r, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: other:
; CHECK:       bfi
; CHECK-NOT:   imadd
; CHECK:       stop
define agx_kernel void @other(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %x = xor i32 %a, %t
  %s = shl i32 %x, 23
  %r = add i32 %b, %s
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  store i32 %r, ptr addrspace(1) %q
  ret void
}
