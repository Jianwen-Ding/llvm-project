; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-uniform-constants -agx-uniform-constants-16=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-uniform-constants -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s --check-prefix=SECTION

; A 16-bit constant that no immediate holds is preloaded too (L15-E644): the
; low half of a 32-bit uniform holding it zero-extended, in extension 6 as
; any other, read as that half (half compares against an exponential's bounds,
; an i16 mask).
; CHECK-LABEL: uconst16:
; CHECK-NOT:     mov_imm
; CHECK:         xor r{{[0-9]+}}l, r{{[0-9]+}}l, u7l
; CHECK-NEXT:    imadd r{{[0-9]+}}l, r{{[0-9]+}}l, u8l, 0
; CHECK-NEXT:    xor r{{[0-9]+}}l, r{{[0-9]+}}l, u7l
; OFF-LABEL: uconst16:
; OFF:           mov_imm r{{[0-9]+}}{{[lh]}}, 4660
; OFF:           mov_imm r{{[0-9]+}}{{[lh]}}, 291
; Extension 6, 2 constants: half 14 = 0x1234, half 16 = 0x0123.
; SECTION: 06000000 02000000
; SECTION: 0e000000 34120000 10000000 23010000
define agx_kernel void @uconst16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr i16, ptr addrspace(1) %in, i32 %i
  %v = load i16, ptr addrspace(1) %p
  %x = xor i16 %v, 4660
  %y = mul i16 %x, 291
  %z = xor i16 %y, 4660
  %q = getelementptr i16, ptr addrspace(1) %out, i32 %i
  store i16 %z, ptr addrspace(1) %q
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
