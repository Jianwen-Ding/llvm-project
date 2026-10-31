; RUN: llc -mtriple=agx -mcpu=agx-g14c -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s

; An argument with "agx-hidden"="address32-base" (the base of the 32-bit
; addresses of a module of PTX's .address_size 32) is not a source argument:
; the record lists the two source arguments (8 bytes from half 0, 4 bytes from
; half 4) and ends with extension 9 and the base's first half, 6.
; CHECK: Hex dump of section '.agx.kernels':
; CHECK-NEXT: 0x00000000 4c000000 01000000 01000000 6b000000
; CHECK-NEXT: 0x00000010 {{[0-9a-f]+}} {{[0-9a-f]+}} ffffffff {{[0-9a-f]+}}
; CHECK-NEXT: 0x00000020 00000000 00000000 02000000 00000000
; CHECK-NEXT: 0x00000030 08000000 00000000 00000000 04000000
; CHECK-NEXT: 0x00000040 04000000 09000000 06000000
define agx_kernel void @k(ptr addrspace(1) %out, i32 %x, i64 "agx-hidden"="address32-base" %base) {
  %e = zext i32 %x to i64
  %a = add i64 %base, %e
  %p = inttoptr i64 %a to ptr addrspace(1)
  store i32 1, ptr addrspace(1) %p
  ret void
}
