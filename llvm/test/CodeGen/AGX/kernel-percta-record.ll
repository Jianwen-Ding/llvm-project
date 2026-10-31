; RUN: llc -mtriple=agx -mcpu=agx-g14c -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s

; An argument with "agx-hidden"="perctamemory:<size>" (PTX's per-CTA global
; memory, .minperctamemory) is not a source argument: the record lists one
; argument and ends with extension 8, the argument's first half and the
; slice size (2048 = 0x800).
; CHECK: Hex dump of section '.agx.kernels':
; CHECK-NEXT: 0x00000000 44000000 01000000 01000000 6b000000
; CHECK-NEXT: 0x00000010 {{[0-9a-f]+}} {{[0-9a-f]+}} ffffffff {{[0-9a-f]+}}
; CHECK-NEXT: 0x00000020 00000000 00000000 01000000 00000000
; CHECK-NEXT: 0x00000030 08000000 00000000 08000000 04000000
; CHECK-NEXT: 0x00000040 00080000
define agx_kernel void @k(ptr addrspace(1) %out, i64 "agx-hidden"="perctamemory:2048" %base) {
  store i64 %base, ptr addrspace(1) %out
  ret void
}
