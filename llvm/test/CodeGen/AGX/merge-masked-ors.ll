; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; A byte permute with a constant selector folds to one masked part per byte;
; the parts of one shifted value are merged (here b << 16 for bytes 2 and 3),
; so prmt 0x5410 (the low halves of a and b) is the register's halves: one
; 16-bit move.

; CHECK-LABEL: p5410:
; CHECK-NOT: {{^[[:space:]]+or[[:space:]]}}
; CHECK: bitop 0101, r{{[0-9]+}}h, r{{[0-9]+}}l, 0
; CHECK-NOT: {{^[[:space:]]+or[[:space:]]}}
; CHECK: device_store
define agx_kernel void @p5410(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load i32, ptr addrspace(1) %q
  %qb = getelementptr i8, ptr addrspace(1) %q, i64 4
  %b = load i32, ptr addrspace(1) %qb
  %byte2 = shl i32 %b, 16
  %p2 = and i32 %byte2, 16711680
  %byte3 = and i32 %byte2, -16777216
  %o01 = and i32 %a, 65535
  %t = or disjoint i32 %byte3, %o01
  %r = or disjoint i32 %t, %p2
  store i32 %r, ptr addrspace(1) %p
  ret void
}
