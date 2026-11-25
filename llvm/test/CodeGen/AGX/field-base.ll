; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-field-base-16=0 < %s | FileCheck %s --check-prefix=OFF

; A field of x under the bits a constant leaves clear, (x >> s) | c or
; ((x >> s) & m) | c, is one bfeil with c as its base, the field as wide as
; its known bits (a byte's high nibble under 0x6400, the half 1024: four
; bits, not the twelve of a 16-bit shift by 4; L15-E630).
; CHECK-LABEL: nibble:
; CHECK: bfeil {{r[0-9]+[lh]}}, {{r[0-9]+[lh]}}, {{r[0-9]+[lh]}}, 4, mask 0xF
; CHECK-NOT: or
; CHECK: device_store
; OFF-LABEL: nibble:
; OFF: bfeil {{r[0-9]+[lh]}}, 0, {{r[0-9]+[lh]}}, 4
; OFF: or
define agx_kernel void @nibble(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %t to i64
  %pp = getelementptr i8, ptr addrspace(1) %p, i64 %z
  %b = load i8, ptr addrspace(1) %pp
  %bz = zext i8 %b to i16
  %h = lshr i16 %bz, 4
  %r = or i16 %h, 25600
  %qq = getelementptr i16, ptr addrspace(1) %q, i64 %z
  store i16 %r, ptr addrspace(1) %qq
  ret void
}

; A masked field of a 16-bit value.
; CHECK-LABEL: masked:
; CHECK: bfeil {{r[0-9]+[lh]}}, {{r[0-9]+[lh]}}, {{r[0-9]+[lh]}}, 3, mask 0x3F
; CHECK-NOT: or
; CHECK: device_store
define agx_kernel void @masked(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %t to i64
  %pp = getelementptr i16, ptr addrspace(1) %p, i64 %z
  %x = load i16, ptr addrspace(1) %pp
  %h = lshr i16 %x, 3
  %m = and i16 %h, 63
  %r = or i16 %m, 3072
  %qq = getelementptr i16, ptr addrspace(1) %q, i64 %z
  store i16 %r, ptr addrspace(1) %qq
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
