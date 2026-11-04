; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; 32-bit values whose bytes are bytes of others (constant byte permutes, PTX
; prmt) are built from 16-bit halves: each half one bit-field insert of a
; byte into the other byte's half, the halves the register's.

; bytes a0 a2 b0 b2 (prmt 0x6420): two 16-bit inserts
; CHECK-LABEL: even:
; CHECK: bfi r{{[0-9]+}}l, u2l, u2h, 8, mask 0xFF
; CHECK-NEXT: bfi r{{[0-9]+}}h, u3l, u3h, 8, mask 0xFF
; CHECK-NEXT: device_store
define agx_kernel void @even(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %a0 = and i32 %a, 255
  %a2 = lshr i32 %a, 8
  %a2m = and i32 %a2, 65280
  %b0 = shl i32 %b, 16
  %b0m = and i32 %b0, 16711680
  %b2 = shl i32 %b, 8
  %b2m = and i32 %b2, -16777216
  %t0 = or disjoint i32 %a0, %a2m
  %t1 = or disjoint i32 %t0, %b0m
  %r = or disjoint i32 %t1, %b2m
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; a sign-extended byte (prmt 0x8880, as InstCombine writes its sign fill):
; the sign extension in register (the signed byte on the ALUs)
; CHECK-LABEL: sext:
; CHECK: and r1, u2, 255
; CHECK-NEXT: xor r1, r1, 128
; CHECK-NEXT: isub r1, r1, 128
; CHECK-NEXT: device_store
define agx_kernel void @sext(ptr addrspace(1) %p, i32 %a) {
  %b0 = and i32 %a, 255
  %m = and i32 %a, 128
  %pos = icmp eq i32 %m, 0
  %s = select i1 %pos, i32 0, i32 -256
  %r = or disjoint i32 %b0, %s
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; a byte in every byte (prmt 0x0000): one insert, the half twice
; CHECK-LABEL: splat:
; CHECK: bfi r{{[0-9]+}}l, u2l, u2l, 8, mask 0xFF
; CHECK-NOT: {{and|or|shl|bfi}}
; CHECK: device_store
define agx_kernel void @splat(ptr addrspace(1) %p, i32 %a) {
  %b0 = and i32 %a, 255
  %b1 = shl i32 %b0, 8
  %b2 = shl i32 %b0, 16
  %b3 = shl i32 %a, 24
  %t0 = or disjoint i32 %b0, %b1
  %t1 = or disjoint i32 %t0, %b2
  %r = or disjoint i32 %t1, %b3
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; the halves of values as they are: left to the halves' patterns
; CHECK-LABEL: halves:
; CHECK-NOT: bfi
; CHECK: device_store
define agx_kernel void @halves(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %l = and i32 %a, 65535
  %h = shl i32 %b, 16
  %r = or disjoint i32 %l, %h
  store i32 %r, ptr addrspace(1) %p
  ret void
}
