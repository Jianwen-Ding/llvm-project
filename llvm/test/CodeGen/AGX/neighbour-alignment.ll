; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-vector-accesses < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-byte-pair-chains < %s | FileCheck %s --check-prefix=PAIR
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; Accesses of one base in a block that run together prove each other's
; alignment: the bytes after a 2-byte aligned f16 (llama.cpp's q4_0 block)
; are 2-byte aligned at even offsets. With -agx-byte-pair-chains such a pair
; is one access of the byte format.

; CHECK-LABEL: @block(
; CHECK: load half, ptr addrspace(1) %p, align 2
; CHECK: load i8, ptr addrspace(1) %q2, align 2
; CHECK: load i8, ptr addrspace(1) %q3, align 1
; CHECK: load i8, ptr addrspace(1) %q4, align 2
; PAIR-LABEL: block:
; PAIR: device_load {{[01]}}, i8, xy,
; PAIR-NOT: device_load {{[01]}}, i8, x,
define void @block(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %d = load half, ptr addrspace(1) %p, align 2
  %q2 = getelementptr i8, ptr addrspace(1) %p, i64 2
  %q3 = getelementptr i8, ptr addrspace(1) %p, i64 3
  %q4 = getelementptr i8, ptr addrspace(1) %p, i64 4
  %q5 = getelementptr i8, ptr addrspace(1) %p, i64 5
  %b2 = load i8, ptr addrspace(1) %q2, align 1
  %b3 = load i8, ptr addrspace(1) %q3, align 1
  %b4 = load i8, ptr addrspace(1) %q4, align 1
  %b5 = load i8, ptr addrspace(1) %q5, align 1
  %x2 = zext i8 %b2 to i32
  %x3 = zext i8 %b3 to i32
  %x4 = zext i8 %b4 to i32
  %x5 = zext i8 %b5 to i32
  %s1 = add i32 %x2, %x3
  %s2 = add i32 %x4, %x5
  %s = add i32 %s1, %s2
  %f = uitofp i32 %s to float
  %df = fpext half %d to float
  %r = fmul float %f, %df
  store float %r, ptr addrspace(1) %o
  ret void
}


; A 16-bit value zero-extended and shifted right: shifted in its half.
; CHECK-LABEL: @shift(
; PAIR-LABEL: shift:
; PAIR-NOT: bitop 0101
; PAIR: device_store
define void @shift(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %h = load i16, ptr addrspace(1) %p, align 2
  %w = zext i16 %h to i32
  %s = lshr i32 %w, 4
  %t = and i32 %s, 15
  %f = sitofp i32 %t to float
  store float %f, ptr addrspace(1) %o
  ret void
}
