; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; 8- and 16-bit division and remainder: the float reciprocal of the divisor,
; the truncated product and one correction either way, no bit loop.

; CHECK-LABEL: udiv16:
; CHECK: rcp
; CHECK-NOT: while_icmp
; CHECK: stop
define agx_kernel void @udiv16(ptr addrspace(1) %p, i16 %a, i16 %b) {
  %q = udiv i16 %a, %b
  store i16 %q, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: srem8:
; CHECK: rcp
; CHECK-NOT: while_icmp
; CHECK: stop
define agx_kernel void @srem8(ptr addrspace(1) %p, i8 %a, i8 %b) {
  %r = srem i8 %a, %b
  store i8 %r, ptr addrspace(1) %p
  ret void
}

; by a constant: the 32-bit division, a multiply-high (no reciprocal)
; CHECK-LABEL: udiv16c:
; CHECK-NOT: rcp
; CHECK: imadd
; CHECK: stop
define agx_kernel void @udiv16c(ptr addrspace(1) %p, i16 %a) {
  %q = udiv i16 %a, 10
  store i16 %q, ptr addrspace(1) %p
  ret void
}

; ZLUDA's guards of a signed division (d = -1 ? 1 : d, or d + 1 <u 2 ? 1 : d,
; and selects of -x / all ones, 0 / x): the expansion divides by d itself,
; which gives those values, so neither the guard nor the selects remain.
; The conversions read the narrow values.

; CHECK-LABEL: sdiv16_guard:
; CHECK-NOT: iadd
; CHECK: convert s16_to_f
; CHECK: convert s16_to_f
; CHECK: rcp
; CHECK: convert f_to_s32
; CHECK-NEXT: mov_imm
; CHECK-NEXT: icmpsel ueq, {{.*}}, 0,
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @sdiv16_guard(ptr addrspace(1) %p, i16 %a, i16 %b) {
  %m1 = icmp eq i16 %b, -1
  %d = select i1 %m1, i16 1, i16 %b
  %q = sdiv i16 %a, %d
  %n = sub i16 0, %a
  %r = select i1 %m1, i16 %n, i16 %q
  store i16 %r, ptr addrspace(1) %p
  ret void
}

; the remainder in 8 bits, no select (x - q 0 = x)
; CHECK-LABEL: srem8_guard:
; CHECK: convert s8_to_f
; CHECK: convert s8_to_f
; CHECK-NOT: icmpsel
; CHECK: imadd
; CHECK-NEXT: isub
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @srem8_guard(ptr addrspace(1) %p, i8 %a, i8 %b) {
  %m1 = icmp eq i8 %b, -1
  %d = select i1 %m1, i8 1, i8 %b
  %q = srem i8 %a, %d
  %r = select i1 %m1, i8 0, i8 %q
  store i8 %r, ptr addrspace(1) %p
  ret void
}

; ZLUDA's div and rem of one pair after DivRemPairs: one select (d = 0's
; quotient)
; CHECK-LABEL: sdivrem16_zluda:
; CHECK-NOT: iadd
; CHECK: icmpsel ueq, {{.*}}, 0,
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @sdivrem16_zluda(ptr addrspace(1) %p, i16 %a, i16 %b) {
  %z = icmp eq i16 %b, 0
  %m1 = icmp eq i16 %b, -1
  %b1 = add i16 %b, 1
  %g = icmp ult i16 %b1, 2
  %d = select i1 %g, i16 1, i16 %b
  %af = freeze i16 %a
  %df = freeze i16 %d
  %q = sdiv i16 %af, %df
  %n = sub i16 0, %a
  %q1 = select i1 %m1, i16 %n, i16 %q
  %q2 = select i1 %z, i16 -1, i16 %q1
  %m = mul i16 %q, %df
  %r = sub i16 %af, %m
  %r1 = select i1 %m1, i16 0, i16 %r
  %r2 = select i1 %z, i16 %a, i16 %r1
  store i16 %q2, ptr addrspace(1) %p
  %p2 = getelementptr i16, ptr addrspace(1) %p, i32 1
  store i16 %r2, ptr addrspace(1) %p2
  ret void
}

; the quotient's selects lack d = 0's, which the guard takes: guarded as
; written (the guarded divisor is not zero, so no select by zero)
; CHECK-LABEL: sdiv16_nozero:
; CHECK: icmpsel ult
; CHECK: rcp
; CHECK: icmpsel ueq
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @sdiv16_nozero(ptr addrspace(1) %p, i16 %a, i16 %b) {
  %m1 = icmp eq i16 %b, -1
  %b1 = add i16 %b, 1
  %g = icmp ult i16 %b1, 2
  %d = select i1 %g, i16 1, i16 %b
  %q = sdiv i16 %a, %d
  %n = sub i16 0, %a
  %q1 = select i1 %m1, i16 %n, i16 %q
  store i16 %q1, ptr addrspace(1) %p
  ret void
}
