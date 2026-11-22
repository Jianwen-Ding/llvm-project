; Reference semantics from the PTX ISA text (written independently of the
; lifter): slct selects a when c >= 0; mad.cc/madc carry out of / into the
; addition; .sat clamps the signed 32-bit sum to [INT_MIN, INT_MAX].

define i32 @slct_s32(i32 %a, i32 %b, i32 %c) {
  %neg = icmp slt i32 %c, 0
  %d = select i1 %neg, i32 %b, i32 %a
  ret i32 %d
}

; .ftz: a subnormal c becomes a zero of the same sign (so >= 0 holds);
; a NaN c gives an unspecified choice of a or b.
define i32 @slct_f32_ftz(i32 %a, i32 %b, float %c) {
  %bits = bitcast float %c to i32
  %exp = and i32 %bits, 2139095040
  %isden = icmp eq i32 %exp, 0
  %sign = and i32 %bits, -2147483648
  %fl = select i1 %isden, i32 %sign, i32 %bits
  %cf = bitcast i32 %fl to float
  %isnan = fcmp uno float %cf, %cf
  %lt = fcmp olt float %cf, 0.0
  %pick = select i1 %lt, i32 %b, i32 %a
  %any = freeze i1 poison
  %either = select i1 %any, i32 %a, i32 %b
  %d = select i1 %isnan, i32 %either, i32 %pick
  ret i32 %d
}

define i64 @slct_f32_b64(i64 %a, i64 %b, float %c) {
  %isnan = fcmp uno float %c, %c
  %lt = fcmp olt float %c, 0.0
  %pick = select i1 %lt, i64 %b, i64 %a
  %any = freeze i1 poison
  %either = select i1 %any, i64 %a, i64 %b
  %d = select i1 %isnan, i64 %either, i64 %pick
  ret i64 %d
}

define {i32, i32} @mad_lo_cc_u32(i32 %a, i32 %b, i32 %c) {
  %p = mul i32 %a, %b
  %s = add i32 %p, %c
  %cy = icmp ult i32 %s, %p
  %cz = zext i1 %cy to i32
  %r0 = insertvalue {i32, i32} poison, i32 %s, 0
  %r1 = insertvalue {i32, i32} %r0, i32 %cz, 1
  ret {i32, i32} %r1
}

define {i32, i32} @madc_hi_cc_u32(i32 %a, i32 %b, i32 %c, i32 %x, i32 %y) {
  %xs = add i32 %x, %y
  %cin = icmp ult i32 %xs, %x
  %a64 = zext i32 %a to i64
  %b64 = zext i32 %b to i64
  %p = mul i64 %a64, %b64
  %ph = lshr i64 %p, 32
  %c64 = zext i32 %c to i64
  %ci = zext i1 %cin to i64
  %t = add i64 %ph, %c64
  %t2 = add i64 %t, %ci
  %lo = trunc i64 %t2 to i32
  %hi = lshr i64 %t2, 32
  %cout = trunc i64 %hi to i32
  %r0 = insertvalue {i32, i32} poison, i32 %lo, 0
  %r1 = insertvalue {i32, i32} %r0, i32 %cout, 1
  ret {i32, i32} %r1
}

define i64 @madc_lo_s64(i64 %a, i64 %b, i64 %c, i64 %x, i64 %y) {
  %xs = add i64 %x, %y
  %cin = icmp ult i64 %xs, %x
  %p = mul i64 %a, %b
  %ci = zext i1 %cin to i64
  %t = add i64 %p, %c
  %d = add i64 %t, %ci
  ret i64 %d
}

define i32 @mad_hi_sat_s32(i32 %a, i32 %b, i32 %c) {
  %a64 = sext i32 %a to i64
  %b64 = sext i32 %b to i64
  %p = mul i64 %a64, %b64
  %ph = ashr i64 %p, 32
  %c64 = sext i32 %c to i64
  %s = add i64 %ph, %c64
  %big = icmp sgt i64 %s, 2147483647
  %small = icmp slt i64 %s, -2147483648
  %t = trunc i64 %s to i32
  %u = select i1 %big, i32 2147483647, i32 %t
  %d = select i1 %small, i32 -2147483648, i32 %u
  ret i32 %d
}

define i32 @mad24_hi_sat_s32(i32 %a, i32 %b, i32 %c) {
  %a24 = shl i32 %a, 8
  %a2 = ashr i32 %a24, 8
  %b24 = shl i32 %b, 8
  %b2 = ashr i32 %b24, 8
  %a64 = sext i32 %a2 to i64
  %b64 = sext i32 %b2 to i64
  %p = mul i64 %a64, %b64
  %ph = ashr i64 %p, 16
  %pt = trunc i64 %ph to i32
  %ps = sext i32 %pt to i64
  %c64 = sext i32 %c to i64
  %s = add i64 %ps, %c64
  %big = icmp sgt i64 %s, 2147483647
  %small = icmp slt i64 %s, -2147483648
  %t = trunc i64 %s to i32
  %u = select i1 %big, i32 2147483647, i32 %t
  %d = select i1 %small, i32 -2147483648, i32 %u
  ret i32 %d
}
