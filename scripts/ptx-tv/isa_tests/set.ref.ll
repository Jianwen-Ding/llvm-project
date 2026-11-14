; set: the comparison of setp, then all ones / 1.0 / 0.

define i32 @set_lt_u32_f32(float %a, float %b) {
  %c = fcmp olt float %a, %b
  %d = select i1 %c, i32 -1, i32 0
  ret i32 %d
}

define float @set_ge_f32_s32(i32 %a, i32 %b) {
  %c = icmp sge i32 %a, %b
  %d = select i1 %c, float 1.0, float 0.0
  ret float %d
}

define i32 @set_ne_and_s32_u16(i16 %a, i16 %b, i32 %x) {
  %c = icmp ne i16 %a, %b
  %q = icmp ne i32 %x, 0
  %nq = xor i1 %q, true
  %r = and i1 %c, %nq
  %d = select i1 %r, i32 -1, i32 0
  ret i32 %d
}



