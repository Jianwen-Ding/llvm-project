define i32 @slct_s32(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "slct.b32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
define i32 @slct_f32_ftz(i32 %a, i32 %b, float %c) {
  %d = call i32 asm "slct.ftz.b32.f32 $0, $1, $2, $3;", "=r,r,r,f"(i32 %a, i32 %b, float %c)
  ret i32 %d
}
define i64 @slct_f32_b64(i64 %a, i64 %b, float %c) {
  %d = call i64 asm "slct.b64.f32 $0, $1, $2, $3;", "=l,l,l,f"(i64 %a, i64 %b, float %c)
  ret i64 %d
}
define {i32, i32} @mad_lo_cc_u32(i32 %a, i32 %b, i32 %c) {
  %d = call {i32, i32} asm "mad.lo.cc.u32 $0, $2, $3, $4;\0A\09addc.u32 $1, 0, 0;", "=r,=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret {i32, i32} %d
}
define {i32, i32} @madc_hi_cc_u32(i32 %a, i32 %b, i32 %c, i32 %x, i32 %y) {
  %d = call {i32, i32} asm "add.cc.u32 $0, $5, $6;\0A\09madc.hi.cc.u32 $0, $2, $3, $4;\0A\09addc.u32 $1, 0, 0;", "=&r,=r,r,r,r,r,r"(i32 %a, i32 %b, i32 %c, i32 %x, i32 %y)
  ret {i32, i32} %d
}
define i64 @madc_lo_s64(i64 %a, i64 %b, i64 %c, i64 %x, i64 %y) {
  %d = call i64 asm "add.cc.u64 $0, $4, $5;\0A\09madc.lo.s64 $0, $1, $2, $3;", "=&l,l,l,l,l,l"(i64 %a, i64 %b, i64 %c, i64 %x, i64 %y)
  ret i64 %d
}
define i32 @mad_hi_sat_s32(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "mad.hi.sat.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
define i32 @mad24_hi_sat_s32(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "mad24.hi.sat.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
