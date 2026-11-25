define i32 @set_lt_u32_f32(float %a, float %b) {
  %d = call i32 asm "set.lt.u32.f32 $0, $1, $2;", "=r,f,f"(float %a, float %b)
  ret i32 %d
}
define float @set_ge_f32_s32(i32 %a, i32 %b) {
  %d = call float asm "set.ge.f32.s32 $0, $1, $2;", "=f,r,r"(i32 %a, i32 %b)
  ret float %d
}
define i32 @set_ne_and_s32_u16(i16 %a, i16 %b, i32 %x) {
  %d = call i32 asm "{\0A\09.reg .pred q;\0A\09setp.ne.u32 q, $3, 0;\0A\09set.ne.and.s32.u16 $0, $1, $2, !q;\0A\09}", "=r,h,h,r"(i16 %a, i16 %b, i32 %x)
  ret i32 %d
}
