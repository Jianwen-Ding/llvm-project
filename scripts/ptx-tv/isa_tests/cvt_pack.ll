define i32 @pack_u16(i32 %a, i32 %b) {
  %d = call i32 asm "cvt.pack.sat.u16.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @pack_s16(i32 %a, i32 %b) {
  %d = call i32 asm "cvt.pack.sat.s16.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @pack_u8(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "cvt.pack.sat.u8.s32.b32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @pack_s8(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "cvt.pack.sat.s8.s32.b32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @pack_u4(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "cvt.pack.sat.u4.s32.b32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @pack_s4(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "cvt.pack.sat.s4.s32.b32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @pack_u2(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "cvt.pack.sat.u2.s32.b32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @pack_s2(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "cvt.pack.sat.s2.s32.b32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
