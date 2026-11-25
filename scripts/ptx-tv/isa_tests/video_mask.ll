define i32 @vadd2_m0(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32 $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m1(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32.add $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m2(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m3(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m4(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32.add $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m5(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m6(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32 $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m7(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32.add $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_m8(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.u32.sat $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m9(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32 $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m10(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32.add $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m11(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m12(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m13(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32.add $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m14(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m15(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32 $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m16(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32.add $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_m17(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.u32.sat $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m18(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32 $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m19(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32.add $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m20(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m21(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m22(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32.add $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m23(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m24(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32 $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m25(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32.add $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_m26(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.u32.sat $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m27(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32 $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m28(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32.add $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m29(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m30(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m31(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32.add $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m32(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m33(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32 $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m34(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32.add $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_m35(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.u32.sat $0.h10, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m36(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32 $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m37(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.add $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m38(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.sat $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m39(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m40(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.add $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m41(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m42(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32 $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m43(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.add $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m44(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.sat $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m45(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32 $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m46(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.add $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m47(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.sat $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m48(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32 $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m49(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.add $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_m50(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.u32.sat $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m51(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32 $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m52(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.add $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m53(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.sat $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m54(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m55(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.add $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m56(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m57(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32 $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m58(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.add $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m59(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.sat $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m60(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32 $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m61(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.add $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m62(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.sat $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m63(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32 $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m64(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.add $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_m65(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.u32.sat $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m66(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32 $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m67(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.add $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m68(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.sat $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m69(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m70(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.add $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m71(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m72(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32 $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m73(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.add $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m74(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.sat $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m75(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32 $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m76(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.add $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m77(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.sat $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m78(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32 $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m79(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.add $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_m80(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.u32.sat $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m81(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32 $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m82(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.add $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m83(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.sat $0.b0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m84(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m85(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.add $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m86(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m87(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32 $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m88(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.add $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m89(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.sat $0.b31, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m90(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32 $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m91(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.add $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m92(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.sat $0.b210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m93(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32 $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m94(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.add $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_m95(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.u32.sat $0.b3210, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
