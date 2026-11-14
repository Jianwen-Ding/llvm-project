define i32 @vmad_0(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_1(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_2(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_3(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_4(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_5(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_6(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_7(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_8(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_9(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_10(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_11(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_12(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_13(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_14(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_15(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_16(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_17(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_18(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_19(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_20(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_21(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_22(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_23(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_24(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_25(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_26(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_27(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_28(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_29(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_30(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_31(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_32(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_33(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_34(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_35(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_36(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_37(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_38(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_39(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_40(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_41(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_42(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_43(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_44(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_45(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_46(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_47(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_48(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_49(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_50(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_51(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_52(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_53(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_54(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_55(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_56(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_57(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_58(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_59(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_60(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_61(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_62(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_63(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_64(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_65(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_66(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_67(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_68(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_69(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_70(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_71(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.sat.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_72(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_73(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_74(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_75(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_76(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_77(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_78(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_79(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_80(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_81(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_82(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_83(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.s32.s32.po.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_84(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_85(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_86(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_87(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_88(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_89(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_90(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_91(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_92(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_93(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_94(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_95(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_96(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_97(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_98(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_99(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_100(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_101(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_102(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_103(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_104(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_105(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_106(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_107(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_108(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_109(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_110(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_111(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_112(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_113(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_114(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_115(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_116(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_117(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_118(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_119(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_120(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_121(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_122(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_123(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_124(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_125(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_126(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_127(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_128(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_129(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_130(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_131(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_132(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_133(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_134(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_135(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_136(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_137(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_138(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_139(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_140(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_141(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_142(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_143(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_144(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_145(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_146(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_147(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_148(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_149(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_150(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_151(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_152(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_153(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_154(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_155(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.sat.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_156(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_157(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_158(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_159(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_160(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_161(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_162(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_163(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_164(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_165(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_166(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_167(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.u32.po.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_168(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_169(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_170(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_171(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_172(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_173(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_174(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_175(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_176(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_177(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_178(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_179(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_180(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_181(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_182(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_183(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_184(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_185(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_186(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_187(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_188(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_189(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_190(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_191(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_192(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_193(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_194(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_195(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_196(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_197(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_198(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_199(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_200(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_201(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_202(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_203(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_204(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_205(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_206(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_207(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_208(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_209(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_210(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_211(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_212(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_213(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_214(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_215(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_216(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_217(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_218(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_219(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_220(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_221(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_222(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_223(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_224(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_225(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_226(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_227(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_228(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_229(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_230(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_231(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_232(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_233(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_234(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_235(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_236(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_237(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_238(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_239(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.sat.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_240(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_241(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_242(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_243(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_244(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_245(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_246(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_247(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_248(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_249(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_250(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_251(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.s32.u32.s32.po.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_252(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_253(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_254(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_255(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_256(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_257(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_258(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_259(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_260(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_261(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_262(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_263(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_264(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_265(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_266(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_267(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_268(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_269(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_270(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_271(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_272(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_273(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_274(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, -$1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_275(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, -$1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_276(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_277(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_278(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_279(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_280(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_281(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_282(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_283(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_284(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_285(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_286(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, $1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_287(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, $1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_288(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_289(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_290(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_291(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_292(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_293(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_294(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_295(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_296(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_297(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_298(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, -$1, -$2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_299(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, -$1.h1, -$2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_300(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_301(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_302(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_303(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_304(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_305(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_306(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_307(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_308(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_309(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_310(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, $1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_311(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, $1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_312(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_313(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_314(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_315(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_316(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_317(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_318(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_319(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_320(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_321(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr7 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_322(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, -$1, $2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_323(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.sat.shr15 $0, -$1.h1, $2.b2, -$3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_324(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_325(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_326(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_327(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_328(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_329(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_330(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_331(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.sat $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_332(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.sat.shr7 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_333(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.sat.shr7 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_334(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.sat.shr15 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmad_335(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmad.u32.u32.s32.po.sat.shr15 $0, $1.h1, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
