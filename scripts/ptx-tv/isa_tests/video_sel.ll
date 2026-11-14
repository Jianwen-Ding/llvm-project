define i32 @vadd_0(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_1(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_2(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_3(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_4(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_5(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_6(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_7(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_8(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_9(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_10(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_11(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_12(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_13(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_14(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_15(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_16(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_17(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_18(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_19(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_20(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_21(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_22(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_23(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.s32.s32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_24(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_25(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_26(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_27(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_28(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_29(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_30(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_31(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_32(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_33(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_34(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_35(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_36(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_37(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_38(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_39(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_40(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_41(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_42(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_43(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vadd_44(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_45(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_46(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd_47(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd.u32.u32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_48(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_49(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_50(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_51(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_52(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_53(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_54(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_55(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_56(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_57(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_58(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_59(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_60(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_61(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_62(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_63(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_64(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_65(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_66(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_67(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_68(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_69(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_70(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_71(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.s32.s32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_72(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_73(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_74(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_75(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_76(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_77(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_78(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_79(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_80(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_81(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_82(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_83(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_84(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_85(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_86(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_87(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_88(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_89(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_90(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_91(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vsub_92(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_93(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_94(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub_95(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub.u32.u32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_96(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_97(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_98(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_99(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_100(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_101(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_102(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_103(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_104(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_105(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_106(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_107(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_108(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_109(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_110(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_111(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_112(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_113(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_114(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_115(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_116(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_117(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_118(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_119(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.s32.s32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_120(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_121(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_122(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_123(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_124(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_125(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_126(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_127(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_128(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_129(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_130(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_131(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_132(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_133(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_134(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_135(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_136(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_137(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_138(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_139(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vabsdiff_140(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_141(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_142(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff_143(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff.u32.u32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_144(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_145(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_146(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_147(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_148(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_149(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_150(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_151(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_152(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_153(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_154(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_155(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_156(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_157(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_158(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_159(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_160(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_161(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_162(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_163(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_164(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_165(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_166(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_167(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.s32.s32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_168(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_169(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_170(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_171(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_172(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_173(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_174(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_175(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_176(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_177(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_178(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_179(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_180(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_181(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_182(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_183(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_184(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_185(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_186(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_187(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmin_188(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_189(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_190(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin_191(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin.u32.u32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_192(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_193(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_194(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_195(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_196(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_197(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_198(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_199(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_200(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_201(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_202(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_203(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_204(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_205(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_206(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_207(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_208(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_209(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_210(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_211(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_212(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_213(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_214(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_215(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.s32.s32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_216(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_217(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_218(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_219(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.b2, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_220(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_221(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.h1, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_222(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_223(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0, $1.b1, $2.h0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_224(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_225(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.b2, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_226(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_227(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.h1, $1.b1, $2.h0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_228(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_229(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0, $1.h1, $2.b3;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_230(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_231(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.b2, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_232(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_233(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.h1, $1.h1, $2.b3, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_234(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_235(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0, $1.b0, $2.b2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vmax_236(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_237(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.b2, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_238(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32 $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax_239(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax.u32.u32.s32.sat $0.h1, $1.b0, $2.b2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_240(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_241(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_242(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_243(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_244(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_245(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_246(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_247(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.clamp.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_248(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_249(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_250(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_251(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_252(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_253(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_254(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_255(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.s32.u32.wrap.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_256(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_257(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_258(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_259(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_260(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_261(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_262(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_263(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.clamp.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_264(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_265(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_266(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_267(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_268(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_269(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_270(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_271(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.u32.u32.u32.wrap.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_272(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_273(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_274(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_275(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_276(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_277(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_278(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_279(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.clamp.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_280(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_281(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_282(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_283(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_284(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_285(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshl_286(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshl_287(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshl.s32.u32.u32.wrap.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_288(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_289(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_290(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_291(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_292(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_293(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_294(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_295(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.clamp.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_296(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_297(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_298(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_299(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_300(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_301(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_302(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_303(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.s32.u32.wrap.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_304(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_305(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_306(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_307(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_308(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_309(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_310(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_311(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.clamp.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_312(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_313(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_314(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_315(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_316(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_317(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_318(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_319(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.u32.u32.u32.wrap.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_320(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_321(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_322(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_323(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_324(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_325(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_326(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_327(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.clamp.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_328(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_329(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap.sat $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_330(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_331(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap.sat $0.h0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_332(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_333(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap.sat $0, $1.h1, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vshr_334(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vshr_335(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vshr.s32.u32.u32.wrap.sat $0.h0, $1.h1, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset_336(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.lt $0, $1.b3, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vset_337(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.lt $0.b1, $1.b3, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset_338(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.lt $0, $1.h0, $2.h1;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vset_339(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.lt $0.b1, $1.h0, $2.h1, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset_340(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ge $0, $1.b3, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vset_341(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ge $0.b1, $1.b3, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset_342(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ge $0, $1.h0, $2.h1;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vset_343(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ge $0.b1, $1.h0, $2.h1, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset_344(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ne $0, $1.b3, $2.b0;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vset_345(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ne $0.b1, $1.b3, $2.b0, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset_346(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ne $0, $1.h0, $2.h1;", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %d
}

define i32 @vset_347(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset.s32.u32.ne $0.b1, $1.h0, $2.h1, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
