; Scalar video instructions with operand selectors (.b0-.b3/.h0/.h1),
; destination selectors (merge into c; .sat clamps to the field's range)
; and vshl/vshr (.clamp: amount min 32, .wrap: amount mod 32), from the
; PTX ISA pseudo-code, in 128-bit arithmetic.

define i32 @vadd_0(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  ret i32 %v4
}

define i32 @vadd_1(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = icmp slt i128 %v3, -2147483648
  %v5 = select i1 %v4, i128 -2147483648, i128 %v3
  %v6 = icmp sgt i128 %v5, 2147483647
  %v7 = select i1 %v6, i128 2147483647, i128 %v5
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_2(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i8
  %v6 = zext i8 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, -16711681
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vadd_3(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = icmp slt i128 %v3, -128
  %v5 = select i1 %v4, i128 -128, i128 %v3
  %v6 = icmp sgt i128 %v5, 127
  %v7 = select i1 %v6, i128 127, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_4(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i16
  %v6 = zext i16 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, 65535
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vadd_5(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = icmp slt i128 %v3, -32768
  %v5 = select i1 %v4, i128 -32768, i128 %v3
  %v6 = icmp sgt i128 %v5, 32767
  %v7 = select i1 %v6, i128 32767, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_6(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_7(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -2147483648
  %v9 = select i1 %v8, i128 -2147483648, i128 %v7
  %v10 = icmp sgt i128 %v9, 2147483647
  %v11 = select i1 %v10, i128 2147483647, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vadd_8(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_9(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -128
  %v9 = select i1 %v8, i128 -128, i128 %v7
  %v10 = icmp sgt i128 %v9, 127
  %v11 = select i1 %v10, i128 127, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_10(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_11(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -32768
  %v9 = select i1 %v8, i128 -32768, i128 %v7
  %v10 = icmp sgt i128 %v9, 32767
  %v11 = select i1 %v10, i128 32767, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_12(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_13(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -2147483648
  %v9 = select i1 %v8, i128 -2147483648, i128 %v7
  %v10 = icmp sgt i128 %v9, 2147483647
  %v11 = select i1 %v10, i128 2147483647, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vadd_14(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_15(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -128
  %v9 = select i1 %v8, i128 -128, i128 %v7
  %v10 = icmp sgt i128 %v9, 127
  %v11 = select i1 %v10, i128 127, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_16(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_17(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -32768
  %v9 = select i1 %v8, i128 -32768, i128 %v7
  %v10 = icmp sgt i128 %v9, 32767
  %v11 = select i1 %v10, i128 32767, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_18(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_19(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -2147483648
  %v9 = select i1 %v8, i128 -2147483648, i128 %v7
  %v10 = icmp sgt i128 %v9, 2147483647
  %v11 = select i1 %v10, i128 2147483647, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vadd_20(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_21(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -128
  %v9 = select i1 %v8, i128 -128, i128 %v7
  %v10 = icmp sgt i128 %v9, 127
  %v11 = select i1 %v10, i128 127, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_22(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_23(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -32768
  %v9 = select i1 %v8, i128 -32768, i128 %v7
  %v10 = icmp sgt i128 %v9, 32767
  %v11 = select i1 %v10, i128 32767, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_24(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  ret i32 %v4
}

define i32 @vadd_25(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = select i1 %v4, i128 0, i128 %v3
  %v6 = icmp sgt i128 %v5, 4294967295
  %v7 = select i1 %v6, i128 4294967295, i128 %v5
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_26(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i8
  %v6 = zext i8 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, -16711681
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vadd_27(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = select i1 %v4, i128 0, i128 %v3
  %v6 = icmp sgt i128 %v5, 255
  %v7 = select i1 %v6, i128 255, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_28(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i16
  %v6 = zext i16 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, 65535
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vadd_29(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = add i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = select i1 %v4, i128 0, i128 %v3
  %v6 = icmp sgt i128 %v5, 65535
  %v7 = select i1 %v6, i128 65535, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_30(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_31(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 4294967295
  %v11 = select i1 %v10, i128 4294967295, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vadd_32(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_33(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 255
  %v11 = select i1 %v10, i128 255, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_34(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_35(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 65535
  %v11 = select i1 %v10, i128 65535, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_36(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_37(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 4294967295
  %v11 = select i1 %v10, i128 4294967295, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vadd_38(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_39(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 255
  %v11 = select i1 %v10, i128 255, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_40(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_41(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 65535
  %v11 = select i1 %v10, i128 65535, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_42(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_43(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 4294967295
  %v11 = select i1 %v10, i128 4294967295, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vadd_44(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_45(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 255
  %v11 = select i1 %v10, i128 255, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vadd_46(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vadd_47(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = add i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 65535
  %v11 = select i1 %v10, i128 65535, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_48(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  ret i32 %v4
}

define i32 @vsub_49(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, -2147483648
  %v5 = select i1 %v4, i128 -2147483648, i128 %v3
  %v6 = icmp sgt i128 %v5, 2147483647
  %v7 = select i1 %v6, i128 2147483647, i128 %v5
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_50(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i8
  %v6 = zext i8 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, -16711681
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vsub_51(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, -128
  %v5 = select i1 %v4, i128 -128, i128 %v3
  %v6 = icmp sgt i128 %v5, 127
  %v7 = select i1 %v6, i128 127, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_52(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i16
  %v6 = zext i16 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, 65535
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vsub_53(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, -32768
  %v5 = select i1 %v4, i128 -32768, i128 %v3
  %v6 = icmp sgt i128 %v5, 32767
  %v7 = select i1 %v6, i128 32767, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_54(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_55(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -2147483648
  %v9 = select i1 %v8, i128 -2147483648, i128 %v7
  %v10 = icmp sgt i128 %v9, 2147483647
  %v11 = select i1 %v10, i128 2147483647, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vsub_56(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_57(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -128
  %v9 = select i1 %v8, i128 -128, i128 %v7
  %v10 = icmp sgt i128 %v9, 127
  %v11 = select i1 %v10, i128 127, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_58(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_59(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -32768
  %v9 = select i1 %v8, i128 -32768, i128 %v7
  %v10 = icmp sgt i128 %v9, 32767
  %v11 = select i1 %v10, i128 32767, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_60(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_61(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -2147483648
  %v9 = select i1 %v8, i128 -2147483648, i128 %v7
  %v10 = icmp sgt i128 %v9, 2147483647
  %v11 = select i1 %v10, i128 2147483647, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vsub_62(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_63(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -128
  %v9 = select i1 %v8, i128 -128, i128 %v7
  %v10 = icmp sgt i128 %v9, 127
  %v11 = select i1 %v10, i128 127, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_64(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_65(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -32768
  %v9 = select i1 %v8, i128 -32768, i128 %v7
  %v10 = icmp sgt i128 %v9, 32767
  %v11 = select i1 %v10, i128 32767, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_66(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_67(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -2147483648
  %v9 = select i1 %v8, i128 -2147483648, i128 %v7
  %v10 = icmp sgt i128 %v9, 2147483647
  %v11 = select i1 %v10, i128 2147483647, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vsub_68(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_69(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -128
  %v9 = select i1 %v8, i128 -128, i128 %v7
  %v10 = icmp sgt i128 %v9, 127
  %v11 = select i1 %v10, i128 127, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_70(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_71(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, -32768
  %v9 = select i1 %v8, i128 -32768, i128 %v7
  %v10 = icmp sgt i128 %v9, 32767
  %v11 = select i1 %v10, i128 32767, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_72(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  ret i32 %v4
}

define i32 @vsub_73(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = select i1 %v4, i128 0, i128 %v3
  %v6 = icmp sgt i128 %v5, 4294967295
  %v7 = select i1 %v6, i128 4294967295, i128 %v5
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_74(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i8
  %v6 = zext i8 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, -16711681
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vsub_75(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = select i1 %v4, i128 0, i128 %v3
  %v6 = icmp sgt i128 %v5, 255
  %v7 = select i1 %v6, i128 255, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_76(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = trunc i128 %v3 to i32
  %v5 = trunc i32 %v4 to i16
  %v6 = zext i16 %v5 to i32
  %v7 = shl i32 %v6, 16
  %v8 = and i32 %c, 65535
  %v9 = or i32 %v8, %v7
  ret i32 %v9
}

define i32 @vsub_77(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = select i1 %v4, i128 0, i128 %v3
  %v6 = icmp sgt i128 %v5, 65535
  %v7 = select i1 %v6, i128 65535, i128 %v5
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_78(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_79(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 4294967295
  %v11 = select i1 %v10, i128 4294967295, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vsub_80(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_81(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 255
  %v11 = select i1 %v10, i128 255, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_82(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_83(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 65535
  %v11 = select i1 %v10, i128 65535, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_84(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_85(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 4294967295
  %v11 = select i1 %v10, i128 4294967295, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vsub_86(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_87(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 255
  %v11 = select i1 %v10, i128 255, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_88(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_89(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 65535
  %v11 = select i1 %v10, i128 65535, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_90(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_91(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 4294967295
  %v11 = select i1 %v10, i128 4294967295, i128 %v9
  %v12 = trunc i128 %v11 to i32
  ret i32 %v12
}

define i32 @vsub_92(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i8
  %v10 = zext i8 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, -16711681
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_93(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 255
  %v11 = select i1 %v10, i128 255, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i8
  %v14 = zext i8 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, -16711681
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vsub_94(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = trunc i128 %v7 to i32
  %v9 = trunc i32 %v8 to i16
  %v10 = zext i16 %v9 to i32
  %v11 = shl i32 %v10, 16
  %v12 = and i32 %c, 65535
  %v13 = or i32 %v12, %v11
  ret i32 %v13
}

define i32 @vsub_95(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = select i1 %v8, i128 0, i128 %v7
  %v10 = icmp sgt i128 %v9, 65535
  %v11 = select i1 %v10, i128 65535, i128 %v9
  %v12 = trunc i128 %v11 to i32
  %v13 = trunc i32 %v12 to i16
  %v14 = zext i16 %v13 to i32
  %v15 = shl i32 %v14, 16
  %v16 = and i32 %c, 65535
  %v17 = or i32 %v16, %v15
  ret i32 %v17
}

define i32 @vabsdiff_96(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = trunc i128 %v6 to i32
  ret i32 %v7
}

define i32 @vabsdiff_97(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = icmp slt i128 %v6, -2147483648
  %v8 = select i1 %v7, i128 -2147483648, i128 %v6
  %v9 = icmp sgt i128 %v8, 2147483647
  %v10 = select i1 %v9, i128 2147483647, i128 %v8
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_98(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = trunc i128 %v6 to i32
  %v8 = trunc i32 %v7 to i8
  %v9 = zext i8 %v8 to i32
  %v10 = shl i32 %v9, 16
  %v11 = and i32 %c, -16711681
  %v12 = or i32 %v11, %v10
  ret i32 %v12
}

define i32 @vabsdiff_99(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = icmp slt i128 %v6, -128
  %v8 = select i1 %v7, i128 -128, i128 %v6
  %v9 = icmp sgt i128 %v8, 127
  %v10 = select i1 %v9, i128 127, i128 %v8
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_100(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = trunc i128 %v6 to i32
  %v8 = trunc i32 %v7 to i16
  %v9 = zext i16 %v8 to i32
  %v10 = shl i32 %v9, 16
  %v11 = and i32 %c, 65535
  %v12 = or i32 %v11, %v10
  ret i32 %v12
}

define i32 @vabsdiff_101(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = icmp slt i128 %v6, -32768
  %v8 = select i1 %v7, i128 -32768, i128 %v6
  %v9 = icmp sgt i128 %v8, 32767
  %v10 = select i1 %v9, i128 32767, i128 %v8
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_102(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_103(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -2147483648
  %v12 = select i1 %v11, i128 -2147483648, i128 %v10
  %v13 = icmp sgt i128 %v12, 2147483647
  %v14 = select i1 %v13, i128 2147483647, i128 %v12
  %v15 = trunc i128 %v14 to i32
  ret i32 %v15
}

define i32 @vabsdiff_104(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_105(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -128
  %v12 = select i1 %v11, i128 -128, i128 %v10
  %v13 = icmp sgt i128 %v12, 127
  %v14 = select i1 %v13, i128 127, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i8
  %v17 = zext i8 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, -16711681
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_106(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_107(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -32768
  %v12 = select i1 %v11, i128 -32768, i128 %v10
  %v13 = icmp sgt i128 %v12, 32767
  %v14 = select i1 %v13, i128 32767, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i16
  %v17 = zext i16 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, 65535
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_108(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_109(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -2147483648
  %v12 = select i1 %v11, i128 -2147483648, i128 %v10
  %v13 = icmp sgt i128 %v12, 2147483647
  %v14 = select i1 %v13, i128 2147483647, i128 %v12
  %v15 = trunc i128 %v14 to i32
  ret i32 %v15
}

define i32 @vabsdiff_110(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_111(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -128
  %v12 = select i1 %v11, i128 -128, i128 %v10
  %v13 = icmp sgt i128 %v12, 127
  %v14 = select i1 %v13, i128 127, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i8
  %v17 = zext i8 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, -16711681
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_112(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_113(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -32768
  %v12 = select i1 %v11, i128 -32768, i128 %v10
  %v13 = icmp sgt i128 %v12, 32767
  %v14 = select i1 %v13, i128 32767, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i16
  %v17 = zext i16 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, 65535
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_114(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_115(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -2147483648
  %v12 = select i1 %v11, i128 -2147483648, i128 %v10
  %v13 = icmp sgt i128 %v12, 2147483647
  %v14 = select i1 %v13, i128 2147483647, i128 %v12
  %v15 = trunc i128 %v14 to i32
  ret i32 %v15
}

define i32 @vabsdiff_116(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_117(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -128
  %v12 = select i1 %v11, i128 -128, i128 %v10
  %v13 = icmp sgt i128 %v12, 127
  %v14 = select i1 %v13, i128 127, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i8
  %v17 = zext i8 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, -16711681
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_118(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_119(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, -32768
  %v12 = select i1 %v11, i128 -32768, i128 %v10
  %v13 = icmp sgt i128 %v12, 32767
  %v14 = select i1 %v13, i128 32767, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i16
  %v17 = zext i16 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, 65535
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_120(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = trunc i128 %v6 to i32
  ret i32 %v7
}

define i32 @vabsdiff_121(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = icmp slt i128 %v6, 0
  %v8 = select i1 %v7, i128 0, i128 %v6
  %v9 = icmp sgt i128 %v8, 4294967295
  %v10 = select i1 %v9, i128 4294967295, i128 %v8
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_122(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = trunc i128 %v6 to i32
  %v8 = trunc i32 %v7 to i8
  %v9 = zext i8 %v8 to i32
  %v10 = shl i32 %v9, 16
  %v11 = and i32 %c, -16711681
  %v12 = or i32 %v11, %v10
  ret i32 %v12
}

define i32 @vabsdiff_123(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = icmp slt i128 %v6, 0
  %v8 = select i1 %v7, i128 0, i128 %v6
  %v9 = icmp sgt i128 %v8, 255
  %v10 = select i1 %v9, i128 255, i128 %v8
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_124(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = trunc i128 %v6 to i32
  %v8 = trunc i32 %v7 to i16
  %v9 = zext i16 %v8 to i32
  %v10 = shl i32 %v9, 16
  %v11 = and i32 %c, 65535
  %v12 = or i32 %v11, %v10
  ret i32 %v12
}

define i32 @vabsdiff_125(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = sub i128 %v1, %v2
  %v4 = icmp slt i128 %v3, 0
  %v5 = sub i128 0, %v3
  %v6 = select i1 %v4, i128 %v5, i128 %v3
  %v7 = icmp slt i128 %v6, 0
  %v8 = select i1 %v7, i128 0, i128 %v6
  %v9 = icmp sgt i128 %v8, 65535
  %v10 = select i1 %v9, i128 65535, i128 %v8
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_126(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_127(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 4294967295
  %v14 = select i1 %v13, i128 4294967295, i128 %v12
  %v15 = trunc i128 %v14 to i32
  ret i32 %v15
}

define i32 @vabsdiff_128(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_129(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 255
  %v14 = select i1 %v13, i128 255, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i8
  %v17 = zext i8 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, -16711681
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_130(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_131(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 65535
  %v14 = select i1 %v13, i128 65535, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i16
  %v17 = zext i16 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, 65535
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_132(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_133(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 4294967295
  %v14 = select i1 %v13, i128 4294967295, i128 %v12
  %v15 = trunc i128 %v14 to i32
  ret i32 %v15
}

define i32 @vabsdiff_134(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_135(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 255
  %v14 = select i1 %v13, i128 255, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i8
  %v17 = zext i8 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, -16711681
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_136(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_137(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 65535
  %v14 = select i1 %v13, i128 65535, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i16
  %v17 = zext i16 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, 65535
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_138(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_139(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 4294967295
  %v14 = select i1 %v13, i128 4294967295, i128 %v12
  %v15 = trunc i128 %v14 to i32
  ret i32 %v15
}

define i32 @vabsdiff_140(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i8
  %v13 = zext i8 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, -16711681
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_141(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 255
  %v14 = select i1 %v13, i128 255, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i8
  %v17 = zext i8 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, -16711681
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vabsdiff_142(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = trunc i128 %v10 to i32
  %v12 = trunc i32 %v11 to i16
  %v13 = zext i16 %v12 to i32
  %v14 = shl i32 %v13, 16
  %v15 = and i32 %c, 65535
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @vabsdiff_143(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = sub i128 %v3, %v6
  %v8 = icmp slt i128 %v7, 0
  %v9 = sub i128 0, %v7
  %v10 = select i1 %v8, i128 %v9, i128 %v7
  %v11 = icmp slt i128 %v10, 0
  %v12 = select i1 %v11, i128 0, i128 %v10
  %v13 = icmp sgt i128 %v12, 65535
  %v14 = select i1 %v13, i128 65535, i128 %v12
  %v15 = trunc i128 %v14 to i32
  %v16 = trunc i32 %v15 to i16
  %v17 = zext i16 %v16 to i32
  %v18 = shl i32 %v17, 16
  %v19 = and i32 %c, 65535
  %v20 = or i32 %v19, %v18
  ret i32 %v20
}

define i32 @vmin_144(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vmin_145(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, -2147483648
  %v6 = select i1 %v5, i128 -2147483648, i128 %v4
  %v7 = icmp sgt i128 %v6, 2147483647
  %v8 = select i1 %v7, i128 2147483647, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_146(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i8
  %v7 = zext i8 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, -16711681
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmin_147(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, -128
  %v6 = select i1 %v5, i128 -128, i128 %v4
  %v7 = icmp sgt i128 %v6, 127
  %v8 = select i1 %v7, i128 127, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_148(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, 65535
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmin_149(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, -32768
  %v6 = select i1 %v5, i128 -32768, i128 %v4
  %v7 = icmp sgt i128 %v6, 32767
  %v8 = select i1 %v7, i128 32767, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_150(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_151(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmin_152(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_153(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -128
  %v10 = select i1 %v9, i128 -128, i128 %v8
  %v11 = icmp sgt i128 %v10, 127
  %v12 = select i1 %v11, i128 127, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_154(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_155(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_156(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_157(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmin_158(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_159(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -128
  %v10 = select i1 %v9, i128 -128, i128 %v8
  %v11 = icmp sgt i128 %v10, 127
  %v12 = select i1 %v11, i128 127, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_160(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_161(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_162(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_163(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmin_164(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_165(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -128
  %v10 = select i1 %v9, i128 -128, i128 %v8
  %v11 = icmp sgt i128 %v10, 127
  %v12 = select i1 %v11, i128 127, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_166(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_167(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_168(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vmin_169(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 4294967295
  %v8 = select i1 %v7, i128 4294967295, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_170(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i8
  %v7 = zext i8 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, -16711681
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmin_171(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 255
  %v8 = select i1 %v7, i128 255, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_172(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, 65535
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmin_173(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp slt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 65535
  %v8 = select i1 %v7, i128 65535, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_174(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_175(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmin_176(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_177(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 255
  %v12 = select i1 %v11, i128 255, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_178(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_179(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_180(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_181(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmin_182(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_183(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 255
  %v12 = select i1 %v11, i128 255, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_184(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_185(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_186(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_187(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmin_188(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_189(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 255
  %v12 = select i1 %v11, i128 255, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmin_190(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmin_191(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_192(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vmax_193(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, -2147483648
  %v6 = select i1 %v5, i128 -2147483648, i128 %v4
  %v7 = icmp sgt i128 %v6, 2147483647
  %v8 = select i1 %v7, i128 2147483647, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_194(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i8
  %v7 = zext i8 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, -16711681
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmax_195(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, -128
  %v6 = select i1 %v5, i128 -128, i128 %v4
  %v7 = icmp sgt i128 %v6, 127
  %v8 = select i1 %v7, i128 127, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_196(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, 65535
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmax_197(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, -32768
  %v6 = select i1 %v5, i128 -32768, i128 %v4
  %v7 = icmp sgt i128 %v6, 32767
  %v8 = select i1 %v7, i128 32767, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_198(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_199(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmax_200(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_201(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -128
  %v10 = select i1 %v9, i128 -128, i128 %v8
  %v11 = icmp sgt i128 %v10, 127
  %v12 = select i1 %v11, i128 127, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_202(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_203(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_204(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_205(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmax_206(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_207(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -128
  %v10 = select i1 %v9, i128 -128, i128 %v8
  %v11 = icmp sgt i128 %v10, 127
  %v12 = select i1 %v11, i128 127, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_208(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_209(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_210(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_211(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmax_212(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_213(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -128
  %v10 = select i1 %v9, i128 -128, i128 %v8
  %v11 = icmp sgt i128 %v10, 127
  %v12 = select i1 %v11, i128 127, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_214(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_215(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_216(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vmax_217(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 4294967295
  %v8 = select i1 %v7, i128 4294967295, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_218(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i8
  %v7 = zext i8 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, -16711681
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmax_219(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 255
  %v8 = select i1 %v7, i128 255, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_220(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 16
  %v9 = and i32 %c, 65535
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vmax_221(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = sext i32 %b to i128
  %v3 = icmp sgt i128 %v1, %v2
  %v4 = select i1 %v3, i128 %v1, i128 %v2
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 65535
  %v8 = select i1 %v7, i128 65535, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_222(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_223(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmax_224(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_225(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 255
  %v12 = select i1 %v11, i128 255, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_226(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_227(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 16
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_228(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_229(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmax_230(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_231(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 255
  %v12 = select i1 %v11, i128 255, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_232(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_233(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_234(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_235(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vmax_236(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, -16711681
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_237(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 255
  %v12 = select i1 %v11, i128 255, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i8
  %v15 = zext i8 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, -16711681
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vmax_238(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 16
  %v13 = and i32 %c, 65535
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vmax_239(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 24
  %v2 = lshr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 8
  %v5 = ashr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sgt i128 %v3, %v6
  %v8 = select i1 %v7, i128 %v3, i128 %v6
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 16
  %v17 = and i32 %c, 65535
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vshl_240(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  ret i32 %v6
}

define i32 @vshl_241(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -2147483648
  %v7 = select i1 %v6, i128 -2147483648, i128 %v5
  %v8 = icmp sgt i128 %v7, 2147483647
  %v9 = select i1 %v8, i128 2147483647, i128 %v7
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshl_242(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  %v7 = trunc i32 %v6 to i16
  %v8 = zext i16 %v7 to i32
  %v9 = shl i32 %v8, 0
  %v10 = and i32 %c, -65536
  %v11 = or i32 %v10, %v9
  ret i32 %v11
}

define i32 @vshl_243(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -32768
  %v7 = select i1 %v6, i128 -32768, i128 %v5
  %v8 = icmp sgt i128 %v7, 32767
  %v9 = select i1 %v8, i128 32767, i128 %v7
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshl_244(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshl_245(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -2147483648
  %v11 = select i1 %v10, i128 -2147483648, i128 %v9
  %v12 = icmp sgt i128 %v11, 2147483647
  %v13 = select i1 %v12, i128 2147483647, i128 %v11
  %v14 = trunc i128 %v13 to i32
  ret i32 %v14
}

define i32 @vshl_246(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshl_247(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -32768
  %v11 = select i1 %v10, i128 -32768, i128 %v9
  %v12 = icmp sgt i128 %v11, 32767
  %v13 = select i1 %v12, i128 32767, i128 %v11
  %v14 = trunc i128 %v13 to i32
  %v15 = trunc i32 %v14 to i16
  %v16 = zext i16 %v15 to i32
  %v17 = shl i32 %v16, 0
  %v18 = and i32 %c, -65536
  %v19 = or i32 %v18, %v17
  ret i32 %v19
}

define i32 @vshl_248(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vshl_249(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -2147483648
  %v6 = select i1 %v5, i128 -2147483648, i128 %v4
  %v7 = icmp sgt i128 %v6, 2147483647
  %v8 = select i1 %v7, i128 2147483647, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshl_250(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 0
  %v9 = and i32 %c, -65536
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vshl_251(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -32768
  %v6 = select i1 %v5, i128 -32768, i128 %v4
  %v7 = icmp sgt i128 %v6, 32767
  %v8 = select i1 %v7, i128 32767, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshl_252(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshl_253(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vshl_254(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshl_255(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 0
  %v17 = and i32 %c, -65536
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vshl_256(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  ret i32 %v6
}

define i32 @vshl_257(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = icmp slt i128 %v5, 0
  %v7 = select i1 %v6, i128 0, i128 %v5
  %v8 = icmp sgt i128 %v7, 4294967295
  %v9 = select i1 %v8, i128 4294967295, i128 %v7
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshl_258(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  %v7 = trunc i32 %v6 to i16
  %v8 = zext i16 %v7 to i32
  %v9 = shl i32 %v8, 0
  %v10 = and i32 %c, -65536
  %v11 = or i32 %v10, %v9
  ret i32 %v11
}

define i32 @vshl_259(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = icmp slt i128 %v5, 0
  %v7 = select i1 %v6, i128 0, i128 %v5
  %v8 = icmp sgt i128 %v7, 65535
  %v9 = select i1 %v8, i128 65535, i128 %v7
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshl_260(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshl_261(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = icmp slt i128 %v9, 0
  %v11 = select i1 %v10, i128 0, i128 %v9
  %v12 = icmp sgt i128 %v11, 4294967295
  %v13 = select i1 %v12, i128 4294967295, i128 %v11
  %v14 = trunc i128 %v13 to i32
  ret i32 %v14
}

define i32 @vshl_262(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshl_263(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = icmp slt i128 %v9, 0
  %v11 = select i1 %v10, i128 0, i128 %v9
  %v12 = icmp sgt i128 %v11, 65535
  %v13 = select i1 %v12, i128 65535, i128 %v11
  %v14 = trunc i128 %v13 to i32
  %v15 = trunc i32 %v14 to i16
  %v16 = zext i16 %v15 to i32
  %v17 = shl i32 %v16, 0
  %v18 = and i32 %c, -65536
  %v19 = or i32 %v18, %v17
  ret i32 %v19
}

define i32 @vshl_264(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vshl_265(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 4294967295
  %v8 = select i1 %v7, i128 4294967295, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshl_266(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 0
  %v9 = and i32 %c, -65536
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vshl_267(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 65535
  %v8 = select i1 %v7, i128 65535, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshl_268(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshl_269(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vshl_270(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshl_271(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 0
  %v17 = and i32 %c, -65536
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vshl_272(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  ret i32 %v6
}

define i32 @vshl_273(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -2147483648
  %v7 = select i1 %v6, i128 -2147483648, i128 %v5
  %v8 = icmp sgt i128 %v7, 2147483647
  %v9 = select i1 %v8, i128 2147483647, i128 %v7
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshl_274(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  %v7 = trunc i32 %v6 to i16
  %v8 = zext i16 %v7 to i32
  %v9 = shl i32 %v8, 0
  %v10 = and i32 %c, -65536
  %v11 = or i32 %v10, %v9
  ret i32 %v11
}

define i32 @vshl_275(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = shl i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -32768
  %v7 = select i1 %v6, i128 -32768, i128 %v5
  %v8 = icmp sgt i128 %v7, 32767
  %v9 = select i1 %v8, i128 32767, i128 %v7
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshl_276(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshl_277(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -2147483648
  %v11 = select i1 %v10, i128 -2147483648, i128 %v9
  %v12 = icmp sgt i128 %v11, 2147483647
  %v13 = select i1 %v12, i128 2147483647, i128 %v11
  %v14 = trunc i128 %v13 to i32
  ret i32 %v14
}

define i32 @vshl_278(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshl_279(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = shl i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -32768
  %v11 = select i1 %v10, i128 -32768, i128 %v9
  %v12 = icmp sgt i128 %v11, 32767
  %v13 = select i1 %v12, i128 32767, i128 %v11
  %v14 = trunc i128 %v13 to i32
  %v15 = trunc i32 %v14 to i16
  %v16 = zext i16 %v15 to i32
  %v17 = shl i32 %v16, 0
  %v18 = and i32 %c, -65536
  %v19 = or i32 %v18, %v17
  ret i32 %v19
}

define i32 @vshl_280(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vshl_281(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -2147483648
  %v6 = select i1 %v5, i128 -2147483648, i128 %v4
  %v7 = icmp sgt i128 %v6, 2147483647
  %v8 = select i1 %v7, i128 2147483647, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshl_282(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 0
  %v9 = and i32 %c, -65536
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vshl_283(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = shl i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -32768
  %v6 = select i1 %v5, i128 -32768, i128 %v4
  %v7 = icmp sgt i128 %v6, 32767
  %v8 = select i1 %v7, i128 32767, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshl_284(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshl_285(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vshl_286(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshl_287(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = shl i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 0
  %v17 = and i32 %c, -65536
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vshr_288(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = ashr i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  ret i32 %v6
}

define i32 @vshr_289(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = ashr i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -2147483648
  %v7 = select i1 %v6, i128 -2147483648, i128 %v5
  %v8 = icmp sgt i128 %v7, 2147483647
  %v9 = select i1 %v8, i128 2147483647, i128 %v7
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshr_290(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = ashr i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  %v7 = trunc i32 %v6 to i16
  %v8 = zext i16 %v7 to i32
  %v9 = shl i32 %v8, 0
  %v10 = and i32 %c, -65536
  %v11 = or i32 %v10, %v9
  ret i32 %v11
}

define i32 @vshr_291(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = ashr i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -32768
  %v7 = select i1 %v6, i128 -32768, i128 %v5
  %v8 = icmp sgt i128 %v7, 32767
  %v9 = select i1 %v8, i128 32767, i128 %v7
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshr_292(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = ashr i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshr_293(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = ashr i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -2147483648
  %v11 = select i1 %v10, i128 -2147483648, i128 %v9
  %v12 = icmp sgt i128 %v11, 2147483647
  %v13 = select i1 %v12, i128 2147483647, i128 %v11
  %v14 = trunc i128 %v13 to i32
  ret i32 %v14
}

define i32 @vshr_294(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = ashr i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshr_295(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = ashr i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -32768
  %v11 = select i1 %v10, i128 -32768, i128 %v9
  %v12 = icmp sgt i128 %v11, 32767
  %v13 = select i1 %v12, i128 32767, i128 %v11
  %v14 = trunc i128 %v13 to i32
  %v15 = trunc i32 %v14 to i16
  %v16 = zext i16 %v15 to i32
  %v17 = shl i32 %v16, 0
  %v18 = and i32 %c, -65536
  %v19 = or i32 %v18, %v17
  ret i32 %v19
}

define i32 @vshr_296(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = ashr i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vshr_297(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = ashr i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -2147483648
  %v6 = select i1 %v5, i128 -2147483648, i128 %v4
  %v7 = icmp sgt i128 %v6, 2147483647
  %v8 = select i1 %v7, i128 2147483647, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshr_298(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = ashr i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 0
  %v9 = and i32 %c, -65536
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vshr_299(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = ashr i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -32768
  %v6 = select i1 %v5, i128 -32768, i128 %v4
  %v7 = icmp sgt i128 %v6, 32767
  %v8 = select i1 %v7, i128 32767, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshr_300(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = ashr i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshr_301(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = ashr i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vshr_302(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = ashr i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshr_303(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = ashr i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 0
  %v17 = and i32 %c, -65536
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vshr_304(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  ret i32 %v6
}

define i32 @vshr_305(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = icmp slt i128 %v5, 0
  %v7 = select i1 %v6, i128 0, i128 %v5
  %v8 = icmp sgt i128 %v7, 4294967295
  %v9 = select i1 %v8, i128 4294967295, i128 %v7
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshr_306(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  %v7 = trunc i32 %v6 to i16
  %v8 = zext i16 %v7 to i32
  %v9 = shl i32 %v8, 0
  %v10 = and i32 %c, -65536
  %v11 = or i32 %v10, %v9
  ret i32 %v11
}

define i32 @vshr_307(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = icmp slt i128 %v5, 0
  %v7 = select i1 %v6, i128 0, i128 %v5
  %v8 = icmp sgt i128 %v7, 65535
  %v9 = select i1 %v8, i128 65535, i128 %v7
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshr_308(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshr_309(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = icmp slt i128 %v9, 0
  %v11 = select i1 %v10, i128 0, i128 %v9
  %v12 = icmp sgt i128 %v11, 4294967295
  %v13 = select i1 %v12, i128 4294967295, i128 %v11
  %v14 = trunc i128 %v13 to i32
  ret i32 %v14
}

define i32 @vshr_310(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshr_311(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = icmp slt i128 %v9, 0
  %v11 = select i1 %v10, i128 0, i128 %v9
  %v12 = icmp sgt i128 %v11, 65535
  %v13 = select i1 %v12, i128 65535, i128 %v11
  %v14 = trunc i128 %v13 to i32
  %v15 = trunc i32 %v14 to i16
  %v16 = zext i16 %v15 to i32
  %v17 = shl i32 %v16, 0
  %v18 = and i32 %c, -65536
  %v19 = or i32 %v18, %v17
  ret i32 %v19
}

define i32 @vshr_312(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vshr_313(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 4294967295
  %v8 = select i1 %v7, i128 4294967295, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshr_314(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 0
  %v9 = and i32 %c, -65536
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vshr_315(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = icmp slt i128 %v4, 0
  %v6 = select i1 %v5, i128 0, i128 %v4
  %v7 = icmp sgt i128 %v6, 65535
  %v8 = select i1 %v7, i128 65535, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshr_316(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshr_317(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 4294967295
  %v12 = select i1 %v11, i128 4294967295, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vshr_318(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshr_319(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = icmp slt i128 %v8, 0
  %v10 = select i1 %v9, i128 0, i128 %v8
  %v11 = icmp sgt i128 %v10, 65535
  %v12 = select i1 %v11, i128 65535, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 0
  %v17 = and i32 %c, -65536
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vshr_320(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  ret i32 %v6
}

define i32 @vshr_321(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -2147483648
  %v7 = select i1 %v6, i128 -2147483648, i128 %v5
  %v8 = icmp sgt i128 %v7, 2147483647
  %v9 = select i1 %v8, i128 2147483647, i128 %v7
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshr_322(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = trunc i128 %v5 to i32
  %v7 = trunc i32 %v6 to i16
  %v8 = zext i16 %v7 to i32
  %v9 = shl i32 %v8, 0
  %v10 = and i32 %c, -65536
  %v11 = or i32 %v10, %v9
  ret i32 %v11
}

define i32 @vshr_323(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = icmp ugt i128 %v2, 32
  %v4 = select i1 %v3, i128 32, i128 %v2
  %v5 = lshr i128 %v1, %v4
  %v6 = icmp slt i128 %v5, -32768
  %v7 = select i1 %v6, i128 -32768, i128 %v5
  %v8 = icmp sgt i128 %v7, 32767
  %v9 = select i1 %v8, i128 32767, i128 %v7
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshr_324(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  ret i32 %v10
}

define i32 @vshr_325(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -2147483648
  %v11 = select i1 %v10, i128 -2147483648, i128 %v9
  %v12 = icmp sgt i128 %v11, 2147483647
  %v13 = select i1 %v12, i128 2147483647, i128 %v11
  %v14 = trunc i128 %v13 to i32
  ret i32 %v14
}

define i32 @vshr_326(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = trunc i128 %v9 to i32
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = shl i32 %v12, 0
  %v14 = and i32 %c, -65536
  %v15 = or i32 %v14, %v13
  ret i32 %v15
}

define i32 @vshr_327(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ugt i128 %v6, 32
  %v8 = select i1 %v7, i128 32, i128 %v6
  %v9 = lshr i128 %v3, %v8
  %v10 = icmp slt i128 %v9, -32768
  %v11 = select i1 %v10, i128 -32768, i128 %v9
  %v12 = icmp sgt i128 %v11, 32767
  %v13 = select i1 %v12, i128 32767, i128 %v11
  %v14 = trunc i128 %v13 to i32
  %v15 = trunc i32 %v14 to i16
  %v16 = zext i16 %v15 to i32
  %v17 = shl i32 %v16, 0
  %v18 = and i32 %c, -65536
  %v19 = or i32 %v18, %v17
  ret i32 %v19
}

define i32 @vshr_328(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  ret i32 %v5
}

define i32 @vshr_329(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -2147483648
  %v6 = select i1 %v5, i128 -2147483648, i128 %v4
  %v7 = icmp sgt i128 %v6, 2147483647
  %v8 = select i1 %v7, i128 2147483647, i128 %v6
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshr_330(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = trunc i128 %v4 to i32
  %v6 = trunc i32 %v5 to i16
  %v7 = zext i16 %v6 to i32
  %v8 = shl i32 %v7, 0
  %v9 = and i32 %c, -65536
  %v10 = or i32 %v9, %v8
  ret i32 %v10
}

define i32 @vshr_331(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i128
  %v2 = zext i32 %b to i128
  %v3 = urem i128 %v2, 32
  %v4 = lshr i128 %v1, %v3
  %v5 = icmp slt i128 %v4, -32768
  %v6 = select i1 %v5, i128 -32768, i128 %v4
  %v7 = icmp sgt i128 %v6, 32767
  %v8 = select i1 %v7, i128 32767, i128 %v6
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshr_332(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vshr_333(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -2147483648
  %v10 = select i1 %v9, i128 -2147483648, i128 %v8
  %v11 = icmp sgt i128 %v10, 2147483647
  %v12 = select i1 %v11, i128 2147483647, i128 %v10
  %v13 = trunc i128 %v12 to i32
  ret i32 %v13
}

define i32 @vshr_334(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i16
  %v11 = zext i16 %v10 to i32
  %v12 = shl i32 %v11, 0
  %v13 = and i32 %c, -65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vshr_335(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = lshr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = urem i128 %v6, 32
  %v8 = lshr i128 %v3, %v7
  %v9 = icmp slt i128 %v8, -32768
  %v10 = select i1 %v9, i128 -32768, i128 %v8
  %v11 = icmp sgt i128 %v10, 32767
  %v12 = select i1 %v11, i128 32767, i128 %v10
  %v13 = trunc i128 %v12 to i32
  %v14 = trunc i32 %v13 to i16
  %v15 = zext i16 %v14 to i32
  %v16 = shl i32 %v15, 0
  %v17 = and i32 %c, -65536
  %v18 = or i32 %v17, %v16
  ret i32 %v18
}

define i32 @vset_336(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vset_337(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 8
  %v13 = and i32 %c, -65281
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vset_338(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = lshr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vset_339(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = lshr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp slt i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 8
  %v13 = and i32 %c, -65281
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vset_340(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sge i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vset_341(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sge i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 8
  %v13 = and i32 %c, -65281
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vset_342(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = lshr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sge i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vset_343(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = lshr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp sge i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 8
  %v13 = and i32 %c, -65281
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vset_344(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ne i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vset_345(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 0
  %v2 = ashr i32 %v1, 24
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 24
  %v5 = lshr i32 %v4, 24
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ne i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 8
  %v13 = and i32 %c, -65281
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @vset_346(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = lshr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ne i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  ret i32 %v9
}

define i32 @vset_347(i32 %a, i32 %b, i32 %c) {
  %v1 = shl i32 %a, 16
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i128
  %v4 = shl i32 %b, 0
  %v5 = lshr i32 %v4, 16
  %v6 = sext i32 %v5 to i128
  %v7 = icmp ne i128 %v3, %v6
  %v8 = zext i1 %v7 to i128
  %v9 = trunc i128 %v8 to i32
  %v10 = trunc i32 %v9 to i8
  %v11 = zext i8 %v10 to i32
  %v12 = shl i32 %v11, 8
  %v13 = and i32 %c, -65281
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}
