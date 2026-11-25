; cvt.pack.sat (PTX ISA): a and b saturated to the destination type and
; packed (a above b); the 3-operand forms put c above, shifted by twice
; the field width.

define i32 @pack_u16(i32 %a, i32 %b) {
  %v1 = icmp slt i32 %a, 0
  %v2 = select i1 %v1, i32 0, i32 %a
  %v3 = icmp sgt i32 %v2, 65535
  %v4 = select i1 %v3, i32 65535, i32 %v2
  %v5 = trunc i32 %v4 to i16
  %v6 = zext i16 %v5 to i32
  %v7 = icmp slt i32 %b, 0
  %v8 = select i1 %v7, i32 0, i32 %b
  %v9 = icmp sgt i32 %v8, 65535
  %v10 = select i1 %v9, i32 65535, i32 %v8
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = mul i32 %v6, 65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @pack_s16(i32 %a, i32 %b) {
  %v1 = icmp slt i32 %a, -32768
  %v2 = select i1 %v1, i32 -32768, i32 %a
  %v3 = icmp sgt i32 %v2, 32767
  %v4 = select i1 %v3, i32 32767, i32 %v2
  %v5 = trunc i32 %v4 to i16
  %v6 = zext i16 %v5 to i32
  %v7 = icmp slt i32 %b, -32768
  %v8 = select i1 %v7, i32 -32768, i32 %b
  %v9 = icmp sgt i32 %v8, 32767
  %v10 = select i1 %v9, i32 32767, i32 %v8
  %v11 = trunc i32 %v10 to i16
  %v12 = zext i16 %v11 to i32
  %v13 = mul i32 %v6, 65536
  %v14 = or i32 %v13, %v12
  ret i32 %v14
}

define i32 @pack_u8(i32 %a, i32 %b, i32 %c) {
  %v1 = icmp slt i32 %a, 0
  %v2 = select i1 %v1, i32 0, i32 %a
  %v3 = icmp sgt i32 %v2, 255
  %v4 = select i1 %v3, i32 255, i32 %v2
  %v5 = trunc i32 %v4 to i8
  %v6 = zext i8 %v5 to i32
  %v7 = icmp slt i32 %b, 0
  %v8 = select i1 %v7, i32 0, i32 %b
  %v9 = icmp sgt i32 %v8, 255
  %v10 = select i1 %v9, i32 255, i32 %v8
  %v11 = trunc i32 %v10 to i8
  %v12 = zext i8 %v11 to i32
  %v13 = mul i32 %v6, 256
  %v14 = or i32 %v13, %v12
  %v15 = mul i32 %c, 65536
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @pack_s8(i32 %a, i32 %b, i32 %c) {
  %v1 = icmp slt i32 %a, -128
  %v2 = select i1 %v1, i32 -128, i32 %a
  %v3 = icmp sgt i32 %v2, 127
  %v4 = select i1 %v3, i32 127, i32 %v2
  %v5 = trunc i32 %v4 to i8
  %v6 = zext i8 %v5 to i32
  %v7 = icmp slt i32 %b, -128
  %v8 = select i1 %v7, i32 -128, i32 %b
  %v9 = icmp sgt i32 %v8, 127
  %v10 = select i1 %v9, i32 127, i32 %v8
  %v11 = trunc i32 %v10 to i8
  %v12 = zext i8 %v11 to i32
  %v13 = mul i32 %v6, 256
  %v14 = or i32 %v13, %v12
  %v15 = mul i32 %c, 65536
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @pack_u4(i32 %a, i32 %b, i32 %c) {
  %v1 = icmp slt i32 %a, 0
  %v2 = select i1 %v1, i32 0, i32 %a
  %v3 = icmp sgt i32 %v2, 15
  %v4 = select i1 %v3, i32 15, i32 %v2
  %v5 = trunc i32 %v4 to i4
  %v6 = zext i4 %v5 to i32
  %v7 = icmp slt i32 %b, 0
  %v8 = select i1 %v7, i32 0, i32 %b
  %v9 = icmp sgt i32 %v8, 15
  %v10 = select i1 %v9, i32 15, i32 %v8
  %v11 = trunc i32 %v10 to i4
  %v12 = zext i4 %v11 to i32
  %v13 = mul i32 %v6, 16
  %v14 = or i32 %v13, %v12
  %v15 = mul i32 %c, 256
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @pack_s4(i32 %a, i32 %b, i32 %c) {
  %v1 = icmp slt i32 %a, -8
  %v2 = select i1 %v1, i32 -8, i32 %a
  %v3 = icmp sgt i32 %v2, 7
  %v4 = select i1 %v3, i32 7, i32 %v2
  %v5 = trunc i32 %v4 to i4
  %v6 = zext i4 %v5 to i32
  %v7 = icmp slt i32 %b, -8
  %v8 = select i1 %v7, i32 -8, i32 %b
  %v9 = icmp sgt i32 %v8, 7
  %v10 = select i1 %v9, i32 7, i32 %v8
  %v11 = trunc i32 %v10 to i4
  %v12 = zext i4 %v11 to i32
  %v13 = mul i32 %v6, 16
  %v14 = or i32 %v13, %v12
  %v15 = mul i32 %c, 256
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @pack_u2(i32 %a, i32 %b, i32 %c) {
  %v1 = icmp slt i32 %a, 0
  %v2 = select i1 %v1, i32 0, i32 %a
  %v3 = icmp sgt i32 %v2, 3
  %v4 = select i1 %v3, i32 3, i32 %v2
  %v5 = trunc i32 %v4 to i2
  %v6 = zext i2 %v5 to i32
  %v7 = icmp slt i32 %b, 0
  %v8 = select i1 %v7, i32 0, i32 %b
  %v9 = icmp sgt i32 %v8, 3
  %v10 = select i1 %v9, i32 3, i32 %v8
  %v11 = trunc i32 %v10 to i2
  %v12 = zext i2 %v11 to i32
  %v13 = mul i32 %v6, 4
  %v14 = or i32 %v13, %v12
  %v15 = mul i32 %c, 16
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}

define i32 @pack_s2(i32 %a, i32 %b, i32 %c) {
  %v1 = icmp slt i32 %a, -2
  %v2 = select i1 %v1, i32 -2, i32 %a
  %v3 = icmp sgt i32 %v2, 1
  %v4 = select i1 %v3, i32 1, i32 %v2
  %v5 = trunc i32 %v4 to i2
  %v6 = zext i2 %v5 to i32
  %v7 = icmp slt i32 %b, -2
  %v8 = select i1 %v7, i32 -2, i32 %b
  %v9 = icmp sgt i32 %v8, 1
  %v10 = select i1 %v9, i32 1, i32 %v8
  %v11 = trunc i32 %v10 to i2
  %v12 = zext i2 %v11 to i32
  %v13 = mul i32 %v6, 4
  %v14 = or i32 %v13, %v12
  %v15 = mul i32 %c, 16
  %v16 = or i32 %v15, %v14
  ret i32 %v16
}
