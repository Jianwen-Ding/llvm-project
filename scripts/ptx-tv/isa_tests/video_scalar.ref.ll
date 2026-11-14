; Reference semantics of the scalar video instructions (default selectors),
; from the PTX ISA pseudo-code: exact operation on extended operands,
; optional saturation to the dtype range, then the optional secondary
; operation with c (min/max in the dtype's signedness; vset unsigned).

define i32 @vadd_s32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vadd_s32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vadd_s32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp slt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_s32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp sgt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_s32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_s32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vadd_s32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp slt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vadd_s32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp sgt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vadd_u32_u32_u32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vadd_u32_u32_u32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vadd_u32_u32_u32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ult i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_u32_u32_u32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ugt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_u32_u32_u32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_u32_u32_u32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vadd_u32_u32_u32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ult i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vadd_u32_u32_u32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ugt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vadd_s32_u32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vadd_s32_u32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vadd_s32_u32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp slt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_s32_u32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp sgt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_s32_u32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_s32_u32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vadd_s32_u32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp slt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vadd_s32_u32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp sgt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vadd_u32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vadd_u32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vadd_u32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ult i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_u32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ugt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vadd_u32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vadd_u32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vadd_u32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ult i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vadd_u32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = add i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ugt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_s32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vsub_s32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vsub_s32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp slt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_s32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp sgt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_s32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_s32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vsub_s32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp slt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_s32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp sgt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_u32_u32_u32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vsub_u32_u32_u32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vsub_u32_u32_u32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ult i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_u32_u32_u32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ugt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_u32_u32_u32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_u32_u32_u32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vsub_u32_u32_u32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ult i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_u32_u32_u32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ugt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_s32_u32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vsub_s32_u32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vsub_s32_u32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp slt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_s32_u32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp sgt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_s32_u32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_s32_u32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vsub_s32_u32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp slt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_s32_u32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, -2147483648
  %v5 = select i1 %v4, i64 -2147483648, i64 %v3
  %v6 = icmp sgt i64 %v5, 2147483647
  %v7 = select i1 %v6, i64 2147483647, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp sgt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_u32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  ret i32 %v4
}

define i32 @vsub_u32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = add i32 %v4, %c
  ret i32 %v5
}

define i32 @vsub_u32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ult i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_u32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = trunc i64 %v3 to i32
  %v5 = icmp ugt i32 %v4, %c
  %v6 = select i1 %v5, i32 %v4, i32 %c
  ret i32 %v6
}

define i32 @vsub_u32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  ret i32 %v8
}

define i32 @vsub_u32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = add i32 %v8, %c
  ret i32 %v9
}

define i32 @vsub_u32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ult i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vsub_u32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = select i1 %v4, i64 0, i64 %v3
  %v6 = icmp sgt i64 %v5, 4294967295
  %v7 = select i1 %v6, i64 4294967295, i64 %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = icmp ugt i32 %v8, %c
  %v10 = select i1 %v9, i32 %v8, i32 %c
  ret i32 %v10
}

define i32 @vabsdiff_s32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  ret i32 %v7
}

define i32 @vabsdiff_s32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = add i32 %v7, %c
  ret i32 %v8
}

define i32 @vabsdiff_s32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp slt i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_s32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp sgt i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_s32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_s32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = add i32 %v11, %c
  ret i32 %v12
}

define i32 @vabsdiff_s32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp slt i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vabsdiff_s32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp sgt i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vabsdiff_u32_u32_u32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  ret i32 %v7
}

define i32 @vabsdiff_u32_u32_u32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = add i32 %v7, %c
  ret i32 %v8
}

define i32 @vabsdiff_u32_u32_u32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp ult i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_u32_u32_u32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp ugt i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_u32_u32_u32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_u32_u32_u32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = add i32 %v11, %c
  ret i32 %v12
}

define i32 @vabsdiff_u32_u32_u32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp ult i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vabsdiff_u32_u32_u32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp ugt i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vabsdiff_s32_u32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  ret i32 %v7
}

define i32 @vabsdiff_s32_u32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = add i32 %v7, %c
  ret i32 %v8
}

define i32 @vabsdiff_s32_u32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp slt i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_s32_u32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp sgt i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_s32_u32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_s32_u32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = add i32 %v11, %c
  ret i32 %v12
}

define i32 @vabsdiff_s32_u32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp slt i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vabsdiff_s32_u32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, -2147483648
  %v8 = select i1 %v7, i64 -2147483648, i64 %v6
  %v9 = icmp sgt i64 %v8, 2147483647
  %v10 = select i1 %v9, i64 2147483647, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp sgt i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vabsdiff_u32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  ret i32 %v7
}

define i32 @vabsdiff_u32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = add i32 %v7, %c
  ret i32 %v8
}

define i32 @vabsdiff_u32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp ult i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_u32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp ugt i32 %v7, %c
  %v9 = select i1 %v8, i32 %v7, i32 %c
  ret i32 %v9
}

define i32 @vabsdiff_u32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  ret i32 %v11
}

define i32 @vabsdiff_u32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = add i32 %v11, %c
  ret i32 %v12
}

define i32 @vabsdiff_u32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp ult i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vabsdiff_u32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = sub i64 %v1, %v2
  %v4 = icmp slt i64 %v3, 0
  %v5 = sub i64 0, %v3
  %v6 = select i1 %v4, i64 %v5, i64 %v3
  %v7 = icmp slt i64 %v6, 0
  %v8 = select i1 %v7, i64 0, i64 %v6
  %v9 = icmp sgt i64 %v8, 4294967295
  %v10 = select i1 %v9, i64 4294967295, i64 %v8
  %v11 = trunc i64 %v10 to i32
  %v12 = icmp ugt i32 %v11, %c
  %v13 = select i1 %v12, i32 %v11, i32 %c
  ret i32 %v13
}

define i32 @vmin_s32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmin_s32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmin_s32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp slt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_s32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp sgt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_s32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_s32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmin_s32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp slt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmin_s32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp sgt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmin_u32_u32_u32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmin_u32_u32_u32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmin_u32_u32_u32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_u32_u32_u32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_u32_u32_u32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_u32_u32_u32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmin_u32_u32_u32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ult i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmin_u32_u32_u32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ugt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmin_s32_u32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmin_s32_u32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmin_s32_u32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp slt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_s32_u32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp sgt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_s32_u32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_s32_u32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmin_s32_u32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp slt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmin_s32_u32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp sgt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmin_u32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmin_u32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmin_u32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_u32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmin_u32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmin_u32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmin_u32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ult i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmin_u32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ugt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_s32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmax_s32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmax_s32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp slt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_s32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp sgt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_s32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_s32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmax_s32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp slt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_s32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp sgt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_u32_u32_u32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmax_u32_u32_u32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmax_u32_u32_u32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_u32_u32_u32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_u32_u32_u32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_u32_u32_u32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmax_u32_u32_u32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ult i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_u32_u32_u32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ugt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_s32_u32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmax_s32_u32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmax_s32_u32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp slt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_s32_u32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp sgt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_s32_u32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_s32_u32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmax_s32_u32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp slt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_s32_u32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, -2147483648
  %v6 = select i1 %v5, i64 -2147483648, i64 %v4
  %v7 = icmp sgt i64 %v6, 2147483647
  %v8 = select i1 %v7, i64 2147483647, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp sgt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_u32_s32_s32(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vmax_u32_s32_s32_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vmax_u32_s32_s32_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_u32_s32_s32_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vmax_u32_s32_s32_sat(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  ret i32 %v9
}

define i32 @vmax_u32_s32_s32_sat_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = add i32 %v9, %c
  ret i32 %v10
}

define i32 @vmax_u32_s32_s32_sat_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ult i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vmax_u32_s32_s32_sat_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = select i1 %v3, i64 %v1, i64 %v2
  %v5 = icmp slt i64 %v4, 0
  %v6 = select i1 %v5, i64 0, i64 %v4
  %v7 = icmp sgt i64 %v6, 4294967295
  %v8 = select i1 %v7, i64 4294967295, i64 %v6
  %v9 = trunc i64 %v8 to i32
  %v10 = icmp ugt i32 %v9, %c
  %v11 = select i1 %v10, i32 %v9, i32 %c
  ret i32 %v11
}

define i32 @vset_s32_s32_eq(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_s32_eq_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_s32_eq_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_eq_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_eq(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_u32_u32_eq_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_u32_u32_eq_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_eq_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_eq(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_u32_eq_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_u32_eq_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_eq_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp eq i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_ne(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_s32_ne_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_s32_ne_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_ne_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_ne(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_u32_u32_ne_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_u32_u32_ne_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_ne_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_ne(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_u32_ne_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_u32_ne_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_ne_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp ne i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_lt(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_s32_lt_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_s32_lt_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_lt_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_lt(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_u32_u32_lt_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_u32_u32_lt_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_lt_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_lt(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_u32_lt_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_u32_lt_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_lt_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp slt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_le(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_s32_le_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_s32_le_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_le_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_le(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_u32_u32_le_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_u32_u32_le_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_le_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_le(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_u32_le_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_u32_le_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_le_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sle i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_gt(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_s32_gt_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_s32_gt_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_gt_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_gt(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_u32_u32_gt_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_u32_u32_gt_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_gt_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_gt(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_u32_gt_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_u32_gt_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_gt_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sgt i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_ge(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_s32_ge_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_s32_ge_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_s32_ge_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = sext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_ge(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_u32_u32_ge_add(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_u32_u32_ge_min(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_u32_u32_ge_max(i32 %a, i32 %b, i32 %c) {
  %v1 = zext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_ge(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  ret i32 %v5
}

define i32 @vset_s32_u32_ge_add(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = add i32 %v5, %c
  ret i32 %v6
}

define i32 @vset_s32_u32_ge_min(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ult i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}

define i32 @vset_s32_u32_ge_max(i32 %a, i32 %b, i32 %c) {
  %v1 = sext i32 %a to i64
  %v2 = zext i32 %b to i64
  %v3 = icmp sge i64 %v1, %v2
  %v4 = zext i1 %v3 to i64
  %v5 = trunc i64 %v4 to i32
  %v6 = icmp ugt i32 %v5, %c
  %v7 = select i1 %v6, i32 %v5, i32 %c
  ret i32 %v7
}
