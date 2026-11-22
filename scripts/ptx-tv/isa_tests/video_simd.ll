define i32 @vadd2_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd2_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd2.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub2_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub2.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg2_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg2.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff2_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff2.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin2_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin2.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax2_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax2.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_eqplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.eq $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_eqadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.eq.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_eqplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.eq $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_eqadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.eq.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_eqplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.eq $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_eqadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.eq.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_neplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.ne $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_neadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.ne.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_neplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.ne $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_neadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.ne.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_neplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.ne $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_neadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.ne.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_ltplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.lt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_ltadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.lt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_ltplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.lt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_ltadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.lt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_ltplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.lt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_ltadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.lt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_leplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.le $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_leadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.le.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_leplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.le $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_leadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.le.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_leplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.le $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_leadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.le.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_gtplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.gt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_gtadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.gt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_gtplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.gt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_gtadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.gt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_gtplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.gt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_gtadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.gt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_geplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.ge $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_s32s32_geadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.s32.s32.ge.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_geplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.ge $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32u32_geadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.u32.ge.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_geplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.ge $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset2_u32s32_geadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset2.u32.s32.ge.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vadd4_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vadd4.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vsub4_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vsub4.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vavrg4_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vavrg4.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vabsdiff4_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vabsdiff4.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmin4_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmin4.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_s32s32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_s32s32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_s32s32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.s32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_u32u32u32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.u32.u32.u32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_u32u32u32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.u32.u32.u32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_u32u32u32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_s32u32s32_plain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.u32.s32 $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_s32u32s32_sat(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.u32.s32.sat $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vmax4_s32u32s32_add(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vmax4.s32.u32.s32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_eqplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.eq $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_eqadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.eq.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_eqplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.eq $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_eqadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.eq.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_eqplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.eq $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_eqadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.eq.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_neplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.ne $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_neadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.ne.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_neplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.ne $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_neadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.ne.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_neplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.ne $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_neadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.ne.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_ltplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.lt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_ltadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.lt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_ltplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.lt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_ltadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.lt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_ltplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.lt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_ltadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.lt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_leplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.le $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_leadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.le.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_leplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.le $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_leadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.le.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_leplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.le $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_leadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.le.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_gtplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.gt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_gtadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.gt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_gtplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.gt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_gtadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.gt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_gtplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.gt $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_gtadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.gt.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_geplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.ge $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_s32s32_geadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.s32.s32.ge.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_geplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.ge $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32u32_geadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.u32.ge.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_geplain(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.ge $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}

define i32 @vset4_u32s32_geadd(i32 %a, i32 %b, i32 %c) {
  %d = call i32 asm "vset4.u32.s32.ge.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %a, i32 %b, i32 %c)
  ret i32 %d
}
