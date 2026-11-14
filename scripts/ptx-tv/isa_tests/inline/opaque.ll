; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx80 | FileCheck %s
define i64 @pol_frac(float %f) {
  %p = call i64 asm "createpolicy.fractional.L2::evict_last.b64 $0, $1;", "=l,f"(float %f)
  ret i64 %p
}
define i64 @pol_range(i64 %a, i32 %x, i32 %y) {
  %p = call i64 asm "createpolicy.range.L2::evict_last.L2::evict_unchanged.b64 $0, [$1], $2, $3;", "=l,l,r,r"(i64 %a, i32 %x, i32 %y)
  ret i64 %p
}
define i32 @isty(i64 %h) {
  %p = call i32 asm "{\0A\09.reg .pred p;\0A\09istypeof.texref p, $1;\0A\09selp.u32 $0, 1, 0, p;\0A\09}", "=r,l"(i64 %h)
  ret i32 %p
}
define {float, float, float, float} @gather(i64 %t, float %x, float %y) {
  %r = call {float, float, float, float} asm "tld4.r.2d.v4.f32.f32 {$0, $1, $2, $3}, [$4, {$5, $6}];", "=f,=f,=f,=f,l,f,f"(i64 %t, float %x, float %y)
  ret {float, float, float, float} %r
}
define void @sred(i64 %s, i32 %x, i32 %v) {
  call void asm sideeffect "sured.b.add.1d.u32.trap [$0, {$1}], $2;", "l,r,r"(i64 %s, i32 %x, i32 %v)
  ret void
}
