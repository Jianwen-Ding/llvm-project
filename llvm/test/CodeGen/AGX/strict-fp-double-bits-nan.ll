; A canonical NaN made on the bits of a double stays an integer test: AGXStrictFP
; makes a float comparison of one on a float, but doubles are software by then,
; and a comparison of them could not be selected ("no libcall available to soften
; floating-point setuo compare with type f64"). agx-lower-air recovers such doubles
; from AIR, where --strict-fp's NaNs are made on the bits.
; RUN: llc -mtriple=agx -mcpu=agx-g14g -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g14g -mattr=+ieee-fp32,+canonical-nan -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g14g -print-after=agx-strict-fp -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

; CHECK-NOT: fcmp uno double
; CHECK: icmp ugt i64 %a, 9218868437227405312

define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %f = load float, ptr addrspace(1) %p
  %d = fpext float %f to double
  %i = bitcast double %d to i64
  %a = and i64 %i, 9223372036854775807
  %c = icmp ugt i64 %a, 9218868437227405312
  %s = select i1 %c, i64 9223372036854775807, i64 %i
  store i64 %s, ptr addrspace(1) %q
  ret void
}
