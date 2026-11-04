; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -stop-after=agx-strict-fp < %s | FileCheck %s --check-prefix=CANON

; A canonical NaN made on the bits (as a front end does where float
; comparisons may be taken for fast math) is one fcmpsel on AGX, whose float
; comparisons see NaNs; nothing where the backend canonicalizes the value
; itself.

; CHECK-LABEL: h:
; CHECK: fmul16
; CHECK-NOT: icmpsel
; CHECK: fcmpsel eq
; CANON-LABEL: @h(
; CANON: fcmp uno half
; CANON-NOT: fcmp
; CANON-NOT: icmp
; CANON: ret
define void @h(ptr addrspace(1) %p) {
  %a = load half, ptr addrspace(1) %p
  %q = getelementptr half, ptr addrspace(1) %p, i64 1
  %b = load half, ptr addrspace(1) %q
  %m = fmul half %a, %b
  %bits = bitcast half %m to i16
  %abs = and i16 %bits, 32767
  %nan = icmp ugt i16 %abs, 31744
  %c = select i1 %nan, i16 32767, i16 %bits
  %r = bitcast i16 %c to half
  store half %r, ptr addrspace(1) %p
  ret void
}

; InstCombine's form of it, on a vector.
; CHECK-LABEL: v:
; CHECK-NOT: icmpsel
; CHECK: fcmpsel eq
; CHECK: fcmpsel eq
; CANON-LABEL: @v(
; CANON-NOT: icmp
; CANON: ret
define void @v(ptr addrspace(1) %p) {
  %a = load <2 x half>, ptr addrspace(1) %p
  %q = getelementptr <2 x half>, ptr addrspace(1) %p, i64 1
  %b = load <2 x half>, ptr addrspace(1) %q
  %m = fmul <2 x half> %a, %b
  %f = call <2 x half> @llvm.fabs.v2f16(<2 x half> %m)
  %bits = bitcast <2 x half> %f to <2 x i16>
  %nan = icmp ugt <2 x i16> %bits, splat (i16 31744)
  %r = select <2 x i1> %nan, <2 x half> splat (half 0xH7FFF), <2 x half> %m
  store <2 x half> %r, ptr addrspace(1) %p
  ret void
}

; Not a canonical NaN (another constant): kept.
; CHECK-LABEL: o:
; CHECK: icmpsel
define void @o(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %bits = bitcast float %x to i32
  %abs = and i32 %bits, 2147483647
  %nan = icmp ugt i32 %abs, 2139095040
  %c = select i1 %nan, i32 2143289344, i32 %bits
  store i32 %c, ptr addrspace(1) %p
  ret void
}
