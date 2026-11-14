; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-shlhi-shift=0 < %s | FileCheck %s --check-prefix=OFF

; 64-bit shifts by a variable amount (0 to 63): shl's bits of the low word
; that move into the high word are shlhi 0, lo, s (for extr 0, lo, 64 - s
; and its subtraction); ashr's low word is asrh hi, s or'ed with
; extr lo, 0, s (for a select of extr and asr by s - 32) (L15-E321).

; CHECK-LABEL: shl64:
; CHECK: shlhi
; CHECK-NOT: isub
; OFF-LABEL: shl64:
; OFF: isub
define agx_kernel void @shl64(ptr addrspace(1) %p, i64 %x, i64 %s) {
  %r = shl i64 %x, %s
  store i64 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: ashr64:
; CHECK-DAG: asrh
; CHECK-DAG: extr
; CHECK-NOT: icmpsel
; OFF-LABEL: ashr64:
; OFF: icmpsel
define agx_kernel void @ashr64(ptr addrspace(1) %p, i64 %x, i64 %s) {
  %r = ashr i64 %x, %s
  store i64 %r, ptr addrspace(1) %p
  ret void
}
