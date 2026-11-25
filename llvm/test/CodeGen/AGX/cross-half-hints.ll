; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-cross-half-hints=0 < %s | FileCheck %s --check-prefix=OFF

; An instruction that writes one half of a register and reads its other
; half among two register sources waits for its destination's earlier
; write: the product of two halves loaded into r1l and r1h goes to a half
; of another register (L15-E632).
; CHECK-LABEL: mul:
; CHECK: fmul16 [[D:r[0-9]+]]l, [[S:r[0-9]+]]l, [[S]]h
; CHECK-NOT: fmul16 [[S]]
; OFF-LABEL: mul:
; OFF: fmul16 [[S:r[0-9]+]]l, [[S]]l, [[S]]h
define agx_kernel void @mul(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load half, ptr addrspace(1) %p, align 2
  %b = load half, ptr addrspace(1) %q, align 2
  %c = fmul half %a, %b
  store half %c, ptr addrspace(1) %q, align 2
  ret void
}
