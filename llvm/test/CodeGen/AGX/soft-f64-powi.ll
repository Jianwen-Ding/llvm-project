; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; powi of a double (SelectionDAG cannot soften it: "do not know how to
; soften fpowi") is compiler-rt's __powidf2 in double operations, which
; become integer code: r *= x where the bit of n is set, x *= x while bits
; remain, 1 / r for a negative n.

; CHECK-LABEL: @powi(
; CHECK-NOT: call double @llvm.powi
; CHECK: powi.loop:
; CHECK: br i1 {{.*}}, label %powi.mul, label %powi.step
; CHECK: powi.next:
; CHECK: powi.mul:
; CHECK: powi.step:
; CHECK: sdiv i32 {{.*}}, 2
; CHECK: br i1 {{.*}}, label %powi.done, label %powi.next
; CHECK: icmp slt i32 %n, 0
; CHECK-NOT: {{fmul|fdiv}} double
; ASM-LABEL: powi:
; ASM-NOT: error
define agx_kernel void @powi(ptr addrspace(1) %p, double %x, i32 %n) {
  %r = call double @llvm.powi.f64.i32(double %x, i32 %n)
  store double %r, ptr addrspace(1) %p
  ret void
}

; is.fpclass of a double: integer tests of the bits.
; CHECK-LABEL: @cls(
; CHECK-NOT: is.fpclass
; ASM-LABEL: cls:
define agx_kernel void @cls(ptr addrspace(1) %p, double %x) {
  %c = call i1 @llvm.is.fpclass.f64(double %x, i32 516)
  %z = zext i1 %c to i32
  store i32 %z, ptr addrspace(1) %p
  ret void
}
