; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-signed-bytes-alu16=0 < %s | FileCheck %s --check-prefix=OFF

; A loaded byte sign-extended in a 16-bit register is (x ^ 128) - 128 on the
; ALU (the zero-extending load gives the zero high byte), not the bit-field
; unit's shift left and arithmetic shift right; a byte that is converted or
; extended to 32 bits keeps the forms that fold it (L15-E311).

; CHECK-LABEL: smax8:
; CHECK-NOT: bfi
; CHECK-DAG: xor {{r[0-9]+}}l, {{r[0-9]+}}l, 128
; CHECK-DAG: isub {{r[0-9]+}}l, {{r[0-9]+}}l, 128
; CHECK-NOT: asr
; CHECK: icmpsel sgt
; OFF-LABEL: smax8:
; OFF: bfi
; OFF: asr
define agx_kernel void @smax8(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i8, ptr addrspace(1) %p, i32 %t
  %b = getelementptr i8, ptr addrspace(1) %q, i32 %t
  %x = load i8, ptr addrspace(1) %a
  %y = load i8, ptr addrspace(1) %b
  %m = call i8 @llvm.smax.i8(i8 %x, i8 %y)
  store i8 %m, ptr addrspace(1) %a
  ret void
}

; CHECK-LABEL: tofloat:
; CHECK: convert s8_to_f
; CHECK-NOT: isub
define agx_kernel void @tofloat(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i8, ptr addrspace(1) %p, i32 %t
  %x = load i8, ptr addrspace(1) %a
  %f = sitofp i8 %x to float
  %b = getelementptr float, ptr addrspace(1) %o, i32 %t
  store float %f, ptr addrspace(1) %b
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i8 @llvm.smax.i8(i8, i8)
