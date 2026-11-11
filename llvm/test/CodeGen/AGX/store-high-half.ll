; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; A 16-bit store of x >> 16 stores x's high half register, without the
; shift: a float rounded to bfloat, the high half of a packed pair. Checked
; with agx-tv and, for the bfloat rounding, on the simulator's functional
; model.

; CHECK-LABEL: h:
; CHECK: iadd [[R:r[0-9]+]],
; CHECK-NEXT: device_store 0, i16, x, [[R]]h,
define agx_kernel void @h(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %s = add i32 %a, %b
  %h = lshr i32 %s, 16
  %t = trunc i32 %h to i16
  store i16 %t, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: bf:
; CHECK-NOT: bfeil {{r[0-9]+}}, 0, {{r[0-9]+}}, 16
; CHECK: device_store 0, i16, x, {{r[0-9]+}}h,
define agx_kernel void @bf(ptr addrspace(1) %p, float %a) {
  %b = fptrunc float %a to bfloat
  store bfloat %b, ptr addrspace(1) %p
  ret void
}

@buf = internal addrspace(3) global [64 x i16] poison, align 2
; CHECK-LABEL: tg:
; CHECK: threadgroup_store i16, x, {{r[0-9]+}}h,
define agx_kernel void @tg(i32 %a, i32 %b, i16 %i) {
  %s = add i32 %a, %b
  %h = lshr i32 %s, 16
  %t = trunc i32 %h to i16
  %p = getelementptr i16, ptr addrspace(3) @buf, i16 %i
  store i16 %t, ptr addrspace(3) %p
  ret void
}
