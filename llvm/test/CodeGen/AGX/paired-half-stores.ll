; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-paired-half-stores=false < %s | FileCheck %s --check-prefix=OFF

; A 4-byte aligned store of 16-bit pairs is one 32-bit store per pair whose
; halves are the elements, computed into a register's halves.

; CHECK-LABEL: f16x2:
; CHECK: fmul16 [[R:r[0-9]+]]l,
; CHECK-NEXT: fmul16 [[R]]h,
; CHECK-NEXT: device_store 0, i32, x, [[R]], u2_u3, 0, signed
; OFF-LABEL: f16x2:
; OFF: device_store 0, i16, x
; OFF: device_store 0, i16, x
define void @f16x2(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %a = load <2 x half>, ptr addrspace(1) %p, align 4
  %q = getelementptr i8, ptr addrspace(1) %p, i64 4
  %b = load <2 x half>, ptr addrspace(1) %q, align 4
  %m = fmul <2 x half> %a, %b
  store <2 x half> %m, ptr addrspace(1) %o, align 4
  ret void
}

; Four elements: one store of two words.
; CHECK-LABEL: i16x4:
; CHECK: device_store 0, i32, xy,
; CHECK-NOT: device_store
define void @i16x4(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %a = load <4 x i16>, ptr addrspace(1) %p, align 8
  %m = add <4 x i16> %a, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %m, ptr addrspace(1) %o, align 8
  ret void
}

; 2-byte aligned: a 16-bit store each.
; CHECK-LABEL: unaligned:
; CHECK: device_store 0, i16, x
; CHECK: device_store 0, i16, x
define void @unaligned(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %a = load <2 x i16>, ptr addrspace(1) %p, align 4
  %m = add <2 x i16> %a, <i16 1, i16 2>
  store <2 x i16> %m, ptr addrspace(1) %o, align 2
  ret void
}
