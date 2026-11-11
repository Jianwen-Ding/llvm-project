; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-codegenprepare < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null

; An integer of more than 24 bits converts to bfloat with one rounding: |x|
; rounded to odd at 24 significant bits (the lower bits cleared, a sticky bit
; in the lowest kept one), which converts to float exactly, then rounded to
; nearest bfloat. (Through float rounded to nearest, 21430271 gave 0x4ba4,
; not 0x4ba3.) alive-tv proves the i32 and i64 forms.

; CHECK-LABEL: @s32(
; CHECK: [[A:%.*]] = call i32 @llvm.abs.i32(i32 %x, i1 false)
; CHECK: call i32 @llvm.ctlz.i32(i32 [[A]], i1 false)
; CHECK: [[F:%.*]] = uitofp i32 {{.*}} to float
; CHECK: [[R:%.*]] = fptrunc float [[F]] to bfloat
; CHECK: fneg bfloat [[R]]
define agx_kernel void @s32(ptr addrspace(1) %p, i32 %x) {
  %r = sitofp i32 %x to bfloat
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @u64(
; CHECK: call i64 @llvm.ctlz.i64(i64 %x, i1 false)
; CHECK: uitofp i64 {{.*}} to float
; CHECK-NEXT: fptrunc float {{.*}} to bfloat
; CHECK-NOT: fneg
define agx_kernel void @u64(ptr addrspace(1) %p, i64 %x) {
  %r = uitofp i64 %x to bfloat
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; 24 bits or fewer convert to float exactly: unchanged.
; CHECK-LABEL: @s16(
; CHECK: sitofp i16 %x to bfloat
define agx_kernel void @s16(ptr addrspace(1) %p, i16 %x) {
  %r = sitofp i16 %x to bfloat
  store bfloat %r, ptr addrspace(1) %p
  ret void
}
