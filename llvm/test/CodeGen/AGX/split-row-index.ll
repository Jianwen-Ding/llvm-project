; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-split-row-index=0 < %s | FileCheck %s --check-prefix=OFF

; A row + column address, base + ((row + sext(i)) << 2), is accessed through
; the base base + (row << 2) and the 32-bit index i (llama.cpp's k_bin_bcast:
; the store through a whole 64-bit register address is ~1.6x slower).

; CHECK-LABEL: int_form:
; CHECK: device_load 0, i32, x, r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}, signed
; CHECK: device_store 0, i32, x, r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}, signed
; OFF-LABEL: int_form:
; OFF: device_store 0, i32, x, r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 0, signed
define void @int_form(i64 %d, i64 %s, i64 %row, i32 %i) {
  %e = sext i32 %i to i64
  %ix = add i64 %row, %e
  %o = shl i64 %ix, 2
  %as = add i64 %s, %o
  %ps = inttoptr i64 %as to ptr addrspace(1)
  %v = load float, ptr addrspace(1) %ps
  %w = fadd float %v, 1.0
  %ad = add i64 %d, %o
  %pd = inttoptr i64 %ad to ptr addrspace(1)
  store float %w, ptr addrspace(1) %pd
  ret void
}

; CHECK-LABEL: gep_form:
; CHECK: device_store 0, i32, x, r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}, unsigned
define void @gep_form(ptr addrspace(1) %d, i64 %row, i32 %i, float %v) {
  %e = zext i32 %i to i64
  %ix = add i64 %row, %e
  %pd = getelementptr float, ptr addrspace(1) %d, i64 %ix
  store float %v, ptr addrspace(1) %pd
  ret void
}
