; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs \
; RUN:   -agx-select-count-half=0 < %s | FileCheck %s --check-prefix=OFF

; A 64-bit ctlz / cttz selects the half that decides first and counts it
; with one ffs (-1 for zero, as an M2 Max gives it): 31 - ffs(hi), or
; 63 - ffs(lo) where hi is zero, 64 for zero, where the function has register
; headroom; not both halves' counts and a select.

; CHECK-LABEL: lz:
; CHECK-COUNT-1: ffs
; CHECK-NOT: ffs
; CHECK: stop
; OFF-LABEL: lz:
; OFF-COUNT-2: ffs
; CHECK-LABEL: tz:
; CHECK-COUNT-1: bitrev
; CHECK-NOT: bitrev
; CHECK: stop
; OFF-LABEL: tz:
; OFF-COUNT-2: bitrev
define agx_kernel void @lz(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i64, ptr addrspace(1) %p, i32 %t
  %x = load i64, ptr addrspace(1) %q, align 8
  %y = call i64 @llvm.ctlz.i64(i64 %x, i1 false)
  store i64 %y, ptr addrspace(1) %q, align 8
  ret void
}
define agx_kernel void @tz(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i64, ptr addrspace(1) %p, i32 %t
  %x = load i64, ptr addrspace(1) %q, align 8
  %y = call i64 @llvm.cttz.i64(i64 %x, i1 false)
  store i64 %y, ptr addrspace(1) %q, align 8
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.cttz.i64(i64, i1)
