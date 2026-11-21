; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-log-fast-two-sum=0 < %s | FileCheck %s --check-prefix=TWOSUM

; log as hi + lo: f + y and e ln 2 + (f + y) with their errors by Fast2Sum
; (three operations each, the first addend the larger), which gives the
; same exact error as the six of TwoSum (L15-E616). The adds of a lone log:
; e and f, the two sums and their errors, lo and hi + lo.
; CHECK-LABEL: lone:
; CHECK-COUNT-11: fadd32
; CHECK-NOT: fadd32
; CHECK: device_store
; TWOSUM-LABEL: lone:
; TWOSUM-COUNT-17: fadd32
; TWOSUM-NOT: fadd32
; TWOSUM: device_store
define agx_kernel void @lone(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %t to i64
  %pp = getelementptr float, ptr addrspace(1) %p, i64 %z
  %x = load float, ptr addrspace(1) %pp
  %l = call float @llvm.log.f32(float %x)
  %qq = getelementptr float, ptr addrspace(1) %q, i64 %z
  store float %l, ptr addrspace(1) %qq
  ret void
}

declare float @llvm.log.f32(float)
declare i32 @llvm.agx.thread.position.in.grid.x()
