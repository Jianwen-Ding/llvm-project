; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-select-implied-arms=0 < %s | FileCheck %s --check-prefix=OFF
;
; An i64 compare with a constant below 2^32 expands to select (hi == 0,
; lo cmp K, hi cmp 0). For a >u K the false arm, hi >u 0, is hi != 0, which
; the select's own condition decides there: 1 (L15-E639). Two compare-selects
; instead of three, as a <u K already took (its hi <u 0 folds to 0).

; CHECK-LABEL: {{^}}ugtv:
; CHECK: icmpsel ugt, [[L:r[0-9]+]], {{r[0-9]+}}, 254, 1, 0
; CHECK-NEXT: icmpsel ueq, {{r[0-9]+}}, {{r[0-9]+}}, 0, [[L]], 1
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: {{^}}ugtv:
; OFF-COUNT-3: icmpsel
; OFF: stop

; CHECK-LABEL: {{^}}ultv:
; CHECK: icmpsel ult, [[M:r[0-9]+]], {{r[0-9]+}}, 255, 1, 0
; CHECK-NEXT: icmpsel ueq, {{r[0-9]+}}, {{r[0-9]+}}, 0, [[M]], 0
; CHECK-NOT: icmpsel
; CHECK: stop

define agx_kernel void @ugtv(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %g = getelementptr i64, ptr addrspace(1) %q, i32 %i
  %x = load i64, ptr addrspace(1) %g, align 8
  %c = icmp ugt i64 %x, 254
  %z = zext i1 %c to i32
  %h = getelementptr i32, ptr addrspace(1) %p, i32 %i
  store i32 %z, ptr addrspace(1) %h, align 4
  ret void
}
define agx_kernel void @ultv(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %g = getelementptr i64, ptr addrspace(1) %q, i32 %i
  %x = load i64, ptr addrspace(1) %g, align 8
  %c = icmp ult i64 %x, 255
  %z = zext i1 %c to i32
  %h = getelementptr i32, ptr addrspace(1) %p, i32 %i
  store i32 %z, ptr addrspace(1) %h, align 4
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
