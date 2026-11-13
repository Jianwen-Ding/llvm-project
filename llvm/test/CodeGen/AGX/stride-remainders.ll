; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-stride-remainders=0 < %s | FileCheck %s --check-prefix=OFF

; A tile copied by the threadgroup, for (l = tid; l < n; l += 256) with
; column l % 16 (the PTX front end's l - (l & -16)): the step adds only
; multiples of 16, so the column is tid's, computed before the loop
; (L15-E249).

declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; CHECK-LABEL: tile:
; CHECK: and {{r[0-9]+}}, {{r[0-9]+}}, 15
; CHECK: if_fcmp
; CHECK-NOT: and
; CHECK: jmp_exec_any
; OFF-LABEL: tile:
; OFF: if_fcmp
; OFF: and
; OFF: jmp_exec_any
define agx_kernel void @tile(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %go = icmp slt i32 %t, %n
  br i1 %go, label %l, label %e
l:
  %i = phi i32 [ %t, %entry ], [ %i1, %l ]
  %hi = and i32 %i, -16
  %c = sub i32 %i, %hi
  %p = getelementptr float, ptr addrspace(1) %src, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %q = getelementptr float, ptr addrspace(1) %dst, i32 %c
  store float %v, ptr addrspace(1) %q, align 4
  %i1 = add nsw i32 %i, 256
  %more = icmp slt i32 %i1, %n
  br i1 %more, label %l, label %e
e:
  ret void
}
