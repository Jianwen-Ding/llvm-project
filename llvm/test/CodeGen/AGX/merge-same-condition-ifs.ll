; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-same-condition-ifs=0 < %s | FileCheck %s --check-prefix=OFF

; Two guarded loads on the same condition (v = c ? a[i] : x; w = c ? b[i] : y,
; the PTX front end's block reductions): the join of the first if only holds
; phis and branches on c again, so the second if's code runs at the end of
; the first: one if (L15-E413).

declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; CHECK-LABEL: guarded:
; CHECK: if_icmp
; CHECK-NOT: if_icmp
; CHECK: threadgroup_load
; CHECK-NOT: pop_exec
; CHECK: threadgroup_load
; CHECK: pop_exec
; CHECK-NOT: if_icmp
; CHECK: stop
; OFF-LABEL: guarded:
; OFF: if_icmp
; OFF: threadgroup_load
; OFF: pop_exec
; OFF: if_icmp
; OFF: threadgroup_load
; OFF: pop_exec
define agx_kernel void @guarded(ptr addrspace(3) %a, ptr addrspace(3) %b, i32 %n, ptr addrspace(1) %o) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %c = icmp ult i32 %t, %n
  br i1 %c, label %t1, label %m
t1:
  %pa = getelementptr float, ptr addrspace(3) %a, i32 %t
  %v = load float, ptr addrspace(3) %pa, align 4
  br label %m
m:
  %p = phi float [ -3.0e38, %entry ], [ %v, %t1 ]
  br i1 %c, label %t2, label %j
t2:
  %pb = getelementptr i32, ptr addrspace(3) %b, i32 %t
  %w = load i32, ptr addrspace(3) %pb, align 4
  %pw = bitcast float %p to i32
  %x = xor i32 %w, %pw
  br label %j
j:
  %q = phi i32 [ 0, %m ], [ %x, %t2 ]
  %po = getelementptr float, ptr addrspace(1) %o, i32 %t
  store float %p, ptr addrspace(1) %po, align 4
  %o2 = getelementptr i32, ptr addrspace(1) %po, i32 1024
  store i32 %q, ptr addrspace(1) %o2, align 4
  ret void
}
