; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-light-row-stores-apart=0 < %s | FileCheck %s --check-prefix=CLUSTER

; A thread's row of light work (halves widened and scaled, L15-E589's
; region): each 16-byte store issues once its value is ready, the first
; before the wait for the second half of the loads (L15-E619), not held
; until both are (widen_dyn: 4 GPRs more).
; CHECK-LABEL: widen:
; CHECK: wait 0
; CHECK-NOT: wait
; CHECK: device_store
; CHECK: wait 1
; CLUSTER-LABEL: widen:
; CLUSTER: wait 0
; CLUSTER-NOT: device_store
; CLUSTER: wait 1
; CLUSTER: device_store
; CLUSTER-NEXT: device_store
define agx_kernel void @widen(ptr addrspace(1) %x, ptr addrspace(1) %y, float %s, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 8
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %k = add i32 %row, %j
  %z = zext i32 %k to i64
  %p = getelementptr half, ptr addrspace(1) %x, i64 %z
  %v = load <8 x half>, ptr addrspace(1) %p, align 4
  %e = fpext <8 x half> %v to <8 x float>
  %si = insertelement <8 x float> poison, float %s, i64 0
  %ss = shufflevector <8 x float> %si, <8 x float> poison, <8 x i32> zeroinitializer
  %m = fmul <8 x float> %e, %ss
  %q = getelementptr float, ptr addrspace(1) %y, i64 %z
  store <8 x float> %m, ptr addrspace(1) %q, align 4
  %ja = add i32 %j, 8
  %B_k = add i32 %row, %ja
  %B_z = zext i32 %B_k to i64
  %B_p = getelementptr half, ptr addrspace(1) %x, i64 %B_z
  %B_v = load <8 x half>, ptr addrspace(1) %B_p, align 4
  %B_e = fpext <8 x half> %B_v to <8 x float>
  %B_si = insertelement <8 x float> poison, float %s, i64 0
  %B_ss = shufflevector <8 x float> %B_si, <8 x float> poison, <8 x i32> zeroinitializer
  %B_m = fmul <8 x float> %B_e, %B_ss
  %B_q = getelementptr float, ptr addrspace(1) %y, i64 %B_z
  store <8 x float> %B_m, ptr addrspace(1) %B_q, align 4
  %j1 = add i32 %j, 16
  %c = icmp ult i32 %j1, %n
  br i1 %c, label %loop, label %out
out:
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
