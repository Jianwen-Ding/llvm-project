; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-share-if-levels=0 < %s | FileCheck %s --check-prefix=OFF

; An if without an else whose pop is followed at once by the pop of the
; construct around it deactivates its failing threads by a while at that
; construct's level, and one pop ends both (L15-E412).

; CHECK-LABEL: nested:
; CHECK: if_icmp r0l, ult, r1, u2, 1
; CHECK: while_icmp r0l, sgt, r2, 0, 1
; CHECK: device_store
; CHECK-NOT: pop_exec
; CHECK: .LBB0_4:
; CHECK-NEXT: pop_exec r0l, 1
; CHECK-NEXT: stop
; OFF-LABEL: nested:
; OFF: if_icmp r0l, sgt, r2, 0, 1
; OFF: pop_exec r0l, 1
; OFF: pop_exec r0l, 1
; OFF-NEXT: stop

; Code after the inner if needs its level.
; CHECK-LABEL: after:
; CHECK: if_icmp r0l, sgt, r3, 0, 1
; CHECK: pop_exec r0l, 1
; CHECK: device_store
; CHECK: pop_exec r0l, 1

declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @nested(ptr addrspace(1) %a, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %go = icmp ult i32 %t, %n
  br i1 %go, label %b, label %e
b:
  %p = getelementptr i32, ptr addrspace(1) %a, i32 %t
  %v = load i32, ptr addrspace(1) %p, align 4
  %pos = icmp sgt i32 %v, 0
  br i1 %pos, label %c, label %e
c:
  %w = mul i32 %v, %v
  store i32 %w, ptr addrspace(1) %p, align 4
  br label %e
e:
  ret void
}

define agx_kernel void @after(ptr addrspace(1) %a, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %go = icmp ult i32 %t, %n
  br i1 %go, label %b, label %e
b:
  %p = getelementptr i32, ptr addrspace(1) %a, i32 %t
  %v = load i32, ptr addrspace(1) %p, align 4
  %pos = icmp sgt i32 %v, 0
  br i1 %pos, label %c, label %d
c:
  %w = mul i32 %v, %v
  store i32 %w, ptr addrspace(1) %p, align 4
  br label %d
d:
  %q = getelementptr i32, ptr addrspace(1) %a, i32 %n
  store i32 %t, ptr addrspace(1) %q, align 4
  br label %e
e:
  ret void
}
