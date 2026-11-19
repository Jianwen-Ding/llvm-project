; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-share-loop-levels=0 < %s | FileCheck %s --check-prefix=OFF

; A loop whose exit pop is followed at once by the pop of the construct
; around it runs at that construct's level: no push on entry (if_fcmp r0l,
; eq, 0.0, 0.0, 1) and one pop (L15-E411). Code after the loop in the
; construct (@in_if), or after a loop at the top level (@top), needs the
; loop's own level.

; CHECK-LABEL: in_if:
; CHECK: if_icmp r0l, ult
; CHECK: if_fcmp r0l, eq, 0.0, 0.0, 1
; CHECK: jmp_exec_any
; CHECK: pop_exec r0l, 1
; CHECK-NEXT: device_store
; CHECK: pop_exec r0l, 1
; CHECK-NEXT: stop

; The loop is the else side's last construct.
; CHECK-LABEL: then_else:
; CHECK: else_fcmp r0l, eq, 0.0, 0.0, 1
; CHECK-NOT: if_fcmp
; CHECK: while_icmp r0l, ult
; CHECK-NEXT: jmp_exec_any
; CHECK-NOT: pop_exec
; CHECK: .LBB1_{{[0-9]+}}:
; CHECK-NEXT: pop_exec r0l, 1
; CHECK-NEXT: device_store
; CHECK-NEXT: stop
; OFF-LABEL: then_else:
; OFF: else_fcmp r0l, eq, 0.0, 0.0, 1
; OFF: if_fcmp r0l, eq, 0.0, 0.0, 1
; OFF: jmp_exec_any
; OFF: pop_exec r0l, 1
; OFF: .LBB1_{{[0-9]+}}:
; OFF-NEXT: pop_exec r0l, 1

; CHECK-LABEL: top:
; CHECK: if_fcmp r0l, eq, 0.0, 0.0, 1
; CHECK: jmp_exec_any
; CHECK: pop_exec r0l, 1
; CHECK-NEXT: device_store

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @in_if(ptr addrspace(1) %a, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %go = icmp ult i32 %t, %n
  br i1 %go, label %l, label %e
l:
  %i = phi i32 [ %t, %entry ], [ %i1, %l ]
  %s = phi i32 [ 0, %entry ], [ %s1, %l ]
  %p = getelementptr i32, ptr addrspace(1) %a, i32 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  %s1 = add i32 %s, %v
  %i1 = add nuw i32 %i, 64
  %more = icmp ult i32 %i1, %n
  br i1 %more, label %l, label %x
x:
  %q = getelementptr i32, ptr addrspace(1) %a, i32 %t
  store i32 %s1, ptr addrspace(1) %q, align 4
  br label %e
e:
  ret void
}

define agx_kernel void @then_else(ptr addrspace(1) %a, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %go = icmp ult i32 %t, %n
  br i1 %go, label %l, label %el
l:
  %i = phi i32 [ %t, %entry ], [ %i1, %l ]
  %s = phi i32 [ 0, %entry ], [ %s1, %l ]
  %p = getelementptr i32, ptr addrspace(1) %a, i32 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  %s1 = add i32 %s, %v
  %i1 = add nuw i32 %i, 64
  %more = icmp ult i32 %i1, %n
  br i1 %more, label %l, label %e
el:
  %w = mul i32 %t, 3
  br label %e
e:
  %r = phi i32 [ %s1, %l ], [ %w, %el ]
  %q = getelementptr i32, ptr addrspace(1) %a, i32 %t
  store i32 %r, ptr addrspace(1) %q, align 4
  ret void
}

define agx_kernel void @top(ptr addrspace(1) %a, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  br label %l
l:
  %i = phi i32 [ %t, %entry ], [ %i1, %l ]
  %s = phi i32 [ 0, %entry ], [ %s1, %l ]
  %p = getelementptr i32, ptr addrspace(1) %a, i32 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  %s1 = add i32 %s, %v
  %i1 = add nuw i32 %i, 64
  %more = icmp ult i32 %i1, %n
  br i1 %more, label %l, label %e
e:
  %q = getelementptr i32, ptr addrspace(1) %a, i32 %t
  store i32 %s1, ptr addrspace(1) %q, align 4
  ret void
}
