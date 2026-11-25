; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs -agx-double-word-counters=0 < %s | FileCheck %s --check-prefix=OFF

; LSR walks the 64-bit elements with a pointer next to the 32-bit counter;
; the access takes the counter as its index register (two words an element,
; lsl 1) and the pointer's 64-bit add goes (L15-E680).

; CHECK-LABEL: sum_i64:
; CHECK: .LBB0_1:
; CHECK: device_load 0, i32, xy, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r[[J:[0-9]+]], unsigned, lsl 1
; CHECK-NEXT: iadd r[[J]], r[[J]], 1
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 8
; CHECK: jmp_exec_any .LBB0_1
; OFF-LABEL: sum_i64:
; OFF: .LBB0_1:
; OFF: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 8
define agx_kernel void @sum_i64(ptr addrspace(1) %x, ptr addrspace(1) %out) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 8
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi i64 [ 0, %entry ], [ %s.next, %loop ]
  %j = add i32 %row, %i
  %p = getelementptr inbounds i64, ptr addrspace(1) %x, i32 %j
  %v = load i64, ptr addrspace(1) %p, align 8
  %s.next = add i64 %s, %v
  %i.next = add nuw i32 %i, 1
  %c = icmp ult i32 %i.next, 256
  br i1 %c, label %loop, label %exit
exit:
  %q = getelementptr inbounds i64, ptr addrspace(1) %out, i32 %t
  store i64 %s.next, ptr addrspace(1) %q, align 8
  ret void
}

; Every other element: four words a step (lsl 2), the load and the store.
; CHECK-LABEL: step2_i64:
; CHECK: .LBB1_1:
; CHECK: device_load 0, i32, xy, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r[[K:[0-9]+]], unsigned, lsl 2
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 16
; CHECK: device_store 0, i32, xy, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r[[K]], unsigned, lsl 2
; CHECK: jmp_exec_any .LBB1_1
define agx_kernel void @step2_i64(ptr addrspace(1) noalias %x, ptr addrspace(1) noalias %y) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 6
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %is = shl i32 %i, 1
  %j = add i32 %row, %is
  %p = getelementptr inbounds i64, ptr addrspace(1) %x, i32 %j
  %v = load i64, ptr addrspace(1) %p, align 8
  %r = mul i64 %v, 3
  %q = getelementptr inbounds i64, ptr addrspace(1) %y, i32 %j
  store i64 %r, ptr addrspace(1) %q, align 8
  %i.next = add nuw i32 %i, 1
  %c = icmp ult i32 %i.next, 32
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
