; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs -agx-split-wide-sums=0 %s -o - | FileCheck %s --check-prefix=OFF

; A 64-bit sum of zero-extended 32-bit values is kept in two 32-bit words, a
; carry compare per value instead of a 64-bit add (L15-E678).

; CHECK-LABEL: sum_u32:
; CHECK: .LBB0_1:
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}{{$}}
; CHECK: icmpsel ult
; CHECK: jmp_exec_any .LBB0_1
; CHECK: device_store 0, i32, xy
; OFF-LABEL: sum_u32:
; OFF: .LBB0_1:
; OFF: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}
; OFF-NOT: icmpsel ult
; OFF: jmp_exec_any .LBB0_1
define agx_kernel void @sum_u32(ptr addrspace(1) %x, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi i64 [ 7, %entry ], [ %s.next, %loop ]
  %p = getelementptr inbounds i32, ptr addrspace(1) %x, i32 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  %w = zext i32 %v to i64
  %s.next = add i64 %s, %w
  %i.next = add nuw i32 %i, 1
  %c = icmp ult i32 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  store i64 %s.next, ptr addrspace(1) %out, align 8
  ret void
}

; A sum the loop also reads is the pair of its words.
; CHECK-LABEL: sum_read:
; CHECK: .LBB1_1:
; CHECK: device_store 0, i32, xy, r[[LO:[0-9]+]]_r{{[0-9]+}}
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}{{$}}
; CHECK: iadd r{{[0-9]+}}, r[[LO]], r{{[0-9]+}}
; CHECK: icmpsel ult
; CHECK: jmp_exec_any .LBB1_1
define agx_kernel void @sum_read(ptr addrspace(1) %x, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi i64 [ 0, %entry ], [ %s.next, %loop ]
  %p = getelementptr inbounds i32, ptr addrspace(1) %x, i32 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  %w = zext i32 %v to i64
  %s.next = add i64 %s, %w
  %q = getelementptr inbounds i64, ptr addrspace(1) %out, i32 %i
  store i64 %s, ptr addrspace(1) %q, align 8
  %i.next = add nuw i32 %i, 1
  %c = icmp ult i32 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
