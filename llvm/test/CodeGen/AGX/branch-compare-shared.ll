; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs \
; RUN:   -agx-fold-shared-branch-compares=0 < %s | FileCheck %s --check-prefix=OFF

; Two blocks branch on lane == 0 (a SIMD-group reduction's first lane writes,
; before and after a barrier). MachineCSE gives both branches one icmpsel;
; each branch takes the comparison itself instead of testing the select's
; result.

; CHECK-LABEL: first_lane_twice:
; CHECK-NOT: icmpsel
; CHECK: if_icmp r0l, ueq, [[L:r[0-9]+]], 0, 1
; CHECK: threadgroup_barrier
; CHECK: if_icmp r0l, ueq, [[L]], 0, 1
; CHECK: stop
; OFF-LABEL: first_lane_twice:
; OFF: icmpsel ueq, [[C:r[0-9]+]], {{r[0-9]+}}, 0, 1, 0
; OFF: if_icmp r0l, nueq, [[C]], 0, 1
; OFF: if_icmp r0l, nueq, [[C]], 0, 1
@tg = internal addrspace(3) global [8 x i32] undef, align 4
define agx_kernel void @first_lane_twice(ptr addrspace(1) %out, i32 %v) {
entry:
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %sg = call i32 @llvm.agx.simdgroup.index.in.threadgroup()
  %first = icmp eq i32 %lane, 0
  br i1 %first, label %write, label %sync
write:
  %s = getelementptr [8 x i32], ptr addrspace(3) @tg, i32 0, i32 %sg
  store i32 %v, ptr addrspace(3) %s
  br label %sync
sync:
  call void @llvm.agx.threadgroup.barrier()
  %x = load i32, ptr addrspace(3) @tg
  br i1 %first, label %store, label %done
store:
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %sg
  store i32 %x, ptr addrspace(1) %q
  br label %done
done:
  ret void
}
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.simdgroup.index.in.threadgroup()
declare void @llvm.agx.threadgroup.barrier()
