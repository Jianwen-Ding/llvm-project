; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-split-wide-counters=0 \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=WIDE
; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-narrow-inductions \
; RUN:   < %s | FileCheck %s --check-prefix=IR

; An index loop to a 64-bit bound no range limits (MLIR's index type): loop
; strength reduction counts it down from %n to zero in 64 bits, a 64-bit
; subtraction and an or of both halves per iteration. The loop counts the
; low half in 32 bits; an enclosing loop runs it again for each 2^32 left.

; CHECK-LABEL: dot:
; CHECK:       [[OUTER:\.LBB[0-9_]+]]: {{.*}}%loop.outer
; CHECK:       [[LOOP:\.LBB[0-9_]+]]: {{.*}}%loop{{$}}
; CHECK-NOT:   _r
; CHECK-NOT:   {{^\s+or\s}}
; CHECK:       while_icmp r0l, nueq, r{{[0-9]+}}, 0, 1
; CHECK-NEXT:  jmp_exec_any [[LOOP]]
; CHECK:       while_icmp r0l, nueq, r{{[0-9]+}}, 0, 1
; CHECK-NEXT:  jmp_exec_any [[OUTER]]
; CHECK:       device_store
; WIDE-LABEL:  dot:
; WIDE:        isub r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 1
; WIDE:        or

; IR-LABEL: define agx_kernel void @dot(
; IR:       [[HI:%.*]] = trunc i64 {{.*}} to i32
; IR:       loop.outer:
; IR:         [[LEFT:%.*]] = phi i32 [ [[HI]], %{{.*}} ], [ [[NEXT:%.*]], %loop.outer{{[0-9]*}} ]
; IR:       loop.outer{{[0-9]+}}:
; IR-NEXT:    [[DONE:%.*]] = icmp eq i32 [[LEFT]], 0
; IR-NEXT:    [[NEXT]] = add i32 [[LEFT]], -1
; IR-NEXT:    br i1 [[DONE]], label %exit, label %loop.outer
define agx_kernel void @dot(ptr addrspace(1) %a, ptr addrspace(1) %b,
                            ptr addrspace(1) %c, i64 %n) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %loop, label %exit

loop:
  %acc = phi float [ 0.0, %entry ], [ %r, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k32 = trunc i64 %k to i32
  %ia = add i32 %x, %k32
  %pa = getelementptr inbounds float, ptr addrspace(1) %a, i32 %ia
  %va = load float, ptr addrspace(1) %pa
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i32 %k32
  %vb = load float, ptr addrspace(1) %pb
  %r = call float @llvm.fma.f32(float %va, float %vb, float %acc)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %s = phi float [ 0.0, %entry ], [ %r, %loop ]
  %pc = getelementptr inbounds float, ptr addrspace(1) %c, i32 %x
  store float %s, ptr addrspace(1) %pc
  ret void
}

; The counter's value after the loop is the bound: the exit phi takes %n.
; CHECK-LABEL: count_out:
; CHECK:       while_icmp r0l, nueq, r{{[0-9]+}}, 0, 1
define agx_kernel void @count_out(ptr addrspace(1) %a, ptr addrspace(1) %out,
                                  i64 %s, i64 %n) {
entry:
  br label %loop

loop:
  %k = phi i64 [ %s, %entry ], [ %k.next, %loop ]
  %k32 = trunc i64 %k to i32
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %k32
  store i32 %k32, ptr addrspace(1) %p
  %k.next = add i64 %k, 1
  %done = icmp eq i64 %k.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %last = phi i64 [ %k.next, %loop ]
  store i64 %last, ptr addrspace(1) %out
  ret void
}

; A trip count below 2^32 (a 32-bit count, widened): the low half alone
; counts, without an enclosing loop.
; CHECK-LABEL: bounded:
; CHECK-NOT:   {{^\s+or\s}}
; CHECK:       while_icmp r0l, nueq, r{{[0-9]+}}, 0, 1
; CHECK-NEXT:  jmp_exec_any
; CHECK-NOT:   while_icmp
; CHECK:       stop
; IR-LABEL: define agx_kernel void @bounded(
; IR-NOT:   .outer
; IR:       ret void
define agx_kernel void @bounded(ptr addrspace(1) %a, i32 %n32) {
entry:
  %pos = icmp sgt i32 %n32, 0
  br i1 %pos, label %ph, label %exit

ph:
  %n = zext nneg i32 %n32 to i64
  br label %loop

loop:
  %k = phi i64 [ 0, %ph ], [ %k.next, %loop ]
  %k32 = trunc i64 %k to i32
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %k32
  store i32 %k32, ptr addrspace(1) %p
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.fma.f32(float, float, float)
