; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-narrow-inductions < %s | FileCheck %s

; Two counters with the same start and the same constant step (a narrowed
; load index beside an index_cast's i32 copy of it in an MLIR argmax) hold
; the same value: one phi and one increment (L15-E431). @different's twin
; steps by 3 against 4: both stay.

; CHECK-LABEL: @same(
; CHECK:       %i.lo = phi i32
; CHECK-NOT:   %j = phi
; CHECK:       xor i32 %v, %i.lo
; CHECK-NOT:   %jn =
; CHECK:       ret void
define agx_kernel void @same(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %j = phi i32 [ 0, %entry ], [ %jn, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %an, %loop ]
  %a = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %a, align 4
  %x = xor i32 %v, %j
  %an = add i32 %acc, %x
  %in = add nuw nsw i64 %i, 4
  %jn = add nuw nsw i32 %j, 4
  %c = icmp ult i64 %in, 256
  br i1 %c, label %loop, label %exit
exit:
  store i32 %an, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: @different(
; CHECK:       %i.lo = phi i32
; CHECK:       %j = phi i32
; CHECK:       xor i32 %v, %j
; CHECK:       ret void
define agx_kernel void @different(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %j = phi i32 [ 0, %entry ], [ %jn, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %an, %loop ]
  %a = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %a, align 4
  %x = xor i32 %v, %j
  %an = add i32 %acc, %x
  %in = add nuw nsw i64 %i, 4
  %jn = add nuw nsw i32 %j, 3
  %c = icmp ult i64 %in, 256
  br i1 %c, label %loop, label %exit
exit:
  store i32 %an, ptr addrspace(1) %o, align 4
  ret void
}
