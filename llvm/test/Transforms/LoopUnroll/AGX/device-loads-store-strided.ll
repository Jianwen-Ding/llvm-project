; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-store=0 -agx-device-unroll-strided=0 -S < %s | FileCheck %s --check-prefix=FOUR

; A device-load loop (device-loads.ll) that also stores to device memory (a
; row copy) is unrolled 16 times, and one whose loads all step more than 64
; bytes an iteration (a column sum, at MLIR's zero-extended 32-bit index) 8
; times, where that divides the trip count; 4 otherwise (L15-E235).

define void @copy512(ptr addrspace(1) %p, ptr addrspace(1) %q) {
; CHECK-LABEL: @copy512(
; CHECK-COUNT-16: load <4 x float>
; CHECK-NOT: load <4 x float>
; CHECK: br i1
; FOUR-LABEL: @copy512(
; FOUR-COUNT-4: load <4 x float>
; FOUR-NOT: load <4 x float>
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %g = getelementptr inbounds <4 x float>, ptr addrspace(1) %p, i64 %i
  %v = load <4 x float>, ptr addrspace(1) %g, align 16
  %h = getelementptr inbounds <4 x float>, ptr addrspace(1) %q, i64 %i
  store <4 x float> %v, ptr addrspace(1) %h, align 16
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 512
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @colsum256(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %col) {
; CHECK-LABEL: @colsum256(
; CHECK-COUNT-8: load float
; CHECK-NOT: load float
; CHECK: br i1
; FOUR-LABEL: @colsum256(
; FOUR-COUNT-4: load float
; FOUR-NOT: load float
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %row = shl nuw nsw i32 %i, 9
  %ix = add i32 %row, %col
  %z = zext i32 %ix to i64
  %g = getelementptr inbounds [4 x i8], ptr addrspace(1) %p, i64 %z
  %v = load float, ptr addrspace(1) %g, align 4
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i32 %i, 1
  %c = icmp ult i32 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}

; Contiguous loads stay at 4.
define void @rowsum256(ptr addrspace(1) %p, ptr addrspace(1) %o) {
; CHECK-LABEL: @rowsum256(
; FOUR-LABEL: @rowsum256(
; CHECK-COUNT-4: load float
; CHECK-NOT: load float
; CHECK: br i1
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %z = zext i32 %i to i64
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %z
  %v = load float, ptr addrspace(1) %g, align 4
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i32 %i, 1
  %c = icmp ult i32 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}
