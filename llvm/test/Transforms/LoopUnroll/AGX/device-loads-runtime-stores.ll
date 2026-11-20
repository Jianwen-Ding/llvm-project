; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-stores=0 -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-stores-narrow=0 -S < %s | FileCheck %s --check-prefix=WORDS

; A row loop of a runtime length that stores (L15-E565): its contiguous word
; loads and stores, light arithmetic, unrolled 4 times with a remainder loop,
; so that alias versioning's copy has iterations' loads to issue together.

; CHECK-LABEL: @scale(
; CHECK-COUNT-4: store float
; CHECK: .epil
; OFF-LABEL: @scale(
; OFF: store float
; OFF-NOT: store float
; OFF: ret void
define void @scale(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y, i64 %n) {
entry:
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %pw = getelementptr inbounds float, ptr addrspace(1) %w, i64 %i
  %u = load float, ptr addrspace(1) %pw, align 4
  %o = fmul float %v, %u
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; Byte stores (int8 quantization, roundeven) likewise (L15-E567).
; CHECK-LABEL: @quant(
; CHECK-COUNT-4: store i8
; CHECK: .epil
; WORDS-LABEL: @quant(
; WORDS: store i8
; WORDS-NOT: store i8
; WORDS: ret void
define void @quant(ptr addrspace(1) %x, ptr addrspace(1) %q, i64 %n) {
entry:
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %r = call float @llvm.roundeven.f32(float %v)
  %c1 = call float @llvm.maximum.f32(float %r, float -127.0)
  %c2 = call float @llvm.minimum.f32(float %c1, float 127.0)
  %b = fptosi float %c2 to i8
  %pq = getelementptr inbounds i8, ptr addrspace(1) %q, i64 %i
  store i8 %b, ptr addrspace(1) %pq, align 1
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
declare float @llvm.roundeven.f32(float)
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
