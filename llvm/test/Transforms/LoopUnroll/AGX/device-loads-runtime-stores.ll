; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-stores=0 -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-stores-narrow=0 -S < %s | FileCheck %s --check-prefix=WORDS
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-columns=0 -S < %s | FileCheck %s --check-prefix=NOCOL
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-tanh-stores=0 -S < %s | FileCheck %s --check-prefix=NOTANH
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-div-stores=0 -S < %s | FileCheck %s --check-prefix=NODIV

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

; A column of a runtime length (4096 bytes a step, far) likewise, its trip
; count independent of the thread (L15-E575); not a grid-stride loop, whose
; trip count (n - gid) / 1024 is the thread's.
; CHECK-LABEL: @column(
; CHECK-COUNT-4: store float
; CHECK: .epil
; NOCOL-LABEL: @column(
; NOCOL: store float
; NOCOL-NOT: store float
; NOCOL: ret void
; CHECK-LABEL: @gridstride(
; CHECK: store float
; CHECK-NOT: store float
; CHECK: ret void
declare i32 @llvm.agx.thread.position.in.grid.x()
define void @column(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y, i64 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c0 = zext i32 %t to i64
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %row = shl nuw nsw i64 %i, 10
  %k = add nuw nsw i64 %row, %c0
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %k
  %v = load float, ptr addrspace(1) %px, align 4
  %pw = getelementptr inbounds float, ptr addrspace(1) %w, i64 %i
  %u = load float, ptr addrspace(1) %pw, align 4
  %o = fmul float %v, %u
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %k
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @gridstride(ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %g = zext i32 %t to i64
  %e = icmp ult i64 %g, %n
  br i1 %e, label %loop, label %exit
loop:
  %i = phi i64 [ %g, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %o = fmul float %v, %v
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw i64 %i, 1024
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; tanh too, a GELU row (L15-E577).
; CHECK-LABEL: @gelu(
; CHECK-COUNT-4: call float @llvm.tanh.f32
; CHECK: .epil
; NOTANH-LABEL: @gelu(
; NOTANH: call float @llvm.tanh.f32
; NOTANH-NOT: call float @llvm.tanh.f32
; NOTANH: ret void
define void @gelu(ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %n) {
entry:
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %t = call float @llvm.tanh.f32(float %v)
  %o = fmul float %v, %t
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
declare float @llvm.tanh.f32(float)

; A division too, a SiLU row (L15-E578).
; CHECK-LABEL: @silu(
; CHECK-COUNT-4: fdiv float
; CHECK: .epil
; NODIV-LABEL: @silu(
; NODIV: fdiv float
; NODIV-NOT: fdiv float
; NODIV: ret void
define void @silu(ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %n) {
entry:
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %nv = fneg float %v
  %ex = call float @llvm.exp.f32(float %nv)
  %d = fadd float %ex, 1.0
  %o = fdiv float %v, %d
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
declare float @llvm.exp.f32(float)
