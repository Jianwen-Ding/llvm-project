; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-some-strided=0 -S < %s | FileCheck %s --check-prefix=FOUR

; A device-load loop with a strided load beside a contiguous one (a naive
; matrix product: B's column, 1 KiB a step, beside A's row) takes the strided
; factor, 16 (L15-E535); 4 when every load must be strided.

define void @matmul_naive(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i64 %x, i64 %y) {
; CHECK-LABEL: @matmul_naive(
; CHECK-COUNT-16: call float @llvm.fma.f32
; CHECK-NOT: call float @llvm.fma.f32
; CHECK: br i1
; FOUR-LABEL: @matmul_naive(
; FOUR-COUNT-4: call float @llvm.fma.f32
; FOUR-NOT: call float @llvm.fma.f32
entry:
  %row = mul i64 %y, 256
  br label %loop
loop:
  %k = phi i64 [ 0, %entry ], [ %k1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %m, %loop ]
  %ia = add i64 %row, %k
  %pa = getelementptr inbounds float, ptr addrspace(1) %a, i64 %ia
  %u = load float, ptr addrspace(1) %pa, align 4
  %kb = mul i64 %k, 256
  %ib = add i64 %kb, %x
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %ib
  %v = load float, ptr addrspace(1) %pb, align 4
  %m = call float @llvm.fma.f32(float %u, float %v, float %acc)
  %k1 = add nuw nsw i64 %k, 1
  %cnd = icmp ult i64 %k1, 256
  br i1 %cnd, label %loop, label %exit
exit:
  %pc = getelementptr inbounds float, ptr addrspace(1) %c, i64 %ia
  store float %m, ptr addrspace(1) %pc, align 4
  ret void
}
declare float @llvm.fma.f32(float, float, float)
