; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-uniform-constants -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s --check-prefix=FOUR

; A device-load loop of exponentials that no other factor takes is unrolled 8
; times where the constants are in uniforms (L15-E542), 4 otherwise.

; CHECK-LABEL: @exps(
; CHECK-COUNT-8: call float @llvm.exp.f32
; CHECK-NOT: call float @llvm.exp.f32
; FOUR-LABEL: @exps(
; FOUR-COUNT-4: call float @llvm.exp.f32
; FOUR-NOT: call float @llvm.exp.f32
define void @exps(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %i
  %v = load float, ptr addrspace(1) %g, align 4
  %e = call float @llvm.exp.f32(float %v)
  %s1 = fadd float %s, %e
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}
declare float @llvm.exp.f32(float)
