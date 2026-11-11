; Adam optimizer step: m = b1 m + (1 - b1) g, v = b2 v + (1 - b2) g^2, p += lr * c1 m / (sqrt(c2 v) + eps)
; agx-cost: --uniform 8=3125723603 --uniform 9=1063675494 --uniform 10=1065336439 --uniform 11=841731191 --uniform 12=1065353216 --uniform 13=1065353216 --wg 256
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #0

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define agx_kernel void @adam(ptr addrspace(1) noalias nofree captures(none) %p, ptr addrspace(1) noalias nofree readonly captures(none) %g, ptr addrspace(1) noalias nofree captures(none) %m, ptr addrspace(1) noalias nofree captures(none) %v, float %lr, float %b1, float %b2, float %eps, float %c1, float %c2) local_unnamed_addr #2 {
  %i = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %1 = sext i32 %i to i64
  %pp = getelementptr [4 x i8], ptr addrspace(1) %p, i64 %1
  %pg = getelementptr [4 x i8], ptr addrspace(1) %g, i64 %1
  %pm = getelementptr [4 x i8], ptr addrspace(1) %m, i64 %1
  %pv = getelementptr [4 x i8], ptr addrspace(1) %v, i64 %1
  %P = load float, ptr addrspace(1) %pp, align 4
  %G = load float, ptr addrspace(1) %pg, align 4
  %M = load float, ptr addrspace(1) %pm, align 4
  %V = load float, ptr addrspace(1) %pv, align 4
  %nb1 = fsub float 1.000000e+00, %b1
  %nb2 = fsub float 1.000000e+00, %b2
  %t1 = fmul float %nb1, %G
  %M2 = tail call float @llvm.fmuladd.f32(float %b1, float %M, float %t1)
  %gg = fmul float %G, %G
  %t2 = fmul float %nb2, %gg
  %V2 = tail call float @llvm.fmuladd.f32(float %b2, float %V, float %t2)
  %mh = fmul float %c1, %M2
  %vh = fmul float %c2, %V2
  %sq = tail call float @llvm.sqrt.f32(float %vh)
  %den = fadd float %eps, %sq
  %u = fdiv float %mh, %den
  %P2 = tail call float @llvm.fmuladd.f32(float %u, float %lr, float %P)
  store float %P2, ptr addrspace(1) %pp, align 4
  store float %M2, ptr addrspace(1) %pm, align 4
  store float %V2, ptr addrspace(1) %pv, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="agx-g14c" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "target-cpu"="agx-g14c" }
