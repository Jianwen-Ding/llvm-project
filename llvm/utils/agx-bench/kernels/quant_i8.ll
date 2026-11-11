; int8 quantization: q[4i..4i+3] = clamp(round(x[4i..4i+3] / scale), -127, 127), a float4 in, a packed word out
; agx-cost: --uniform 4=1065353216 --wg 256
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #0

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.round.v4f32(<4 x float>) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define agx_kernel void @quant(ptr addrspace(1) noalias nofree readonly captures(none) %x, ptr addrspace(1) noalias nofree writeonly captures(none) %q, float %scale) local_unnamed_addr #2 {
  %i = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %1 = sext i32 %i to i64
  %px = getelementptr [16 x i8], ptr addrspace(1) %x, i64 %1
  %v = load <4 x float>, ptr addrspace(1) %px, align 16
  %inv = fdiv float 1.000000e+00, %scale
  %s0 = insertelement <4 x float> poison, float %inv, i64 0
  %s = shufflevector <4 x float> %s0, <4 x float> poison, <4 x i32> zeroinitializer
  %m = fmul <4 x float> %s, %v
  %r = tail call <4 x float> @llvm.round.v4f32(<4 x float> %m)
  %l = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %r, <4 x float> splat (float -1.270000e+02))
  %h = tail call <4 x float> @llvm.minnum.v4f32(<4 x float> %l, <4 x float> splat (float 1.270000e+02))
  %c = fptosi <4 x float> %h to <4 x i8>
  %pq = getelementptr [4 x i8], ptr addrspace(1) %q, i64 %1
  store <4 x i8> %c, ptr addrspace(1) %pq, align 4
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="agx-g14c" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "target-cpu"="agx-g14c" }
