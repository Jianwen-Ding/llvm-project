; bf16 axpy: y = a * x + y on four bf16s per thread (8-byte loads of x and y, fpext to float, fmas, RNE back to
; bf16 and one 8-byte store)
; agx-cost: --uniform 4=1069547520 --wg 256
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #0

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define agx_kernel void @axpy_bf16(ptr addrspace(1) noalias readonly captures(none) %x, ptr addrspace(1) noalias nofree captures(none) %y, float %a) local_unnamed_addr #1 {
  %i = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %1 = sext i32 %i to i64
  %px = getelementptr [8 x i8], ptr addrspace(1) %x, i64 %1
  %py = getelementptr [8 x i8], ptr addrspace(1) %y, i64 %1
  %vx = load <4 x bfloat>, ptr addrspace(1) %px, align 8
  %vy = load <4 x bfloat>, ptr addrspace(1) %py, align 8
  %fx = fpext <4 x bfloat> %vx to <4 x float>
  %fy = fpext <4 x bfloat> %vy to <4 x float>
  %s0 = insertelement <4 x float> poison, float %a, i64 0
  %s = shufflevector <4 x float> %s0, <4 x float> poison, <4 x i32> zeroinitializer
  %r = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %s, <4 x float> %fx, <4 x float> %fy)
  %b = fptrunc <4 x float> %r to <4 x bfloat>
  store <4 x bfloat> %b, ptr addrspace(1) %py, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) #2

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
