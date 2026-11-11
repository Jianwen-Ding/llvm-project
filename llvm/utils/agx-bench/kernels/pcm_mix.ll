; int16 audio mixing: out = sat16(a + b) and g = sat16((a * gain) >> 8), two samples per thread
; agx-cost: --uniform 8=300 --wg 256
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #0

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define agx_kernel void @pcm_mix(ptr addrspace(1) noalias readonly captures(none) %a, ptr addrspace(1) noalias readonly captures(none) %b, ptr addrspace(1) noalias nofree writeonly captures(none) %out, ptr addrspace(1) noalias nofree writeonly captures(none) %g, i32 %gain) local_unnamed_addr #1 {
  %i = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %1 = sext i32 %i to i64
  %pa = getelementptr [4 x i8], ptr addrspace(1) %a, i64 %1
  %pb = getelementptr [4 x i8], ptr addrspace(1) %b, i64 %1
  %va = load <2 x i16>, ptr addrspace(1) %pa, align 4
  %vb = load <2 x i16>, ptr addrspace(1) %pb, align 4
  %ea = sext <2 x i16> %va to <2 x i32>
  %2 = tail call <2 x i16> @llvm.sadd.sat.v2i16(<2 x i16> %va, <2 x i16> %vb)
  %po = getelementptr [4 x i8], ptr addrspace(1) %out, i64 %1
  store <2 x i16> %2, ptr addrspace(1) %po, align 4
  %gs = insertelement <2 x i32> poison, i32 %gain, i64 0
  %gv = shufflevector <2 x i32> %gs, <2 x i32> poison, <2 x i32> zeroinitializer
  %m = mul nsw <2 x i32> %gv, %ea
  %m8 = ashr <2 x i32> %m, splat (i32 8)
  %m1 = tail call <2 x i32> @llvm.smax.v2i32(<2 x i32> %m8, <2 x i32> splat (i32 -32768))
  %m2 = tail call <2 x i32> @llvm.smin.v2i32(<2 x i32> %m1, <2 x i32> splat (i32 32767))
  %mt = trunc nsw <2 x i32> %m2 to <2 x i16>
  %pg = getelementptr [4 x i8], ptr addrspace(1) %g, i64 %1
  store <2 x i16> %mt, ptr addrspace(1) %pg, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i32> @llvm.smax.v2i32(<2 x i32>, <2 x i32>) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i32> @llvm.smin.v2i32(<2 x i32>, <2 x i32>) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i16> @llvm.sadd.sat.v2i16(<2 x i16>, <2 x i16>) #3

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "target-cpu"="agx-g14c" }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
