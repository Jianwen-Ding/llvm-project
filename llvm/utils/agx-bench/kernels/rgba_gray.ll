; Image: RGBA8 to 8-bit luma, y = round(0.299 r + 0.587 g + 0.114 b) clamped to 255, one pixel per thread.
; agx-cost: --wg 256
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #0
declare float @llvm.fmuladd.f32(float, float, float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.rint.f32(float) #1
declare i32 @llvm.smax.i32(i32, i32) #1
declare i32 @llvm.umin.i32(i32, i32) #3
define agx_kernel void @gray(ptr addrspace(1) noalias nofree readonly captures(none) %in, ptr addrspace(1) noalias nofree writeonly captures(none) %out) local_unnamed_addr #2 {
  %t = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %1 = sext i32 %t to i64
  %q = getelementptr [4 x i8], ptr addrspace(1) %in, i64 %1
  %w = load i32, ptr addrspace(1) %q, align 4
  %r8 = and i32 %w, 255
  %g0 = lshr i32 %w, 8
  %g8 = and i32 %g0, 255
  %b0 = lshr i32 %w, 16
  %b8 = and i32 %b0, 255
  %r = uitofp nneg i32 %r8 to float
  %g = uitofp nneg i32 %g8 to float
  %b = uitofp nneg i32 %b8 to float
  %y0 = fmul nnan float %r, 2.990000e-01
  %y1 = tail call float @llvm.fmuladd.f32(float %g, float 5.870000e-01, float %y0)
  %y2 = tail call float @llvm.fmuladd.f32(float %b, float 1.140000e-01, float %y1)
  %y3 = tail call float @llvm.rint.f32(float %y2)
  %y4 = tail call float @llvm.minnum.f32(float %y3, float 2.550000e+02)
  %yi = fptoui float %y4 to i8
  %qo = getelementptr i8, ptr addrspace(1) %out, i64 %1
  store i8 %yi, ptr addrspace(1) %qo, align 1
  ret void
}
attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
