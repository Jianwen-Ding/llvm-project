; Image: NV12-style YUV (BT.601 integer form) to RGBA8 with clamps to [0, 255] and alpha 255, one pixel per thread.
; agx-cost: --wg 256
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #0
declare float @llvm.fmuladd.f32(float, float, float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.rint.f32(float) #1
declare i32 @llvm.smax.i32(i32, i32) #1
declare i32 @llvm.umin.i32(i32, i32) #3
define agx_kernel void @yuv2rgb(ptr addrspace(1) noalias nofree readonly captures(none) %y, ptr addrspace(1) noalias nofree readonly captures(none) %uv, ptr addrspace(1) noalias nofree writeonly captures(none) %out) local_unnamed_addr #2 {
  %t = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %1 = sext i32 %t to i64
  %qy = getelementptr i8, ptr addrspace(1) %y, i64 %1
  %yb = load i8, ptr addrspace(1) %qy, align 1
  %h = lshr i32 %t, 1
  %2 = zext nneg i32 %h to i64
  %quv = getelementptr [2 x i8], ptr addrspace(1) %uv, i64 %2
  %uvv = load <2 x i8>, ptr addrspace(1) %quv, align 2
  %ub = extractelement <2 x i8> %uvv, i64 0
  %vb = extractelement <2 x i8> %uvv, i64 1
  %Y = zext i8 %yb to i32
  %U = zext i8 %ub to i32
  %V = zext i8 %vb to i32
  %d = add nsw i32 %U, -128
  %e = add nsw i32 %V, -128
  %3 = mul nuw nsw i32 %Y, 298
  %r0 = mul nsw i32 %e, 409
  %r1 = add nsw i32 %3, -4640
  %r2 = add nsw i32 %r1, %r0
  %r3 = ashr i32 %r2, 8
  %g0 = mul nsw i32 %d, -100
  %g1 = mul nsw i32 %e, -208
  %g3 = add nsw i32 %r1, %g0
  %g4 = add nsw i32 %g3, %g1
  %g5 = ashr i32 %g4, 8
  %b0 = mul nsw i32 %d, 516
  %b2 = add nsw i32 %r1, %b0
  %b3 = ashr i32 %b2, 8
  %rc = tail call i32 @llvm.smax.i32(i32 %r3, i32 0)
  %4 = tail call i32 @llvm.umin.i32(i32 %rc, i32 255)
  %gc = tail call i32 @llvm.smax.i32(i32 %g5, i32 0)
  %5 = tail call i32 @llvm.umin.i32(i32 %gc, i32 255)
  %bc = tail call i32 @llvm.smax.i32(i32 %b3, i32 0)
  %6 = tail call i32 @llvm.umin.i32(i32 %bc, i32 255)
  %gs = shl nuw nsw i32 %5, 8
  %bs = shl nuw nsw i32 %6, 16
  %p0 = or disjoint i32 %gs, %4
  %p1 = or disjoint i32 %p0, %bs
  %p2 = or disjoint i32 %p1, -16777216
  %qo = getelementptr [4 x i8], ptr addrspace(1) %out, i64 %1
  store i32 %p2, ptr addrspace(1) %qo, align 4
  ret void
}
attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
