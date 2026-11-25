; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merged-index-chains=0 < %s | FileCheck %s --check-prefix=OFF

; A device access at a chain of geps (a memref subview's offset, then the
; element's index, as expand-strided-metadata lowers them), whose indices
; the thread bounds keep non-negative with a sum below 2^32: the uniform
; base and one 32-bit index register instead of 64-bit adds (L15-E303).

; CHECK-LABEL: subview:
; CHECK-NOT: iadd {{r[0-9]+}}_{{r[0-9]+}}
; CHECK: device_load 0, i32, x, {{r[0-9]+}}, u0_u1, {{r[0-9]+}}, unsigned
; CHECK: device_store 0, i32, x, {{r[0-9]+}}, u2_u3, {{r[0-9]+}}, unsigned
; OFF-LABEL: subview:
; OFF: iadd {{r[0-9]+}}_{{r[0-9]+}}
define agx_kernel void @subview(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree writeonly captures(none) %1) local_unnamed_addr #0 {
entry:
  %2 = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %3 = zext i32 %2 to i64
  %4 = tail call i32 @llvm.agx.threadgroup.position.in.grid.y()
  %5 = zext i32 %4 to i64
  %6 = tail call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %7 = zext i32 %6 to i64
  %8 = tail call i32 @llvm.agx.thread.position.in.threadgroup.y()
  %9 = zext i32 %8 to i64
  %10 = shl nuw nsw i64 %3, 4
  %11 = shl nuw nsw i64 %5, 14
  %12 = add nuw nsw i64 %11, %10
  %13 = getelementptr [4 x i8], ptr addrspace(1) %0, i64 %12
  %14 = shl nuw nsw i64 %9, 10
  %15 = add nuw nsw i64 %14, %7
  %16 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %13, i64 %15
  %17 = load float, ptr addrspace(1) %16, align 4
  %18 = tail call i64 @llvm.umin.i64(i64 %7, i64 14)
  %19 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %13, i64 %14
  %20 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %19, i64 %18
  %21 = getelementptr inbounds nuw i8, ptr addrspace(1) %20, i64 4
  %22 = load float, ptr addrspace(1) %21, align 4
  %23 = fadd float %17, %22
  %24 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %12
  %25 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %24, i64 %15
  store float %23, ptr addrspace(1) %25, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.x()

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.y()

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.x()

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.y()

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64)




attributes #0 = { "agx-max-threads-per-threadgroup"="256" "agx-threadgroups-per-grid"="64,64,1" "agx-threads-per-grid"="1024,1024,1" "agx-threads-per-threadgroup"="16,16,1" }
