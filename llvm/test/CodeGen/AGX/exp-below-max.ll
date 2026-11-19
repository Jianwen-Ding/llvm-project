; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-exp-below-loop-max=0 -agx-exp-max-difference=0 < %s | FileCheck %s --check-prefix=OFF

; e^(x - m) where m is at least x: a softmax's exponent. The precise exp
; clamps its argument above 89 and selects +inf where the scaled result
; overflows; x - m is at most 0 (a denormal where max's comparison flushes)
; or a NaN, so both go (L15-E270). @xent: m an earlier loop's maximum of the
; values the loop with the exp loads again (agx-bench cross_entropy);
; @local: m = maxnum(x, y) itself; @simd: m a SIMD-group's fmax of x (agx-bench
; softmax_row).

; CHECK-LABEL: xent:
; CHECK:       // %l2
; CHECK-NOT:   icmpsel sgt
; CHECK:       jmp_exec_any
; OFF-LABEL:   xent:
; OFF:         // %l2
; OFF:         icmpsel sgt
; CHECK-LABEL: local:
; CHECK-NOT:   icmpsel sgt
; CHECK:       stop
; OFF-LABEL:   local:
; OFF:         icmpsel sgt
; CHECK-LABEL: simd:
; CHECK-NOT:   icmpsel sgt
; CHECK:       stop
; OFF-LABEL:   simd:
; OFF:         icmpsel sgt

target triple = "agx"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #0

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.log.f32(float) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @xent(ptr addrspace(1) noalias nofree readonly captures(none) %x, ptr addrspace(1) noalias nofree readonly captures(none) %lab, ptr addrspace(1) noalias nofree writeonly captures(none) %loss, i32 %n) local_unnamed_addr #2 {
entry:
  %r = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %n, %r
  %umax = tail call i32 @llvm.umax.i32(i32 %n, i32 1)
  br label %l1

l1:                                               ; preds = %l1, %entry
  %j = phi i32 [ 0, %entry ], [ %j1, %l1 ]
  %m = phi float [ -inf, %entry ], [ %m1, %l1 ]
  %ix = add i32 %j, %row
  %0 = sext i32 %ix to i64
  %p = getelementptr [4 x i8], ptr addrspace(1) %x, i64 %0
  %v = load float, ptr addrspace(1) %p, align 4
  %m1 = tail call float @llvm.maxnum.f32(float %m, float %v)
  %j1 = add nuw i32 %j, 1
  %exitcond.not = icmp eq i32 %j1, %umax
  br i1 %exitcond.not, label %l2, label %l1

l2:                                               ; preds = %l1, %l2
  %k = phi i32 [ %k1, %l2 ], [ 0, %l1 ]
  %s = phi float [ %s1, %l2 ], [ 0.000000e+00, %l1 ]
  %ix2 = add i32 %k, %row
  %1 = sext i32 %ix2 to i64
  %p2 = getelementptr [4 x i8], ptr addrspace(1) %x, i64 %1
  %v2 = load float, ptr addrspace(1) %p2, align 4
  %d = fsub float %v2, %m1
  %e = tail call float @llvm.exp.f32(float %d)
  %s1 = fadd float %s, %e
  %k1 = add nuw i32 %k, 1
  %exitcond1.not = icmp eq i32 %k1, %umax
  br i1 %exitcond1.not, label %end, label %l2

end:                                              ; preds = %l2
  %lg = tail call float @llvm.log.f32(float %s1)
  %2 = sext i32 %r to i64
  %pl = getelementptr [4 x i8], ptr addrspace(1) %lab, i64 %2
  %l = load i32, ptr addrspace(1) %pl, align 4
  %il = add i32 %l, %row
  %3 = sext i32 %il to i64
  %pxl = getelementptr [4 x i8], ptr addrspace(1) %x, i64 %3
  %xl = load float, ptr addrspace(1) %pxl, align 4
  %t = fadd float %m1, %lg
  %out = fsub float %t, %xl
  %po = getelementptr [4 x i8], ptr addrspace(1) %loss, i64 %2
  store float %out, ptr addrspace(1) %po, align 4
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umax.i32(i32, i32) #3

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="agx-g14c" }
attributes #2 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="agx-g14c" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
define agx_kernel void @local(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %i
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %i
  %po = getelementptr float, ptr addrspace(1) %o, i32 %i
  %x = load float, ptr addrspace(1) %pa
  %y = load float, ptr addrspace(1) %pb
  %m = call float @llvm.maxnum.f32(float %x, float %y)
  %d = fsub float %x, %m
  %e = call float @llvm.exp.f32(float %d)
  store float %e, ptr addrspace(1) %po
  ret void
}

declare float @llvm.agx.simd.reduce.f32(float, i32)
define agx_kernel void @simd(ptr addrspace(1) %x) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %m = call float @llvm.agx.simd.reduce.f32(float %v, i32 7)
  %d = fsub float %v, %m
  %e = call float @llvm.exp.f32(float %d)
  store float %e, ptr addrspace(1) %p, align 4
  ret void
}

; @xent_rows: the MLIR bench's xent_rows, m the SIMD-group's maximum of the lanes'
; loop maxima (llvm.maximum of frozen loads) as the signed maximum of their
; order-preserving integer encoding, a NaN as 0x7fc00000 (L15-E450).
; CHECK-LABEL: xent_rows:
; CHECK:       simd_max.s
; CHECK:       Loop Header
; CHECK-NOT:   icmpsel sgt
; CHECK:       jmp_exec_any
; OFF-LABEL:   xent_rows:
; OFF:         simd_max.s
; OFF:         Loop Header
; OFF:         icmpsel sgt
declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare i64 @llvm.umin.i64(i64, i64)
declare float @llvm.maximum.f32(float, float)
declare i32 @llvm.agx.simd.reduce.i32(i32, i32)
define agx_kernel void @xent_rows(ptr addrspace(1) noalias readonly %0, ptr addrspace(1) noalias readonly %1, ptr addrspace(1) noalias writeonly %2) #2 {
entry:
  %3 = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %4 = zext i32 %3 to i64
  %5 = tail call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %6 = zext i32 %5 to i64
  %7 = icmp ult i32 %5, 1000
  %8 = sub nsw i64 1000, %6
  %9 = tail call i64 @llvm.umin.i64(i64 %8, i64 1000)
  %10 = trunc nuw nsw i64 %9 to i32
  %11 = select i1 %7, i32 %10, i32 0
  %.not = icmp eq i32 %11, 0
  br i1 %.not, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %entry
  %12 = mul i32 %3, 1000
  %invariant.op = add i32 %12, %5
  %13 = zext nneg i32 %11 to i64
  br label %14

14:                                               ; preds = %.lr.ph, %14
  %indvars.iv = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next, %14 ]
  %15 = phi float [ -inf, %.lr.ph ], [ %.nan, %14 ]
  %16 = trunc nuw nsw i64 %indvars.iv to i32
  %.reass = add i32 %invariant.op, %16
  %17 = zext i32 %.reass to i64
  %18 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %17
  %19 = load float, ptr addrspace(1) %18, align 4
  %.fr = freeze float %19
  %.nan = tail call float @llvm.maximum.f32(float %15, float %.fr)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 32
  %20 = icmp samesign ult i64 %indvars.iv.next, %13
  br i1 %20, label %14, label %._crit_edge

._crit_edge:                                      ; preds = %14, %entry
  %.lcssa4 = phi float [ -inf, %entry ], [ %.nan, %14 ]
  %21 = fcmp uno float %.lcssa4, 0.000000e+00
  %22 = bitcast float %.lcssa4 to i32
  %isneg = icmp slt i32 %22, 0
  %23 = select i1 %isneg, i32 2147483647, i32 0
  %24 = xor i32 %23, %22
  %25 = select i1 %21, i32 2143289344, i32 %24
  %26 = tail call i32 @llvm.agx.simd.reduce.i32(i32 %25, i32 22)
  %isneg3 = icmp slt i32 %26, 0
  %27 = select i1 %isneg3, i32 2147483647, i32 0
  %28 = xor i32 %27, %26
  %29 = bitcast i32 %28 to float
  br i1 %.not, label %._crit_edge7, label %.lr.ph6

.lr.ph6:                                          ; preds = %._crit_edge
  %30 = mul i32 %3, 1000
  %invariant.op9 = add i32 %30, %5
  %31 = zext nneg i32 %11 to i64
  br label %32

32:                                               ; preds = %.lr.ph6, %32
  %indvars.iv13 = phi i64 [ 0, %.lr.ph6 ], [ %indvars.iv.next14, %32 ]
  %33 = phi float [ 0.000000e+00, %.lr.ph6 ], [ %40, %32 ]
  %34 = trunc nuw nsw i64 %indvars.iv13 to i32
  %.reass10 = add i32 %invariant.op9, %34
  %35 = zext i32 %.reass10 to i64
  %36 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %35
  %37 = load float, ptr addrspace(1) %36, align 4
  %38 = fsub float %37, %29
  %39 = tail call float @llvm.exp.f32(float %38)
  %40 = fadd float %33, %39
  %indvars.iv.next14 = add nuw nsw i64 %indvars.iv13, 32
  %41 = icmp samesign ult i64 %indvars.iv.next14, %31
  br i1 %41, label %32, label %._crit_edge7

._crit_edge7:                                     ; preds = %32, %._crit_edge
  %.lcssa = phi float [ 0.000000e+00, %._crit_edge ], [ %40, %32 ]
  %42 = tail call float @llvm.agx.simd.reduce.f32(float %.lcssa, i32 1)
  %43 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %4
  store float %42, ptr addrspace(1) %43, align 4
  ret void
}
