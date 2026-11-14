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
