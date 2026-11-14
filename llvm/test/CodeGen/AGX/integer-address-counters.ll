; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-integer-address-counters=0 < %s | FileCheck %s --check-prefix=OFF

; CSR y = A x (agx-bench spmv_csr, the PTX front end's integer addresses).
; The loop unrolled by 4 stepped two i64 addresses by 16 beside a counter
; from k - end (runtime, below zero) up to 0. Each address is one base and
; an unsigned index register, a counter of its own from 0 by 4 (j - j0, below
; 2^32 by the trip count's bound 2^30 - 1), and the exit test compares it
; with end - k, so that the old counter goes (L15-E261): one 32-bit add per
; iteration instead of two 64-bit ones and a 32-bit one.

; CHECK-LABEL: spmv_csr:
; CHECK:      device_load 0, i32, xyzw, {{.*}}, r{{[0-9]+}}_r{{[0-9]+}}, [[I:r[0-9]+]], unsigned
; CHECK-NEXT: device_load 1, i32, xyzw, {{.*}}, r{{[0-9]+}}_r{{[0-9]+}}, [[I]], unsigned
; CHECK-NEXT: iadd [[I]], [[I]], 4
; CHECK-NOT:  iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK:      while_icmp r0l, nueq, [[I]], r{{[0-9]+}}, 1
; OFF-LABEL:  spmv_csr:
; OFF:        device_load 0, i32, xyzw, {{.*}}, -2, signed
; OFF:        iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 16

target triple = "agx-apple-macosx"

; Function Attrs: nofree norecurse nosync nounwind denormal_fpenv(preservesign, float: ieee) memory(readwrite, inaccessiblemem: none, target_mem: none)
define agx_kernel void @spmv_csr(ptr addrspace(4) nofree readonly byref(i64) captures(none) %"322", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"323", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"324", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"325", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"326", ptr addrspace(4) nofree readonly byref(i32) captures(none) %"327") local_unnamed_addr #0 {
"318":
  %0 = load i32, ptr addrspace(4) %"327", align 4
  %vx.i41 = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %vx.i40 = tail call i32 @llvm.agx.threads.per.threadgroup.x()
  %vx.i = tail call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %1 = mul i32 %vx.i41, %vx.i40
  %"631" = add i32 %1, %vx.i
  %.not = icmp slt i32 %"631", %0
  br i1 %.not, label %"196", label %"83"

"196":                                            ; preds = %"318"
  %2 = load i64, ptr addrspace(4) %"322", align 8
  %3 = load i64, ptr addrspace(4) %"326", align 8
  %4 = inttoptr i64 %3 to ptr
  %5 = addrspacecast ptr %4 to ptr addrspace(1)
  %"453" = ptrtoint ptr addrspace(1) %5 to i64
  %6 = inttoptr i64 %2 to ptr
  %7 = addrspacecast ptr %6 to ptr addrspace(1)
  %"455" = ptrtoint ptr addrspace(1) %7 to i64
  %"639" = sext i32 %"631" to i64
  %"641" = shl nsw i64 %"639", 2
  %"643" = add i64 %"641", %"455"
  %"647" = inttoptr i64 %"643" to ptr addrspace(1)
  %8 = load i32, ptr addrspace(1) %"647", align 4
  %"281" = getelementptr inbounds nuw i8, ptr addrspace(1) %"647", i64 4
  %9 = load i32, ptr addrspace(1) %"281", align 4
  %.not38 = icmp slt i32 %8, %9
  br i1 %.not38, label %"198", label %"82"

"198":                                            ; preds = %"196"
  %10 = load i64, ptr addrspace(4) %"323", align 8
  %11 = load i64, ptr addrspace(4) %"325", align 8
  %12 = inttoptr i64 %11 to ptr
  %13 = addrspacecast ptr %12 to ptr addrspace(1)
  %"475" = ptrtoint ptr addrspace(1) %13 to i64
  %14 = load i64, ptr addrspace(4) %"324", align 8
  %15 = inttoptr i64 %14 to ptr
  %16 = addrspacecast ptr %15 to ptr addrspace(1)
  %"478" = ptrtoint ptr addrspace(1) %16 to i64
  %17 = inttoptr i64 %10 to ptr
  %18 = addrspacecast ptr %17 to ptr addrspace(1)
  %"480" = ptrtoint ptr addrspace(1) %18 to i64
  %"655" = sub i32 %9, %8
  %"485" = and i32 %"655", 3
  %19 = icmp eq i32 %"485", 0
  br i1 %19, label %"80", label %"200"

"200":                                            ; preds = %"198"
  %"661" = sext i32 %8 to i64
  %20 = shl nsw i64 %"661", 2
  %"663" = add i64 %20, %"478"
  %"666" = add i64 %20, %"480"
  br label %"79"

"79":                                             ; preds = %"79", %"200"
  %"359.0" = phi i32 [ %"485", %"200" ], [ %"686", %"79" ]
  %"391.0" = phi float [ 0.000000e+00, %"200" ], [ %"517", %"79" ]
  %"435.0" = phi i64 [ %"666", %"200" ], [ %"684", %"79" ]
  %"436.0" = phi i64 [ %"663", %"200" ], [ %"682", %"79" ]
  %"671" = inttoptr i64 %"436.0" to ptr addrspace(1)
  %21 = load float, ptr addrspace(1) %"671", align 4
  %"673" = inttoptr i64 %"435.0" to ptr addrspace(1)
  %22 = load i32, ptr addrspace(1) %"673", align 4
  %23 = sext i32 %22 to i64
  %"674" = shl nsw i64 %23, 2
  %"676" = add i64 %"674", %"475"
  %"679" = inttoptr i64 %"676" to ptr addrspace(1)
  %24 = load float, ptr addrspace(1) %"679", align 4
  %"517" = tail call float @llvm.fma.f32(float %21, float %24, float %"391.0")
  %"682" = add i64 %"436.0", 4
  %"684" = add i64 %"435.0", 4
  %"686" = add nsw i32 %"359.0", -1
  %.not39 = icmp eq i32 %"686", 0
  br i1 %.not39, label %"80.loopexit", label %"79"

"80.loopexit":                                    ; preds = %"79"
  %25 = add i32 %8, %"485"
  br label %"80"

"80":                                             ; preds = %"80.loopexit", %"198"
  %"361.1" = phi i32 [ %8, %"198" ], [ %25, %"80.loopexit" ]
  %"391.1" = phi float [ 0.000000e+00, %"198" ], [ %"517", %"80.loopexit" ]
  %"689" = sub i32 %8, %9
  %26 = icmp ugt i32 %"689", -4
  br i1 %26, label %"82", label %"204"

"204":                                            ; preds = %"80"
  %"693" = sub i32 %9, %"361.1"
  %27 = sext i32 %"361.1" to i64
  %"696" = shl nsw i64 %27, 2
  %"698" = add nsw i64 %"696", 8
  %"700" = add i64 %"698", %"480"
  %"703" = add i64 %"698", %"478"
  br label %"81"

"81":                                             ; preds = %"81", %"204"
  %"362.0" = phi i32 [ %"693", %"204" ], [ %"742", %"81" ]
  %"391.2" = phi float [ %"391.1", %"204" ], [ %"607", %"81" ]
  %"437.0" = phi i64 [ %"703", %"204" ], [ %"746", %"81" ]
  %"438.0" = phi i64 [ %"700", %"204" ], [ %"744", %"81" ]
  %"706" = inttoptr i64 %"437.0" to ptr addrspace(1)
  %"298" = getelementptr inbounds i8, ptr addrspace(1) %"706", i64 -8
  %28 = load float, ptr addrspace(1) %"298", align 4
  %"707" = inttoptr i64 %"438.0" to ptr addrspace(1)
  %"300" = getelementptr inbounds i8, ptr addrspace(1) %"707", i64 -8
  %29 = load i32, ptr addrspace(1) %"300", align 4
  %30 = sext i32 %29 to i64
  %"709" = shl nsw i64 %30, 2
  %"711" = add i64 %"709", %"475"
  %"714" = inttoptr i64 %"711" to ptr addrspace(1)
  %31 = load float, ptr addrspace(1) %"714", align 4
  %"562" = tail call float @llvm.fma.f32(float %28, float %31, float %"391.2")
  %"303" = getelementptr inbounds i8, ptr addrspace(1) %"706", i64 -4
  %32 = load float, ptr addrspace(1) %"303", align 4
  %"305" = getelementptr inbounds i8, ptr addrspace(1) %"707", i64 -4
  %33 = load i32, ptr addrspace(1) %"305", align 4
  %34 = sext i32 %33 to i64
  %"718" = shl nsw i64 %34, 2
  %"720" = add i64 %"718", %"475"
  %"723" = inttoptr i64 %"720" to ptr addrspace(1)
  %35 = load float, ptr addrspace(1) %"723", align 4
  %"577" = tail call float @llvm.fma.f32(float %32, float %35, float %"562")
  %36 = load float, ptr addrspace(1) %"706", align 4
  %37 = load i32, ptr addrspace(1) %"707", align 4
  %38 = sext i32 %37 to i64
  %"727" = shl nsw i64 %38, 2
  %"729" = add i64 %"727", %"475"
  %"732" = inttoptr i64 %"729" to ptr addrspace(1)
  %39 = load float, ptr addrspace(1) %"732", align 4
  %"592" = tail call float @llvm.fma.f32(float %36, float %39, float %"577")
  %"309" = getelementptr inbounds nuw i8, ptr addrspace(1) %"706", i64 4
  %40 = load float, ptr addrspace(1) %"309", align 4
  %"311" = getelementptr inbounds nuw i8, ptr addrspace(1) %"707", i64 4
  %41 = load i32, ptr addrspace(1) %"311", align 4
  %42 = sext i32 %41 to i64
  %"736" = shl nsw i64 %42, 2
  %"738" = add i64 %"736", %"475"
  %"741" = inttoptr i64 %"738" to ptr addrspace(1)
  %43 = load float, ptr addrspace(1) %"741", align 4
  %"607" = tail call float @llvm.fma.f32(float %40, float %43, float %"592")
  %"742" = add i32 %"362.0", -4
  %"744" = add i64 %"438.0", 16
  %"746" = add i64 %"437.0", 16
  %44 = icmp eq i32 %"742", 0
  br i1 %44, label %"82", label %"81"

"82":                                             ; preds = %"81", %"80", %"196"
  %"391.3" = phi float [ 0.000000e+00, %"196" ], [ %"391.1", %"80" ], [ %"607", %"81" ]
  %"749" = add i64 %"641", %"453"
  %"752" = inttoptr i64 %"749" to ptr addrspace(1)
  store float %"391.3", ptr addrspace(1) %"752", align 4
  br label %"83"

"83":                                             ; preds = %"82", %"318"
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threads.per.threadgroup.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.x() #2

attributes #0 = { nofree norecurse nosync nounwind denormal_fpenv(preservesign, float: ieee) memory(readwrite, inaccessiblemem: none, target_mem: none) "no-trapping-math"="true" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"agx-stack-size", i32 1024}
