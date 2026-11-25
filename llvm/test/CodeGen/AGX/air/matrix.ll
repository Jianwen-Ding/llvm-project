; AIR's SIMD-group matrices become AGX's fragments: each thread holds the
; elements (r, c) and (r, c + 1), r = 4 * (t / 16) + (t / 2) % 4 and
; c = 4 * ((t / 8) % 2) + 2 * (t % 2). Loads and stores move those two
; elements at (c + j + origin.x) * stride.x + (r + origin.y) * stride.y (as
; one <2 x T> access where stride.x is 1, L15-E856; else one each),
; multiply-accumulate is llvm.agx.simdgroup.matrix.fma (f16 operands of an
; f32 one extended first), and element-wise operations, AIR's element-wise
; functions, element-wise intrinsics, casts, phis and comparisons with the
; selects by them (L15-E855) apply to the fragment.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o /dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %[[T:.*]] = call i32 @llvm.agx.thread.index.in.simdgroup()
; CHECK: %matrix.row = zext i32 %{{.*}} to i64
; CHECK: %matrix.col = zext i32 %{{.*}} to i64
; CHECK: loop:
; CHECK: %acc = phi <2 x float> [ zeroinitializer, %entry ], [ %next, %loop ]
; CHECK: %ma = load <2 x float>, ptr addrspace(1) %{{.*}}, align 4
; CHECK: load float, ptr addrspace(1)
; CHECK: %next = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %acc)
; CHECK: done:
; CHECK: %scaled = fmul <2 x float> %next, splat (float 2.000000e+00)
; CHECK: %fused = call <2 x float> @llvm.fma.v2f32(<2 x float> %scaled, <2 x float> %scaled, <2 x float> %next)
; CHECK: store <2 x float> %fused, ptr addrspace(1) %{{.*}}, align 4
; CHECK-NOT: <64 x float>

; CHECK-LABEL: define agx_kernel void @mixed(
; CHECK: %ha = load <2 x half>, ptr addrspace(1) %{{.*}}, align 2
; CHECK: %hc32 = fpext <2 x half> %hc to <2 x float>
; CHECK: %[[EA:.*]] = fpext <2 x half> %ha to <2 x float>
; CHECK: %[[EB:.*]] = fpext <2 x half> %hb to <2 x float>
; CHECK: %d = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %[[EA]], <2 x float> %[[EB]], <2 x float> %hc32)
; CHECK: %m = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %d, <2 x float> zeroinitializer)
; CHECK: %h = fptrunc <2 x float> %m to <2 x half>
; CHECK: store <2 x half> %h
; CHECK-NOT: <64 x

; CHECK-LABEL: define agx_kernel void @compare(
; CHECK: %gt = fcmp ogt <2 x float> %{{.*}}, %{{.*}}
; CHECK: %pick = select <2 x i1> %gt, <2 x float> %{{.*}}, <2 x float> %{{.*}}
; CHECK: %nan = fcmp uno <2 x float>
; CHECK: %max = select <2 x i1> %nan, <2 x float> splat (float +qnan), <2 x float> %pick
; CHECK-NOT: <64 x

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float)
declare <64 x float> @air.fma.v64f32(<64 x float>, <64 x float>, <64 x float>)

define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
entry:
  %zero = call <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float 0.0)
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi <64 x float> [ %zero, %entry ], [ %next, %loop ]
  %ma = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %a, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mb = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %b, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 8, i64 1>, <2 x i64> zeroinitializer)
  %next = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %ma, <64 x float> %mb, <64 x float> %acc)
  %i.next = add i32 %i, 1
  %last = icmp eq i32 %i.next, 4
  br i1 %last, label %done, label %loop
done:
  %scaled = fmul <64 x float> %next, splat (float 2.0)
  %fused = call <64 x float> @air.fma.v64f32(<64 x float> %scaled, <64 x float> %scaled, <64 x float> %next)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %fused, ptr addrspace(1) %c, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  ret void
}

declare <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.simdgroup_matrix_8x8_store.v64f16.p1f16(<64 x half>, ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half>, <64 x half>, <64 x float>)

define void @mixed(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
  %ha = call <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) %a, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %hb = call <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) %b, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %hc = call <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) %c, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %hc32 = fpext <64 x half> %hc to <64 x float>
  %d = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(<64 x half> %ha, <64 x half> %hb, <64 x float> %hc32)
  %m = call <64 x float> @llvm.maxnum.v64f32(<64 x float> %d, <64 x float> zeroinitializer)
  %h = fptrunc <64 x float> %m to <64 x half>
  call void @air.simdgroup_matrix_8x8_store.v64f16.p1f16(<64 x half> %h, ptr addrspace(1) %c, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  ret void
}

define void @compare(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
  %x = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %a, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %y = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %b, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %gt = fcmp ogt <64 x float> %x, %y
  %pick = select <64 x i1> %gt, <64 x float> %x, <64 x float> %y
  %nan = fcmp uno <64 x float> %x, %y
  %max = select <64 x i1> %nan, <64 x float> splat (float 0x7FF8000000000000), <64 x float> %pick
  call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %max, ptr addrspace(1) %c, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  ret void
}

!air.kernel = !{!0, !4, !5}
!0 = !{ptr @k, !{}, !{!1, !2, !3}}
!4 = !{ptr @mixed, !{}, !{!1, !2, !3}}
!5 = !{ptr @compare, !{}, !{!1, !2, !3}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1}
