; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll-full' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll-full' -agx-tile-full-unroll-threshold=0 -S < %s | FileCheck %s --check-prefix=OFF

; The k loop of a register-blocked matrix product over a 16-deep threadgroup
; tile (4x4 outputs per thread: 8 threadgroup loads, 16 fmas per iteration)
; is fully unrolled: above the generic full-unroll threshold, below the AGX
; one for short loops over threadgroup tiles in functions with registers to
; spare.

@As = internal addrspace(3) global [16 x [64 x float]] poison, align 4
@Bs = internal addrspace(3) global [16 x [64 x float]] poison, align 4

define void @tile(ptr addrspace(1) %out, i32 %tx, i32 %ty) {
; CHECK-LABEL: @tile(
; CHECK-NOT: br i1
; CHECK-COUNT-256: call float @llvm.fma.f32
; CHECK-NOT: br i1
; CHECK: ret void
; OFF-LABEL: @tile(
; OFF-COUNT-16: call float @llvm.fma.f32
; OFF-NOT: call float @llvm.fma.f32
; OFF: br i1
entry:
  %ai0 = add i32 %ty, 0
  %bi0 = add i32 %tx, 0
  %ai1 = add i32 %ty, 16
  %bi1 = add i32 %tx, 16
  %ai2 = add i32 %ty, 32
  %bi2 = add i32 %tx, 32
  %ai3 = add i32 %ty, 48
  %bi3 = add i32 %tx, 48
  br label %loop
loop:
  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]
  %c0 = phi float [ 0.0, %entry ], [ %n0, %loop ]
  %c1 = phi float [ 0.0, %entry ], [ %n1, %loop ]
  %c2 = phi float [ 0.0, %entry ], [ %n2, %loop ]
  %c3 = phi float [ 0.0, %entry ], [ %n3, %loop ]
  %c4 = phi float [ 0.0, %entry ], [ %n4, %loop ]
  %c5 = phi float [ 0.0, %entry ], [ %n5, %loop ]
  %c6 = phi float [ 0.0, %entry ], [ %n6, %loop ]
  %c7 = phi float [ 0.0, %entry ], [ %n7, %loop ]
  %c8 = phi float [ 0.0, %entry ], [ %n8, %loop ]
  %c9 = phi float [ 0.0, %entry ], [ %n9, %loop ]
  %c10 = phi float [ 0.0, %entry ], [ %n10, %loop ]
  %c11 = phi float [ 0.0, %entry ], [ %n11, %loop ]
  %c12 = phi float [ 0.0, %entry ], [ %n12, %loop ]
  %c13 = phi float [ 0.0, %entry ], [ %n13, %loop ]
  %c14 = phi float [ 0.0, %entry ], [ %n14, %loop ]
  %c15 = phi float [ 0.0, %entry ], [ %n15, %loop ]
  %pa0 = getelementptr [16 x [64 x float]], ptr addrspace(3) @As, i32 0, i32 %k, i32 %ai0
  %a0 = load float, ptr addrspace(3) %pa0, align 4
  %pb0 = getelementptr [16 x [64 x float]], ptr addrspace(3) @Bs, i32 0, i32 %k, i32 %bi0
  %b0 = load float, ptr addrspace(3) %pb0, align 4
  %pa1 = getelementptr [16 x [64 x float]], ptr addrspace(3) @As, i32 0, i32 %k, i32 %ai1
  %a1 = load float, ptr addrspace(3) %pa1, align 4
  %pb1 = getelementptr [16 x [64 x float]], ptr addrspace(3) @Bs, i32 0, i32 %k, i32 %bi1
  %b1 = load float, ptr addrspace(3) %pb1, align 4
  %pa2 = getelementptr [16 x [64 x float]], ptr addrspace(3) @As, i32 0, i32 %k, i32 %ai2
  %a2 = load float, ptr addrspace(3) %pa2, align 4
  %pb2 = getelementptr [16 x [64 x float]], ptr addrspace(3) @Bs, i32 0, i32 %k, i32 %bi2
  %b2 = load float, ptr addrspace(3) %pb2, align 4
  %pa3 = getelementptr [16 x [64 x float]], ptr addrspace(3) @As, i32 0, i32 %k, i32 %ai3
  %a3 = load float, ptr addrspace(3) %pa3, align 4
  %pb3 = getelementptr [16 x [64 x float]], ptr addrspace(3) @Bs, i32 0, i32 %k, i32 %bi3
  %b3 = load float, ptr addrspace(3) %pb3, align 4
  %n0 = call float @llvm.fma.f32(float %a0, float %b0, float %c0)
  %n1 = call float @llvm.fma.f32(float %a0, float %b1, float %c1)
  %n2 = call float @llvm.fma.f32(float %a0, float %b2, float %c2)
  %n3 = call float @llvm.fma.f32(float %a0, float %b3, float %c3)
  %n4 = call float @llvm.fma.f32(float %a1, float %b0, float %c4)
  %n5 = call float @llvm.fma.f32(float %a1, float %b1, float %c5)
  %n6 = call float @llvm.fma.f32(float %a1, float %b2, float %c6)
  %n7 = call float @llvm.fma.f32(float %a1, float %b3, float %c7)
  %n8 = call float @llvm.fma.f32(float %a2, float %b0, float %c8)
  %n9 = call float @llvm.fma.f32(float %a2, float %b1, float %c9)
  %n10 = call float @llvm.fma.f32(float %a2, float %b2, float %c10)
  %n11 = call float @llvm.fma.f32(float %a2, float %b3, float %c11)
  %n12 = call float @llvm.fma.f32(float %a3, float %b0, float %c12)
  %n13 = call float @llvm.fma.f32(float %a3, float %b1, float %c13)
  %n14 = call float @llvm.fma.f32(float %a3, float %b2, float %c14)
  %n15 = call float @llvm.fma.f32(float %a3, float %b3, float %c15)
  %k1 = add nuw nsw i32 %k, 1
  %c = icmp ult i32 %k1, 16
  br i1 %c, label %loop, label %exit
exit:
  %o0 = getelementptr float, ptr addrspace(1) %out, i32 0
  store float %n0, ptr addrspace(1) %o0, align 4
  %o1 = getelementptr float, ptr addrspace(1) %out, i32 1
  store float %n1, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr float, ptr addrspace(1) %out, i32 2
  store float %n2, ptr addrspace(1) %o2, align 4
  %o3 = getelementptr float, ptr addrspace(1) %out, i32 3
  store float %n3, ptr addrspace(1) %o3, align 4
  %o4 = getelementptr float, ptr addrspace(1) %out, i32 4
  store float %n4, ptr addrspace(1) %o4, align 4
  %o5 = getelementptr float, ptr addrspace(1) %out, i32 5
  store float %n5, ptr addrspace(1) %o5, align 4
  %o6 = getelementptr float, ptr addrspace(1) %out, i32 6
  store float %n6, ptr addrspace(1) %o6, align 4
  %o7 = getelementptr float, ptr addrspace(1) %out, i32 7
  store float %n7, ptr addrspace(1) %o7, align 4
  %o8 = getelementptr float, ptr addrspace(1) %out, i32 8
  store float %n8, ptr addrspace(1) %o8, align 4
  %o9 = getelementptr float, ptr addrspace(1) %out, i32 9
  store float %n9, ptr addrspace(1) %o9, align 4
  %o10 = getelementptr float, ptr addrspace(1) %out, i32 10
  store float %n10, ptr addrspace(1) %o10, align 4
  %o11 = getelementptr float, ptr addrspace(1) %out, i32 11
  store float %n11, ptr addrspace(1) %o11, align 4
  %o12 = getelementptr float, ptr addrspace(1) %out, i32 12
  store float %n12, ptr addrspace(1) %o12, align 4
  %o13 = getelementptr float, ptr addrspace(1) %out, i32 13
  store float %n13, ptr addrspace(1) %o13, align 4
  %o14 = getelementptr float, ptr addrspace(1) %out, i32 14
  store float %n14, ptr addrspace(1) %o14, align 4
  %o15 = getelementptr float, ptr addrspace(1) %out, i32 15
  store float %n15, ptr addrspace(1) %o15, align 4
  ret void
}

declare float @llvm.fma.f32(float, float, float)
