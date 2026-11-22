; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-keep-occupancy-need=0 -agx-sched-loads-first-min-groups=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-keep-occupancy-need=0 -agx-sched-original-order-loads-first=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-keep-occupancy-need=0 < %s | FileCheck %s --check-prefix=DEF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=NEED
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s --check-prefix=UNIF
;
; MLIR's attention decode (mlir/utils/agx-kernels/bench/attn_decode_f16):
; its loop's free schedule needs registers that cut its resident groups, so
; the region keeps its original order (L15-E239), but with its ready device
; loads first (no barrier or threadgroup access in it), whose registers still
; let its threadgroups of 32 in: the keys' and values' loads ahead of each
; key's work instead of waited for one by one (L15-E498). (75 and 51 since
; L15-E513's shared fma, 77 and 53 before.) Since L15-E530 that order is
; dropped where it leaves the core fewer than 28 SIMD-groups: this loop's 26
; (40 in the plain order) hid less latency than they saved. (OFF 50 since
; L15-E681's selects on b - a.)
;
; Since L15-E753 (-agx-sched-keep-occupancy-need) the per-key loop of 64 loads
; and no stores keeps its free order, its 40 -> 28 groups a third and its
; threadgroups of 32 still in (the NEED order 2.8% slower in agx_cost here,
; with the constants in registers); with them in uniforms, as agx-bench and
; agx-build-fatbin.py build it, the same 51 registers either way (UNIF).

; CHECK: general purpose registers: 75 x 32-bit
; OFF: general purpose registers: 50 x 32-bit
; DEF: general purpose registers: 52 x 32-bit
; NEED: general purpose registers: 68 x 32-bit
; UNIF: general purpose registers: 51 x 32-bit

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @attn_decode_f16(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree readonly captures(none) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #0 {
entry:
  %4 = tail call i32 @llvm.agx.thread.index.in.simdgroup()
  %5 = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %6 = zext i32 %5 to i64
  %7 = zext i32 %4 to i64
  %8 = shl nuw nsw i64 %7, 2
  %9 = shl nuw nsw i64 %6, 7
  %10 = add nuw nsw i64 %9, %8
  %11 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %10
  %12 = load <4 x half>, ptr addrspace(1) %11, align 2
  %13 = fpext <4 x half> %12 to <4 x float>
  %14 = shl nuw nsw i64 %6, 16
  %invariant.op = add nuw nsw i64 %14, %8
  br label %15

15:                                               ; preds = %15, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.7, %15 ]
  %16 = phi <4 x float> [ zeroinitializer, %entry ], [ %242, %15 ]
  %17 = phi float [ 0.000000e+00, %entry ], [ %233, %15 ]
  %18 = phi float [ -inf, %entry ], [ %.nan.7, %15 ]
  %19 = shl nuw nsw i64 %indvars.iv, 7
  %.reass = add nuw nsw i64 %invariant.op, %19
  %20 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass
  %21 = load <4 x half>, ptr addrspace(1) %20, align 2
  %22 = fpext <4 x half> %21 to <4 x float>
  %23 = fmul <4 x float> %13, %22
  %24 = extractelement <4 x float> %23, i64 0
  %25 = extractelement <4 x float> %23, i64 1
  %26 = fadd float %24, %25
  %27 = extractelement <4 x float> %23, i64 2
  %28 = fadd float %27, %26
  %29 = extractelement <4 x float> %23, i64 3
  %30 = fadd float %29, %28
  %31 = tail call float @llvm.agx.simd.reduce.f32(float %30, i32 1)
  %32 = fmul float %31, f0x3DB504F3
  %.nan = tail call float @llvm.maximum.f32(float %18, float %32)
  %33 = fsub float %18, %.nan
  %34 = tail call float @llvm.exp.f32(float %33)
  %35 = fsub float %32, %.nan
  %36 = tail call float @llvm.exp.f32(float %35)
  %37 = tail call float @llvm.fma.f32(float %17, float %34, float %36)
  %38 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass
  %39 = load <4 x half>, ptr addrspace(1) %38, align 2
  %40 = fpext <4 x half> %39 to <4 x float>
  %41 = insertelement <4 x float> poison, float %34, i64 0
  %42 = shufflevector <4 x float> %41, <4 x float> poison, <4 x i32> zeroinitializer
  %43 = insertelement <4 x float> poison, float %36, i64 0
  %44 = shufflevector <4 x float> %43, <4 x float> poison, <4 x i32> zeroinitializer
  %45 = fmul <4 x float> %44, %40
  %46 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %16, <4 x float> %42, <4 x float> %45)
  %indvars.iv.next = shl i64 %indvars.iv, 7
  %47 = or disjoint i64 %indvars.iv.next, 128
  %.reass.1 = add nuw nsw i64 %invariant.op, %47
  %48 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass.1
  %49 = load <4 x half>, ptr addrspace(1) %48, align 2
  %50 = fpext <4 x half> %49 to <4 x float>
  %51 = fmul <4 x float> %13, %50
  %52 = extractelement <4 x float> %51, i64 0
  %53 = extractelement <4 x float> %51, i64 1
  %54 = fadd float %52, %53
  %55 = extractelement <4 x float> %51, i64 2
  %56 = fadd float %55, %54
  %57 = extractelement <4 x float> %51, i64 3
  %58 = fadd float %57, %56
  %59 = tail call float @llvm.agx.simd.reduce.f32(float %58, i32 1)
  %60 = fmul float %59, f0x3DB504F3
  %.nan.1 = tail call float @llvm.maximum.f32(float %.nan, float %60)
  %61 = fsub float %.nan, %.nan.1
  %62 = tail call float @llvm.exp.f32(float %61)
  %63 = fsub float %60, %.nan.1
  %64 = tail call float @llvm.exp.f32(float %63)
  %65 = tail call float @llvm.fma.f32(float %37, float %62, float %64)
  %66 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass.1
  %67 = load <4 x half>, ptr addrspace(1) %66, align 2
  %68 = fpext <4 x half> %67 to <4 x float>
  %69 = insertelement <4 x float> poison, float %62, i64 0
  %70 = shufflevector <4 x float> %69, <4 x float> poison, <4 x i32> zeroinitializer
  %71 = insertelement <4 x float> poison, float %64, i64 0
  %72 = shufflevector <4 x float> %71, <4 x float> poison, <4 x i32> zeroinitializer
  %73 = fmul <4 x float> %72, %68
  %74 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %46, <4 x float> %70, <4 x float> %73)
  %indvars.iv.next.1 = shl i64 %indvars.iv, 7
  %75 = or disjoint i64 %indvars.iv.next.1, 256
  %.reass.2 = add nuw nsw i64 %invariant.op, %75
  %76 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass.2
  %77 = load <4 x half>, ptr addrspace(1) %76, align 2
  %78 = fpext <4 x half> %77 to <4 x float>
  %79 = fmul <4 x float> %13, %78
  %80 = extractelement <4 x float> %79, i64 0
  %81 = extractelement <4 x float> %79, i64 1
  %82 = fadd float %80, %81
  %83 = extractelement <4 x float> %79, i64 2
  %84 = fadd float %83, %82
  %85 = extractelement <4 x float> %79, i64 3
  %86 = fadd float %85, %84
  %87 = tail call float @llvm.agx.simd.reduce.f32(float %86, i32 1)
  %88 = fmul float %87, f0x3DB504F3
  %.nan.2 = tail call float @llvm.maximum.f32(float %.nan.1, float %88)
  %89 = fsub float %.nan.1, %.nan.2
  %90 = tail call float @llvm.exp.f32(float %89)
  %91 = fsub float %88, %.nan.2
  %92 = tail call float @llvm.exp.f32(float %91)
  %93 = tail call float @llvm.fma.f32(float %65, float %90, float %92)
  %94 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass.2
  %95 = load <4 x half>, ptr addrspace(1) %94, align 2
  %96 = fpext <4 x half> %95 to <4 x float>
  %97 = insertelement <4 x float> poison, float %90, i64 0
  %98 = shufflevector <4 x float> %97, <4 x float> poison, <4 x i32> zeroinitializer
  %99 = insertelement <4 x float> poison, float %92, i64 0
  %100 = shufflevector <4 x float> %99, <4 x float> poison, <4 x i32> zeroinitializer
  %101 = fmul <4 x float> %100, %96
  %102 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %74, <4 x float> %98, <4 x float> %101)
  %indvars.iv.next.2 = shl i64 %indvars.iv, 7
  %103 = or disjoint i64 %indvars.iv.next.2, 384
  %.reass.3 = add nuw nsw i64 %invariant.op, %103
  %104 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass.3
  %105 = load <4 x half>, ptr addrspace(1) %104, align 2
  %106 = fpext <4 x half> %105 to <4 x float>
  %107 = fmul <4 x float> %13, %106
  %108 = extractelement <4 x float> %107, i64 0
  %109 = extractelement <4 x float> %107, i64 1
  %110 = fadd float %108, %109
  %111 = extractelement <4 x float> %107, i64 2
  %112 = fadd float %111, %110
  %113 = extractelement <4 x float> %107, i64 3
  %114 = fadd float %113, %112
  %115 = tail call float @llvm.agx.simd.reduce.f32(float %114, i32 1)
  %116 = fmul float %115, f0x3DB504F3
  %.nan.3 = tail call float @llvm.maximum.f32(float %.nan.2, float %116)
  %117 = fsub float %.nan.2, %.nan.3
  %118 = tail call float @llvm.exp.f32(float %117)
  %119 = fsub float %116, %.nan.3
  %120 = tail call float @llvm.exp.f32(float %119)
  %121 = tail call float @llvm.fma.f32(float %93, float %118, float %120)
  %122 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass.3
  %123 = load <4 x half>, ptr addrspace(1) %122, align 2
  %124 = fpext <4 x half> %123 to <4 x float>
  %125 = insertelement <4 x float> poison, float %118, i64 0
  %126 = shufflevector <4 x float> %125, <4 x float> poison, <4 x i32> zeroinitializer
  %127 = insertelement <4 x float> poison, float %120, i64 0
  %128 = shufflevector <4 x float> %127, <4 x float> poison, <4 x i32> zeroinitializer
  %129 = fmul <4 x float> %128, %124
  %130 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %102, <4 x float> %126, <4 x float> %129)
  %indvars.iv.next.3 = shl i64 %indvars.iv, 7
  %131 = or disjoint i64 %indvars.iv.next.3, 512
  %.reass.4 = add nuw nsw i64 %invariant.op, %131
  %132 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass.4
  %133 = load <4 x half>, ptr addrspace(1) %132, align 2
  %134 = fpext <4 x half> %133 to <4 x float>
  %135 = fmul <4 x float> %13, %134
  %136 = extractelement <4 x float> %135, i64 0
  %137 = extractelement <4 x float> %135, i64 1
  %138 = fadd float %136, %137
  %139 = extractelement <4 x float> %135, i64 2
  %140 = fadd float %139, %138
  %141 = extractelement <4 x float> %135, i64 3
  %142 = fadd float %141, %140
  %143 = tail call float @llvm.agx.simd.reduce.f32(float %142, i32 1)
  %144 = fmul float %143, f0x3DB504F3
  %.nan.4 = tail call float @llvm.maximum.f32(float %.nan.3, float %144)
  %145 = fsub float %.nan.3, %.nan.4
  %146 = tail call float @llvm.exp.f32(float %145)
  %147 = fsub float %144, %.nan.4
  %148 = tail call float @llvm.exp.f32(float %147)
  %149 = tail call float @llvm.fma.f32(float %121, float %146, float %148)
  %150 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass.4
  %151 = load <4 x half>, ptr addrspace(1) %150, align 2
  %152 = fpext <4 x half> %151 to <4 x float>
  %153 = insertelement <4 x float> poison, float %146, i64 0
  %154 = shufflevector <4 x float> %153, <4 x float> poison, <4 x i32> zeroinitializer
  %155 = insertelement <4 x float> poison, float %148, i64 0
  %156 = shufflevector <4 x float> %155, <4 x float> poison, <4 x i32> zeroinitializer
  %157 = fmul <4 x float> %156, %152
  %158 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %130, <4 x float> %154, <4 x float> %157)
  %indvars.iv.next.4 = shl i64 %indvars.iv, 7
  %159 = or disjoint i64 %indvars.iv.next.4, 640
  %.reass.5 = add nuw nsw i64 %invariant.op, %159
  %160 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass.5
  %161 = load <4 x half>, ptr addrspace(1) %160, align 2
  %162 = fpext <4 x half> %161 to <4 x float>
  %163 = fmul <4 x float> %13, %162
  %164 = extractelement <4 x float> %163, i64 0
  %165 = extractelement <4 x float> %163, i64 1
  %166 = fadd float %164, %165
  %167 = extractelement <4 x float> %163, i64 2
  %168 = fadd float %167, %166
  %169 = extractelement <4 x float> %163, i64 3
  %170 = fadd float %169, %168
  %171 = tail call float @llvm.agx.simd.reduce.f32(float %170, i32 1)
  %172 = fmul float %171, f0x3DB504F3
  %.nan.5 = tail call float @llvm.maximum.f32(float %.nan.4, float %172)
  %173 = fsub float %.nan.4, %.nan.5
  %174 = tail call float @llvm.exp.f32(float %173)
  %175 = fsub float %172, %.nan.5
  %176 = tail call float @llvm.exp.f32(float %175)
  %177 = tail call float @llvm.fma.f32(float %149, float %174, float %176)
  %178 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass.5
  %179 = load <4 x half>, ptr addrspace(1) %178, align 2
  %180 = fpext <4 x half> %179 to <4 x float>
  %181 = insertelement <4 x float> poison, float %174, i64 0
  %182 = shufflevector <4 x float> %181, <4 x float> poison, <4 x i32> zeroinitializer
  %183 = insertelement <4 x float> poison, float %176, i64 0
  %184 = shufflevector <4 x float> %183, <4 x float> poison, <4 x i32> zeroinitializer
  %185 = fmul <4 x float> %184, %180
  %186 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %158, <4 x float> %182, <4 x float> %185)
  %indvars.iv.next.5 = shl i64 %indvars.iv, 7
  %187 = or disjoint i64 %indvars.iv.next.5, 768
  %.reass.6 = add nuw nsw i64 %invariant.op, %187
  %188 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass.6
  %189 = load <4 x half>, ptr addrspace(1) %188, align 2
  %190 = fpext <4 x half> %189 to <4 x float>
  %191 = fmul <4 x float> %13, %190
  %192 = extractelement <4 x float> %191, i64 0
  %193 = extractelement <4 x float> %191, i64 1
  %194 = fadd float %192, %193
  %195 = extractelement <4 x float> %191, i64 2
  %196 = fadd float %195, %194
  %197 = extractelement <4 x float> %191, i64 3
  %198 = fadd float %197, %196
  %199 = tail call float @llvm.agx.simd.reduce.f32(float %198, i32 1)
  %200 = fmul float %199, f0x3DB504F3
  %.nan.6 = tail call float @llvm.maximum.f32(float %.nan.5, float %200)
  %201 = fsub float %.nan.5, %.nan.6
  %202 = tail call float @llvm.exp.f32(float %201)
  %203 = fsub float %200, %.nan.6
  %204 = tail call float @llvm.exp.f32(float %203)
  %205 = tail call float @llvm.fma.f32(float %177, float %202, float %204)
  %206 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass.6
  %207 = load <4 x half>, ptr addrspace(1) %206, align 2
  %208 = fpext <4 x half> %207 to <4 x float>
  %209 = insertelement <4 x float> poison, float %202, i64 0
  %210 = shufflevector <4 x float> %209, <4 x float> poison, <4 x i32> zeroinitializer
  %211 = insertelement <4 x float> poison, float %204, i64 0
  %212 = shufflevector <4 x float> %211, <4 x float> poison, <4 x i32> zeroinitializer
  %213 = fmul <4 x float> %212, %208
  %214 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %186, <4 x float> %210, <4 x float> %213)
  %indvars.iv.next.6 = shl i64 %indvars.iv, 7
  %215 = or disjoint i64 %indvars.iv.next.6, 896
  %.reass.7 = add nuw nsw i64 %invariant.op, %215
  %216 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %.reass.7
  %217 = load <4 x half>, ptr addrspace(1) %216, align 2
  %218 = fpext <4 x half> %217 to <4 x float>
  %219 = fmul <4 x float> %13, %218
  %220 = extractelement <4 x float> %219, i64 0
  %221 = extractelement <4 x float> %219, i64 1
  %222 = fadd float %220, %221
  %223 = extractelement <4 x float> %219, i64 2
  %224 = fadd float %223, %222
  %225 = extractelement <4 x float> %219, i64 3
  %226 = fadd float %225, %224
  %227 = tail call float @llvm.agx.simd.reduce.f32(float %226, i32 1)
  %228 = fmul float %227, f0x3DB504F3
  %.nan.7 = tail call float @llvm.maximum.f32(float %.nan.6, float %228)
  %229 = fsub float %.nan.6, %.nan.7
  %230 = tail call float @llvm.exp.f32(float %229)
  %231 = fsub float %228, %.nan.7
  %232 = tail call float @llvm.exp.f32(float %231)
  %233 = tail call float @llvm.fma.f32(float %205, float %230, float %232)
  %234 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %.reass.7
  %235 = load <4 x half>, ptr addrspace(1) %234, align 2
  %236 = fpext <4 x half> %235 to <4 x float>
  %237 = insertelement <4 x float> poison, float %230, i64 0
  %238 = shufflevector <4 x float> %237, <4 x float> poison, <4 x i32> zeroinitializer
  %239 = insertelement <4 x float> poison, float %232, i64 0
  %240 = shufflevector <4 x float> %239, <4 x float> poison, <4 x i32> zeroinitializer
  %241 = fmul <4 x float> %240, %236
  %242 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %214, <4 x float> %238, <4 x float> %241)
  %indvars.iv.next.7 = add nuw nsw i64 %indvars.iv, 8
  %exitcond.not.7 = icmp eq i64 %indvars.iv.next.7, 512
  br i1 %exitcond.not.7, label %243, label %15

243:                                              ; preds = %15
  %244 = fdiv float 1.000000e+00, %233
  %245 = insertelement <4 x float> poison, float %244, i64 0
  %246 = shufflevector <4 x float> %245, <4 x float> poison, <4 x i32> zeroinitializer
  %247 = fmul <4 x float> %242, %246
  %248 = fptrunc <4 x float> %247 to <4 x half>
  %249 = getelementptr [2 x i8], ptr addrspace(1) %3, i64 %10
  store <4 x half> %248, ptr addrspace(1) %249, align 2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.index.in.simdgroup() #1

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.agx.simd.reduce.f32(float, i32 immarg) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #3

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="32" "agx-threads-per-threadgroup"="32,1,1" "agx-upper-bounds:air.threadgroup_position_in_grid"="32,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }


