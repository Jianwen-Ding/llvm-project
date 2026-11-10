; LLM decode GEMV in f16: y[row] = sum_k W[row][k] x[k] with f16 weights and activations, a SIMD-group per row, each
; lane a half2 of W and of x per step (fpext to float, two fmas), a float SIMD-group sum and an f16 store.
; K = 2048: 32 steps per lane.
; agx-cost: --uniform 6=2048 --wg 128
target triple = "agx-apple-macosx"
declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare i32 @llvm.agx.simdgroup.index.in.threadgroup()
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare float @llvm.agx.simd.reduce.f32(float, i32 immarg)
declare float @llvm.fma.f32(float, float, float)
define agx_kernel void @gemv_f16(ptr addrspace(1) noalias readonly %w, ptr addrspace(1) noalias readonly %x, ptr addrspace(1) noalias %y, i32 %k) {
entry:
  %b = call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %sg = call i32 @llvm.agx.simdgroup.index.in.threadgroup()
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %b4 = shl i32 %b, 2
  %row = add i32 %b4, %sg
  %pairs = lshr i32 %k, 1
  %rowoff = mul i32 %row, %pairs
  %any = icmp ult i32 %lane, %pairs
  br i1 %any, label %loop, label %done
loop:
  %j = phi i32 [ %lane, %entry ], [ %jn, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %a2, %loop ]
  %iw = add i32 %rowoff, %j
  %pw = getelementptr <2 x half>, ptr addrspace(1) %w, i32 %iw
  %px = getelementptr <2 x half>, ptr addrspace(1) %x, i32 %j
  %hw = load <2 x half>, ptr addrspace(1) %pw, align 4
  %hx = load <2 x half>, ptr addrspace(1) %px, align 4
  %fw = fpext <2 x half> %hw to <2 x float>
  %fx = fpext <2 x half> %hx to <2 x float>
  %w0 = extractelement <2 x float> %fw, i32 0
  %w1 = extractelement <2 x float> %fw, i32 1
  %x0 = extractelement <2 x float> %fx, i32 0
  %x1 = extractelement <2 x float> %fx, i32 1
  %a1 = call float @llvm.fma.f32(float %w0, float %x0, float %acc)
  %a2 = call float @llvm.fma.f32(float %w1, float %x1, float %a1)
  %jn = add i32 %j, 32
  %more = icmp ult i32 %jn, %pairs
  br i1 %more, label %loop, label %done
done:
  %part = phi float [ 0.0, %entry ], [ %a2, %loop ]
  %sum = call float @llvm.agx.simd.reduce.f32(float %part, i32 1)
  %first = icmp eq i32 %lane, 0
  br i1 %first, label %store, label %end
store:
  %h = fptrunc float %sum to half
  %py = getelementptr half, ptr addrspace(1) %y, i32 %row
  store half %h, ptr addrspace(1) %py, align 2
  br label %end
end:
  ret void
}
