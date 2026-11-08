; Bilinear resize of a one-channel f32 image (PyTorch's upsample_bilinear2d, align_corners = false): a thread per
; output pixel of a 2-D grid, source coordinates (x + 0.5) * scale - 0.5 clamped at 0, floor, the far neighbour
; clamped at the edge, four loads and two lerps. 512 x 512 from 1024 x 1024.
; agx-cost: --uniform 4=1024 --uniform 5=1024 --uniform 6=512 --uniform 7=512 --uniform 8=0x40000000 --uniform 9=0x40000000 --wg 16,16,1
target triple = "agx-apple-macosx"
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.grid.y()
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.floor.f32(float)
declare float @llvm.fma.f32(float, float, float)
declare i32 @llvm.smin.i32(i32, i32)
define agx_kernel void @bilinear_resize(ptr addrspace(1) noalias readonly %in, ptr addrspace(1) noalias %out, i32 %iw, i32 %ih, i32 %ow, i32 %oh, float %scx, float %scy) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %y = call i32 @llvm.agx.thread.position.in.grid.y()
  %inx = icmp ult i32 %x, %ow
  %iny = icmp ult i32 %y, %oh
  %in2 = and i1 %inx, %iny
  br i1 %in2, label %body, label %end
body:
  %xf = uitofp i32 %x to float
  %yf = uitofp i32 %y to float
  %xh = fadd float %xf, 0.5
  %yh = fadd float %yf, 0.5
  %sx0 = call float @llvm.fma.f32(float %xh, float %scx, float -0.5)
  %sy0 = call float @llvm.fma.f32(float %yh, float %scy, float -0.5)
  %sx = call float @llvm.maxnum.f32(float %sx0, float 0.0)
  %sy = call float @llvm.maxnum.f32(float %sy0, float 0.0)
  %fx = call float @llvm.floor.f32(float %sx)
  %fy = call float @llvm.floor.f32(float %sy)
  %x0 = fptosi float %fx to i32
  %y0 = fptosi float %fy to i32
  %tx = fsub float %sx, %fx
  %ty = fsub float %sy, %fy
  %iw1 = add i32 %iw, -1
  %ih1 = add i32 %ih, -1
  %x0c = call i32 @llvm.smin.i32(i32 %x0, i32 %iw1)
  %y0c = call i32 @llvm.smin.i32(i32 %y0, i32 %ih1)
  %x1a = add i32 %x0c, 1
  %y1a = add i32 %y0c, 1
  %x1 = call i32 @llvm.smin.i32(i32 %x1a, i32 %iw1)
  %y1 = call i32 @llvm.smin.i32(i32 %y1a, i32 %ih1)
  %r0 = mul i32 %y0c, %iw
  %r1 = mul i32 %y1, %iw
  %i00 = add i32 %r0, %x0c
  %i01 = add i32 %r0, %x1
  %i10 = add i32 %r1, %x0c
  %i11 = add i32 %r1, %x1
  %p00 = getelementptr float, ptr addrspace(1) %in, i32 %i00
  %p01 = getelementptr float, ptr addrspace(1) %in, i32 %i01
  %p10 = getelementptr float, ptr addrspace(1) %in, i32 %i10
  %p11 = getelementptr float, ptr addrspace(1) %in, i32 %i11
  %v00 = load float, ptr addrspace(1) %p00, align 4
  %v01 = load float, ptr addrspace(1) %p01, align 4
  %v10 = load float, ptr addrspace(1) %p10, align 4
  %v11 = load float, ptr addrspace(1) %p11, align 4
  %d0 = fsub float %v01, %v00
  %d1 = fsub float %v11, %v10
  %t0 = call float @llvm.fma.f32(float %tx, float %d0, float %v00)
  %t1 = call float @llvm.fma.f32(float %tx, float %d1, float %v10)
  %dt = fsub float %t1, %t0
  %r = call float @llvm.fma.f32(float %ty, float %dt, float %t0)
  %ro = mul i32 %y, %ow
  %io = add i32 %ro, %x
  %po = getelementptr float, ptr addrspace(1) %out, i32 %io
  store float %r, ptr addrspace(1) %po, align 4
  br label %end
end:
  ret void
}
