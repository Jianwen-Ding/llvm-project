; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-quad-shifts=0 < %s | FileCheck %s --check-prefix=OFF

; gpu.shuffle down / up by k within segments of 4 lanes (convert-gpu-to-air:
; a SIMD-group shift and a select of the own value where the source lane is
; past the segment) is the quad-group's shift, which keeps the own value
; there; within 2 lanes the select stays on the quad shift. Wider segments,
; and a shift read outside of such selects, stay (L15-E305).

; CHECK-LABEL: four:
; CHECK-NOT: icmpsel
; CHECK-DAG: quad_shuffle_down {{r[0-9]+}}, {{r[0-9]+}}, 1
; CHECK-DAG: quad_shuffle_up {{r[0-9]+}}, {{r[0-9]+}}, 3
; CHECK-NOT: simd_shuffle
; OFF-LABEL: four:
; OFF-NOT: quad_shuffle
define agx_kernel void @four(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %v = load float, ptr addrspace(1) %q
  %vi = bitcast float %v to i32
  %seg = and i32 %lane, -4
  %end = add i32 %seg, 4
  %l1 = add i32 %lane, 1
  %in = icmp slt i32 %l1, %end
  %d = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %vi, i32 1)
  %df = bitcast i32 %d to float
  %r0 = select i1 %in, float %df, float %v
  %lm3 = add i32 %lane, -3
  %out = icmp slt i32 %lm3, %seg
  %u = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %vi, i32 3)
  %uf = bitcast i32 %u to float
  %r1 = select i1 %out, float %v, float %uf
  %s = fadd float %r0, %r1
  %qo = getelementptr float, ptr addrspace(1) %o, i32 %t
  store float %s, ptr addrspace(1) %qo
  ret void
}

; CHECK-LABEL: two:
; CHECK: quad_shuffle_down {{r[0-9]+}}, {{r[0-9]+}}, 1
; CHECK: icmpsel
; CHECK-NOT: simd_shuffle
define agx_kernel void @two(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr half, ptr addrspace(1) %p, i32 %t
  %v = load half, ptr addrspace(1) %q
  %vb = bitcast half %v to i16
  %vi = zext i16 %vb to i32
  %seg = and i32 %lane, -2
  %end = add i32 %seg, 2
  %l1 = add i32 %lane, 1
  %in = icmp slt i32 %l1, %end
  %d = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %vi, i32 1)
  %dt = trunc i32 %d to i16
  %dh = bitcast i16 %dt to half
  %r = select i1 %in, half %dh, half %v
  store half %r, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: stays:
; CHECK-NOT: quad_shuffle
define agx_kernel void @stays(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %v = load i32, ptr addrspace(1) %q
  %seg8 = and i32 %lane, -8
  %end8 = add i32 %seg8, 8
  %l1 = add i32 %lane, 1
  %in8 = icmp slt i32 %l1, %end8
  %d = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %v, i32 1)
  %r0 = select i1 %in8, i32 %d, i32 %v
  store i32 %r0, ptr addrspace(1) %q
  %seg = and i32 %lane, -4
  %l2 = add i32 %lane, 2
  %end = add i32 %seg, 4
  %in = icmp slt i32 %l2, %end
  %d2 = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %v, i32 2)
  %r1 = select i1 %in, i32 %d2, i32 %v
  %s = add i32 %r1, %d2
  %qo = getelementptr i32, ptr addrspace(1) %o, i32 %t
  store i32 %s, ptr addrspace(1) %qo
  ret void
}
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.simd.shuffle.down.i32(i32, i32)
declare i32 @llvm.agx.simd.shuffle.up.i32(i32, i32)
