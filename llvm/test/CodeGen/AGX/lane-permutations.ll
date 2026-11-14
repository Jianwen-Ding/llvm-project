; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-lane-permutations=0 < %s | FileCheck %s --check-prefix=OFF

; A tree of shuffles of one value and selects on the lane whose source lanes
; are one shuffle's is that shuffle (L15-E307): gpu.rotate's shuffle down and
; up and their select is the rotate, within 32 lanes the SIMD-group's and
; within 4 the quad-group's; an index shuffle of an index of the lane is
; the shuffle with those source lanes, for the shuffle per quad position.

; CHECK-LABEL: rotate32:
; CHECK: simd_shuffle_rotate_up {{r[0-9]+}}, {{r[0-9]+}}, 29
; CHECK-NOT: icmpsel
; CHECK-NOT: simd_shuffle_down
; OFF-LABEL: rotate32:
; OFF-NOT: shuffle_rotate
define agx_kernel void @rotate32(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %v = load float, ptr addrspace(1) %q
  %vi = bitcast float %v to i32
  %d = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %vi, i32 3)
  %u = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %vi, i32 29)
  %c = icmp ult i32 %lane, 29
  %r = select i1 %c, i32 %d, i32 %u
  %rf = bitcast i32 %r to float
  store float %rf, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: rotate4:
; CHECK: quad_shuffle_rotate_up {{r[0-9]+}}, {{r[0-9]+}}, 3
; CHECK-NOT: icmpsel
define agx_kernel void @rotate4(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr half, ptr addrspace(1) %p, i32 %t
  %v = load half, ptr addrspace(1) %q
  %vb = bitcast half %v to i16
  %vi = zext i16 %vb to i32
  %d = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %vi, i32 1)
  %u = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %vi, i32 3)
  %pos = and i32 %lane, 3
  %c = icmp eq i32 %pos, 3
  %dt = trunc i32 %d to i16
  %dh = bitcast i16 %dt to half
  %ut = trunc i32 %u to i16
  %uh = bitcast i16 %ut to half
  %r = select i1 %c, half %uh, half %dh
  store half %r, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: index:
; CHECK: simd_shuffle_rotate_up {{r[0-9]+}}, {{r[0-9]+}}, 31
; CHECK: simd_shuffle_xor {{r[0-9]+}}, {{r[0-9]+}}, 31
; CHECK: quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 2
; CHECK-NOT: shuffle
; OFF-LABEL: index:
; OFF-COUNT-4: simd_shuffle {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
define agx_kernel void @index(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %v = load i32, ptr addrspace(1) %q
  %a = add i32 %lane, 1
  %i0 = and i32 %a, 31
  %r0 = call i32 @llvm.agx.simd.shuffle.i32(i32 %v, i32 %i0)
  %i1 = sub i32 31, %lane
  %r1 = call i32 @llvm.agx.simd.shuffle.i32(i32 %r0, i32 %i1)
  %c = and i32 %lane, -4
  %i2 = or i32 %c, 2
  %r2 = call i32 @llvm.agx.simd.shuffle.i32(i32 %r1, i32 %i2)
  store i32 %r2, ptr addrspace(1) %q
  ret void
}

; The shuffles in another block (where other lanes may be active), or read
; elsewhere: they stay.
; CHECK-LABEL: stays:
; CHECK-NOT: shuffle_rotate
; CHECK-COUNT-2: simd_shuffle_down
; CHECK-NOT: shuffle_rotate
define agx_kernel void @stays(ptr addrspace(1) %p, i1 %b) {
entry:
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %v = load i32, ptr addrspace(1) %q
  %d = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %v, i32 3)
  %u = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v, i32 29)
  %c = icmp ult i32 %lane, 29
  %w = xor i32 %v, 1
  %d2 = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %w, i32 3)
  %u2 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %w, i32 29)
  %r2 = select i1 %c, i32 %d2, i32 %u2
  %s2 = add i32 %r2, %u2
  store i32 %s2, ptr addrspace(1) %q
  br i1 %b, label %then, label %done
then:
  %r = select i1 %c, i32 %d, i32 %u
  store i32 %r, ptr addrspace(1) %p
  br label %done
done:
  ret void
}

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.simd.shuffle.i32(i32, i32)
declare i32 @llvm.agx.simd.shuffle.down.i32(i32, i32)
declare i32 @llvm.agx.simd.shuffle.up.i32(i32, i32)
