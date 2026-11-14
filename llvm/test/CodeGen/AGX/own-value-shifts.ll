; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-own-value-shifts=0 < %s | FileCheck %s --check-prefix=OFF

; select(lane < 32 - k, shuffle_down(v, k), v) is the shuffle: AIR's shifts
; leave the lanes without a source lane their own value (as
; convert-gpu-to-air takes them for shifts by constants); the same for
; shuffle_up and lane >= k, through a float's bitcasts (L15-E302).

; CHECK-LABEL: down:
; CHECK: simd_shuffle_down
; CHECK-NOT: icmpsel
; CHECK: stop
; OFF-LABEL: down:
; OFF: icmpsel ult
define agx_kernel void @down(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %v = load float, ptr addrspace(1) %q
  %vi = bitcast float %v to i32
  %s = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %vi, i32 8)
  %sf = bitcast i32 %s to float
  %ok = icmp ult i32 %lane, 24
  %r = select i1 %ok, float %sf, float %v
  %a = fadd float %v, %r
  store float %a, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: up:
; CHECK: simd_shuffle_up
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @up(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %v = load i32, ptr addrspace(1) %q
  %s = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v, i32 4)
  %ok = icmp ugt i32 %lane, 3
  %r = select i1 %ok, i32 %s, i32 %v
  %a = mul i32 %v, %r
  store i32 %a, ptr addrspace(1) %q
  ret void
}

; Not the source test (lane < 20 with a shift of 8): kept.
; CHECK-LABEL: other:
; CHECK: icmpsel
define agx_kernel void @other(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %v = load i32, ptr addrspace(1) %q
  %s = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %v, i32 8)
  %ok = icmp ult i32 %lane, 20
  %r = select i1 %ok, i32 %s, i32 %v
  store i32 %r, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.simd.shuffle.down.i32(i32, i32)
declare i32 @llvm.agx.simd.shuffle.up.i32(i32, i32)
