; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-ranged-compares=0 < %s | FileCheck %s --check-prefix=OFF

; Compares that the SIMD lane's range [0, 32) decides fold, and lane + k
; against a constant becomes lane against c - k (unsigned where both are
; non-negative): a scan step's segment test (lane - d >= lane & -32,
; lane - d < (lane & -32) + 32, as convert-gpu-to-air lowers a shuffle up by
; a uniform offset) is lane >= d, which the scan recognition takes: the
; Hillis-Steele scan is the hardware's prefix sum (L15-E300).

; CHECK-LABEL: scan:
; CHECK: simd_prefix_iadd
; CHECK-NOT: simd_shuffle_up
; CHECK: stop
; OFF-LABEL: scan:
; OFF: simd_shuffle_up
define agx_kernel void @scan(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %x0 = load i32, ptr addrspace(1) %q
  %start = and i32 %lane, -32
  %end = add i32 %start, 32
  %x1 = call i32 @step(i32 %x0, i32 %lane, i32 %start, i32 %end, i32 1)
  %x2 = call i32 @step(i32 %x1, i32 %lane, i32 %start, i32 %end, i32 2)
  %x3 = call i32 @step(i32 %x2, i32 %lane, i32 %start, i32 %end, i32 4)
  %x4 = call i32 @step(i32 %x3, i32 %lane, i32 %start, i32 %end, i32 8)
  %x5 = call i32 @step(i32 %x4, i32 %lane, i32 %start, i32 %end, i32 16)
  store i32 %x5, ptr addrspace(1) %q
  ret void
}
define internal i32 @step(i32 %x, i32 %lane, i32 %start, i32 %end, i32 %d) alwaysinline {
  %src = sub i32 %lane, %d
  %lo = icmp sge i32 %src, %start
  %hi = icmp slt i32 %src, %end
  %ok = and i1 %lo, %hi
  %u = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %x, i32 %d)
  %v = select i1 %ok, i32 %u, i32 0
  %r = add i32 %x, %v
  ret i32 %r
}
; The range through xor (and or) by a constant too: a butterfly step's
; segment test, lane ^ d in [lane & -32, (lane & -32) + 32), is true, and the
; select of the own value goes (L15-E301).
; CHECK-LABEL: butterfly:
; CHECK: simd_shuffle_xor
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @butterfly(ptr addrspace(1) %p) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %x = load i32, ptr addrspace(1) %q
  %start = and i32 %lane, -32
  %end = add i32 %start, 32
  %src = xor i32 %lane, 16
  %lo = icmp sge i32 %src, %start
  %hi = icmp slt i32 %src, %end
  %ok = and i1 %lo, %hi
  %u = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %x, i32 16)
  %v = select i1 %ok, i32 %u, i32 %x
  %r = add i32 %x, %v
  store i32 %r, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.agx.simd.shuffle.xor.i32(i32, i32)
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.simd.shuffle.up.i32(i32, i32)
