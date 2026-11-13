; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-simd-scans=0 < %s | FileCheck %s --check-prefix=OFF

; A SIMD-group's inclusive add scan by shuffles up 1, 2, 4, 8 and 16 (CUDA's
; __shfl_up_sync scan, MLIR's gpu.shuffle up) is the exclusive simd_prefix
; plus the value (L15-E241). A chain that stops at 8 (a scan of 16-lane
; segments) or tests the wrong lanes stays.

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.simd.shuffle.up.i32(i32, i32)

; CHECK-LABEL: scan:
; CHECK-NOT: simd_shuffle_up
; CHECK: simd_prefix_iadd [[P:r[0-9]+]], [[V:r[0-9]+]]
; CHECK-NEXT: iadd {{r[0-9]+}}, [[P]], [[V]]
; CHECK-NOT: simd_shuffle_up
; CHECK: stop
; OFF-LABEL: scan:
; OFF-COUNT-5: simd_shuffle_up
define agx_kernel void @scan(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %l = call i32 @llvm.agx.thread.index.in.simdgroup()
  %v = load i32, ptr addrspace(1) %p, align 4
  %z1 = icmp eq i32 %l, 0
  %s1 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v, i32 1)
  %t1 = select i1 %z1, i32 0, i32 %s1
  %a1 = add i32 %t1, %v
  %c2 = icmp ugt i32 %l, 1
  %s2 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a1, i32 2)
  %t2 = select i1 %c2, i32 %s2, i32 0
  %a2 = add i32 %a1, %t2
  %c4 = icmp uge i32 %l, 4
  %s4 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a2, i32 4)
  %t4 = select i1 %c4, i32 %s4, i32 0
  %a4 = add i32 %a2, %t4
  %c8 = icmp ult i32 %l, 8
  %s8 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a4, i32 8)
  %t8 = select i1 %c8, i32 0, i32 %s8
  %a8 = add i32 %t8, %a4
  %c16 = icmp ugt i32 %l, 15
  %s16 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a8, i32 16)
  %t16 = select i1 %c16, i32 %s16, i32 0
  %a16 = add i32 %a8, %t16
  store i32 %a16, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: segments:
; CHECK: simd_shuffle_up
; CHECK-NOT: simd_prefix
; CHECK: stop
define agx_kernel void @segments(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %l = call i32 @llvm.agx.thread.index.in.simdgroup()
  %v = load i32, ptr addrspace(1) %p, align 4
  %z1 = icmp eq i32 %l, 0
  %s1 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v, i32 1)
  %t1 = select i1 %z1, i32 0, i32 %s1
  %a1 = add i32 %t1, %v
  %c2 = icmp ugt i32 %l, 1
  %s2 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a1, i32 2)
  %t2 = select i1 %c2, i32 %s2, i32 0
  %a2 = add i32 %a1, %t2
  %c4 = icmp ugt i32 %l, 3
  %s4 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a2, i32 4)
  %t4 = select i1 %c4, i32 %s4, i32 0
  %a4 = add i32 %a2, %t4
  %c8 = icmp ugt i32 %l, 7
  %s8 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a4, i32 8)
  %t8 = select i1 %c8, i32 %s8, i32 0
  %a8 = add i32 %a4, %t8
  store i32 %a8, ptr addrspace(1) %o, align 4
  ret void
}

; lane > 2 for the shuffle by 2 (lane 2 loses lane 0's value): not a scan.
; CHECK-LABEL: wronglanes:
; CHECK: simd_shuffle_up
; CHECK-NOT: simd_prefix
; CHECK: stop
define agx_kernel void @wronglanes(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %l = call i32 @llvm.agx.thread.index.in.simdgroup()
  %v = load i32, ptr addrspace(1) %p, align 4
  %z1 = icmp eq i32 %l, 0
  %s1 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v, i32 1)
  %t1 = select i1 %z1, i32 0, i32 %s1
  %a1 = add i32 %t1, %v
  %c2 = icmp ugt i32 %l, 2
  %s2 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a1, i32 2)
  %t2 = select i1 %c2, i32 %s2, i32 0
  %a2 = add i32 %a1, %t2
  %c4 = icmp ugt i32 %l, 3
  %s4 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a2, i32 4)
  %t4 = select i1 %c4, i32 %s4, i32 0
  %a4 = add i32 %a2, %t4
  %c8 = icmp ugt i32 %l, 7
  %s8 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a4, i32 8)
  %t8 = select i1 %c8, i32 %s8, i32 0
  %a8 = add i32 %a4, %t8
  %c16 = icmp ugt i32 %l, 15
  %s16 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %a8, i32 16)
  %t16 = select i1 %c16, i32 %s16, i32 0
  %a16 = add i32 %a8, %t16
  store i32 %a16, ptr addrspace(1) %o, align 4
  ret void
}
