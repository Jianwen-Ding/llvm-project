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

; The last add only feeds the last thread's store of the SIMD-group's total
; and is sunk under it: the prefix stays with the shuffles, before the
; branch, where all the scan's threads are active (in the branch it would
; see only thread 31).
; CHECK-LABEL: sunk:
; CHECK: simd_prefix_iadd
; CHECK: if_icmp
; CHECK-NOT: simd_prefix_iadd
; CHECK: stop
define agx_kernel void @sunk(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %x = load i32, ptr addrspace(1) %p
  %s0 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %x, i32 1)
  %c0 = icmp uge i32 %lane, 1
  %z0 = select i1 %c0, i32 %s0, i32 0
  %v0 = add i32 %x, %z0
  %s1 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v0, i32 2)
  %c1 = icmp uge i32 %lane, 2
  %z1 = select i1 %c1, i32 %s1, i32 0
  %v1 = add i32 %v0, %z1
  %s2 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v1, i32 4)
  %c2 = icmp uge i32 %lane, 4
  %z2 = select i1 %c2, i32 %s2, i32 0
  %v2 = add i32 %v1, %z2
  %s3 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v2, i32 8)
  %c3 = icmp uge i32 %lane, 8
  %z3 = select i1 %c3, i32 %s3, i32 0
  %v3 = add i32 %v2, %z3
  %s4 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %v3, i32 16)
  %c4 = icmp uge i32 %lane, 16
  %z4 = select i1 %c4, i32 %s4, i32 0
  %last = icmp eq i32 %lane, 31
  br i1 %last, label %st, label %done
st:
  %v4 = add i32 %v3, %z4
  store i32 %v4, ptr addrspace(1) %o
  br label %done
done:
  ret void
}

; A scan of a scan: the second's base is the first's last add, which the
; first replacement replaces (the second used it after it was deleted, an add
; of itself; found by a SIMD-group fuzz, L15-E380).
; CHECK-LABEL: scan_of_scan:
; CHECK: simd_prefix_iadd
; CHECK: simd_prefix_iadd
; CHECK: stop
define agx_kernel void @scan_of_scan(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %x = load i32, ptr addrspace(1) %p
  %as0 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %x, i32 1)
  %ac0 = icmp uge i32 %lane, 1
  %az0 = select i1 %ac0, i32 %as0, i32 0
  %av0 = add i32 %x, %az0
  %as1 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %av0, i32 2)
  %ac1 = icmp uge i32 %lane, 2
  %az1 = select i1 %ac1, i32 %as1, i32 0
  %av1 = add i32 %av0, %az1
  %as2 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %av1, i32 4)
  %ac2 = icmp uge i32 %lane, 4
  %az2 = select i1 %ac2, i32 %as2, i32 0
  %av2 = add i32 %av1, %az2
  %as3 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %av2, i32 8)
  %ac3 = icmp uge i32 %lane, 8
  %az3 = select i1 %ac3, i32 %as3, i32 0
  %av3 = add i32 %av2, %az3
  %as4 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %av3, i32 16)
  %ac4 = icmp uge i32 %lane, 16
  %az4 = select i1 %ac4, i32 %as4, i32 0
  %av4 = add i32 %av3, %az4
  %bs0 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %av4, i32 1)
  %bc0 = icmp uge i32 %lane, 1
  %bz0 = select i1 %bc0, i32 %bs0, i32 0
  %bv0 = add i32 %av4, %bz0
  %bs1 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %bv0, i32 2)
  %bc1 = icmp uge i32 %lane, 2
  %bz1 = select i1 %bc1, i32 %bs1, i32 0
  %bv1 = add i32 %bv0, %bz1
  %bs2 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %bv1, i32 4)
  %bc2 = icmp uge i32 %lane, 4
  %bz2 = select i1 %bc2, i32 %bs2, i32 0
  %bv2 = add i32 %bv1, %bz2
  %bs3 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %bv2, i32 8)
  %bc3 = icmp uge i32 %lane, 8
  %bz3 = select i1 %bc3, i32 %bs3, i32 0
  %bv3 = add i32 %bv2, %bz3
  %bs4 = call i32 @llvm.agx.simd.shuffle.up.i32(i32 %bv3, i32 16)
  %bc4 = icmp uge i32 %lane, 16
  %bz4 = select i1 %bc4, i32 %bs4, i32 0
  %bv4 = add i32 %bv3, %bz4
  store i32 %bv4, ptr addrspace(1) %o
  ret void
}
