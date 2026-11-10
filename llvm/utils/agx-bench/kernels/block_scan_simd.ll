; Inclusive prefix sum of a 256-thread threadgroup's i32 values by SIMD-group scans (CUB's BlockScan with warp
; scans): a simd_prefix per SIMD-group, its last lane's total to threadgroup memory, a barrier, the first SIMD-group
; scans the eight totals, a barrier, and each thread adds its SIMD-group's offset.
; agx-cost: --wg 256
target triple = "agx-apple-macosx"
@totals = internal addrspace(3) global [8 x i32] undef
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.simdgroup.index.in.threadgroup()
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.simd.prefix.i32(i32, i32 immarg)
declare void @llvm.agx.threadgroup.barrier()
define agx_kernel void @block_scan_simd(ptr addrspace(1) noalias readonly %in, ptr addrspace(1) noalias %out) {
entry:
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %sg = call i32 @llvm.agx.simdgroup.index.in.threadgroup()
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  %e = call i32 @llvm.agx.simd.prefix.i32(i32 %v, i32 16)
  %s = add i32 %e, %v
  %last = icmp eq i32 %lane, 31
  br i1 %last, label %put, label %sync1
put:
  %pt = getelementptr [8 x i32], ptr addrspace(3) @totals, i32 0, i32 %sg
  store i32 %s, ptr addrspace(3) %pt, align 4
  br label %sync1
sync1:
  call void @llvm.agx.threadgroup.barrier()
  %first = icmp eq i32 %sg, 0
  br i1 %first, label %scan2, label %sync2
scan2:
  %few = icmp ult i32 %lane, 8
  %l7 = and i32 %lane, 7
  %pl = getelementptr [8 x i32], ptr addrspace(3) @totals, i32 0, i32 %l7
  %t = load i32, ptr addrspace(3) %pl, align 4
  %tv = select i1 %few, i32 %t, i32 0
  %te = call i32 @llvm.agx.simd.prefix.i32(i32 %tv, i32 16)
  br i1 %few, label %put2, label %sync2
put2:
  store i32 %te, ptr addrspace(3) %pl, align 4
  br label %sync2
sync2:
  call void @llvm.agx.threadgroup.barrier()
  %po = getelementptr [8 x i32], ptr addrspace(3) @totals, i32 0, i32 %sg
  %off = load i32, ptr addrspace(3) %po, align 4
  %r = add i32 %s, %off
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %r, ptr addrspace(1) %q, align 4
  ret void
}
