// Sparse y = A x in CSR, a thread per row: an scf.for between two row
// offsets loaded as i32 (16 entries per row here), with a gather of x by
// the column indices.
// agx-cost: --uniform 0=0x1000000 --uniform 1=0 --map 0x1000000=@../../../../llvm/utils/agx-bench/kernels/data/rowptr16.bin --uniform 2=0x2000000 --uniform 3=0 --alloc 0x2000000=1048576 --uniform 4=0x3000000 --uniform 5=0 --alloc 0x3000000=1048576 --uniform 6=0x4000000 --uniform 7=0 --alloc 0x4000000=16384 --uniform 8=0x5000000 --uniform 9=0 --alloc 0x5000000=16384 --wg 256
gpu.module @k {
  gpu.func @spmv_csr(%rp: memref<4097xi32, 1>, %ci: memref<262144xi32, 1>, %va: memref<262144xf32, 1>,
                     %x: memref<4096xf32, 1>, %y: memref<4096xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %row = gpu.global_id x upper_bound 4096
    %next = arith.addi %row, %c1 : index
    %b32 = memref.load %rp[%row] : memref<4097xi32, 1>
    %e32 = memref.load %rp[%next] : memref<4097xi32, 1>
    %b = arith.index_cast %b32 : i32 to index
    %e = arith.index_cast %e32 : i32 to index
    %s = scf.for %j = %b to %e step %c1 iter_args(%acc = %zero) -> f32 {
      %col32 = memref.load %ci[%j] : memref<262144xi32, 1>
      %col = arith.index_cast %col32 : i32 to index
      %a = memref.load %va[%j] : memref<262144xf32, 1>
      %xv = memref.load %x[%col] : memref<4096xf32, 1>
      %n = math.fma %a, %xv, %acc : f32
      scf.yield %n : f32
    }
    memref.store %s, %y[%row] : memref<4096xf32, 1>
    gpu.return
  }
}
