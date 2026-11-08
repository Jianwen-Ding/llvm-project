// A 1024 x 1024 transpose through a padded 32 x 33 threadgroup tile, as in
// CUDA's transposeCoalesced: a 32 x 8 block reads four rows of the tile per
// thread, coalesced, waits at a barrier and writes four rows of the
// transposed tile, also coalesced.
// agx-cost: --wg 32,8,1
gpu.module @k {
  gpu.func @transpose_smem(%in: memref<1024x1024xf32, 1>, %out: memref<1024x1024xf32, 1>)
      workgroup(%tile: memref<32x33xf32, 3>) kernel
      attributes {known_block_size = array<i32: 32, 8, 1>} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %tx = gpu.thread_id x
    %ty = gpu.thread_id y
    %bx = gpu.block_id x upper_bound 32
    %by = gpu.block_id y upper_bound 32
    %x0 = arith.muli %bx, %c32 : index
    %y0 = arith.muli %by, %c32 : index
    %x = arith.addi %x0, %tx : index
    scf.for %j = %c0 to %c32 step %c8 {
      %r = arith.addi %ty, %j : index
      %y = arith.addi %y0, %r : index
      %v = memref.load %in[%y, %x] : memref<1024x1024xf32, 1>
      memref.store %v, %tile[%r, %tx] : memref<32x33xf32, 3>
    }
    gpu.barrier
    %ox = arith.addi %y0, %tx : index
    scf.for %j = %c0 to %c32 step %c8 {
      %r = arith.addi %ty, %j : index
      %oy = arith.addi %x0, %r : index
      %v = memref.load %tile[%tx, %r] : memref<32x33xf32, 3>
      memref.store %v, %out[%oy, %ox] : memref<1024x1024xf32, 1>
    }
    gpu.return
  }
}
