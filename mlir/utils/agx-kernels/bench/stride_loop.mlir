// A grid-stride loop to a 32-bit bound n (index_cast): the trip bound
// umin(n - i, 65536) is 64-bit but fits 32 bits (L15-E176).
// agx-cost: --uniform 4=4096
gpu.module @k {
  gpu.func @stride_loop(%x: memref<65536xf32, 1>, %y: memref<65536xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
    %c1024 = arith.constant 1024 : index
    %gid = gpu.global_id x
    %ni = arith.index_cast %n : i32 to index
    scf.for %i = %gid to %ni step %c1024 {
      %v = memref.load %x[%i] : memref<65536xf32, 1>
      %w = arith.mulf %v, %v : f32
      memref.store %w, %y[%i] : memref<65536xf32, 1>
    }
    gpu.return
  }
}
