// A 16 x 16 tile of a row-major 2-D array: index math from the block size.
gpu.module @k {
  gpu.func @tile2d(%in: memref<1048576xf32, 1>, %out: memref<1048576xf32, 1>, %ld: index) kernel
      attributes {known_block_size = array<i32: 16, 16, 1>} {
    %tx = gpu.thread_id x
    %ty = gpu.thread_id y
    %bx = gpu.block_id x
    %by = gpu.block_id y
    %dx = gpu.block_dim x
    %dy = gpu.block_dim y
    %x0 = arith.muli %bx, %dx : index
    %x = arith.addi %x0, %tx : index
    %y0 = arith.muli %by, %dy : index
    %y = arith.addi %y0, %ty : index
    %row = arith.muli %y, %ld : index
    %i = arith.addi %row, %x : index
    %v = memref.load %in[%i] : memref<1048576xf32, 1>
    %two = arith.constant 2.0 : f32
    %r = arith.mulf %v, %two : f32
    memref.store %r, %out[%i] : memref<1048576xf32, 1>
    gpu.return
  }
}
