// A 3-tap filter: loads of x[i], x[i + 1], x[i + 2] for the global id i
// (nuw on the narrowed i + c, L15-E177).
gpu.module @k {
  gpu.func @conv1d(%x: memref<65538xf32, 1>, %y: memref<65536xf32, 1>, %w0: f32, %w1: f32, %w2: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 256, 1, 1>} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %i = gpu.global_id x
    %i1 = arith.addi %i, %c1 : index
    %i2 = arith.addi %i, %c2 : index
    %a = memref.load %x[%i] : memref<65538xf32, 1>
    %b = memref.load %x[%i1] : memref<65538xf32, 1>
    %c = memref.load %x[%i2] : memref<65538xf32, 1>
    %p = arith.mulf %a, %w0 : f32
    %q = math.fma %b, %w1, %p : f32
    %r = math.fma %c, %w2, %q : f32
    memref.store %r, %y[%i] : memref<65536xf32, 1>
    gpu.return
  }
}
