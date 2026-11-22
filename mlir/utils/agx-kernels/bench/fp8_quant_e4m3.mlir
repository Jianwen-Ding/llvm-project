// agx-cost: --wg 256
// Quantize: f32 times a scale to f8E4M3FN, vector<4> a thread-iteration over rows of 256.
gpu.module @k {
  gpu.func @q(%x: memref<256x256xf32, 1>, %w: memref<256x256xf8E4M3FN, 1>, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %r = gpu.thread_id x
    %sv = vector.broadcast %s : f32 to vector<4xf32>
    scf.for %j = %c0 to %c256 step %c4 {
      %v = vector.load %x[%r, %j] : memref<256x256xf32, 1>, vector<4xf32>
      %m = arith.mulf %v, %sv : vector<4xf32>
      %q = arith.truncf %m : vector<4xf32> to vector<4xf8E4M3FN>
      vector.store %q, %w[%r, %j] : memref<256x256xf8E4M3FN, 1>, vector<4xf8E4M3FN>
    }
    gpu.return
  }
}
