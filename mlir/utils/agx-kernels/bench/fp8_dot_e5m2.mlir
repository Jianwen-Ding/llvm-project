// agx-cost: --wg 256
// A row of fp8 weights times an f32 vector: a thread per row of 256 elements, vector<4 x fp8> loads.
gpu.module @k {
  gpu.func @dot(%w: memref<256x256xf8E5M2, 1>, %x: memref<256xf32, 1>, %o: memref<256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %z = arith.constant dense<0.0> : vector<4xf32>
    %r = gpu.thread_id x
    %acc = scf.for %j = %c0 to %c256 step %c4 iter_args(%s = %z) -> (vector<4xf32>) {
      %q = vector.load %w[%r, %j] : memref<256x256xf8E5M2, 1>, vector<4xf8E5M2>
      %f = arith.extf %q : vector<4xf8E5M2> to vector<4xf32>
      %xv = vector.load %x[%j] : memref<256xf32, 1>, vector<4xf32>
      %n = math.fma %f, %xv, %s : vector<4xf32>
      scf.yield %n : vector<4xf32>
    }
    %t = vector.reduction <add>, %acc : vector<4xf32> into f32
    memref.store %t, %o[%r] : memref<256xf32, 1>
    gpu.return
  }
}
