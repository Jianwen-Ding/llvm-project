// agx-cost: --wg 256
// int4 weights (two a byte, vector.bitcast of 4 bytes to 8 x i4) times an f32 vector, a thread per row.
gpu.module @k {
  gpu.func @dot(%w: memref<256x128xi8, 1>, %x: memref<256xf32, 1>, %o: memref<256xf32, 1>, %sc: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %z = arith.constant dense<0.0> : vector<8xf32>
    %r = gpu.thread_id x
    %acc = scf.for %j = %c0 to %c128 step %c4 iter_args(%s = %z) -> (vector<8xf32>) {
      %b = vector.load %w[%r, %j] : memref<256x128xi8, 1>, vector<4xi8>
      %q = vector.bitcast %b : vector<4xi8> to vector<8xi4>
      %e = arith.extsi %q : vector<8xi4> to vector<8xi32>
      %f = arith.sitofp %e : vector<8xi32> to vector<8xf32>
      %k = arith.muli %j, %c2 : index
      %xv = vector.load %x[%k] : memref<256xf32, 1>, vector<8xf32>
      %n = math.fma %f, %xv, %s : vector<8xf32>
      scf.yield %n : vector<8xf32>
    }
    %t = vector.reduction <add>, %acc : vector<8xf32> into f32
    %u = arith.mulf %t, %sc : f32
    memref.store %u, %o[%r] : memref<256xf32, 1>
    gpu.return
  }
}
