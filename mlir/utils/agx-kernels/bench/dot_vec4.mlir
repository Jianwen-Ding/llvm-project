// Row dot products by vector<4xf32> fma, then a vector.reduction <add> of the
// four lanes (L15-E181).
gpu.module @k {
  gpu.func @dot_vec4(%a: memref<256x1024xf32, 1>, %b: memref<1024xf32, 1>, %y: memref<256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %z = arith.constant dense<0.0> : vector<4xf32>
    %r = gpu.thread_id x
    %s = scf.for %j = %c0 to %c1024 step %c4 iter_args(%acc = %z) -> vector<4xf32> {
      %u = vector.load %a[%r, %j] : memref<256x1024xf32, 1>, vector<4xf32>
      %v = vector.load %b[%j] : memref<1024xf32, 1>, vector<4xf32>
      %f = math.fma %u, %v, %acc : vector<4xf32>
      scf.yield %f : vector<4xf32>
    }
    %t = vector.reduction <add>, %s : vector<4xf32> into f32
    memref.store %t, %y[%r] : memref<256xf32, 1>
    gpu.return
  }
}
