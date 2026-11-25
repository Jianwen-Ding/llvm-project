// int8-weight GEMV with float activations: y[r] = s[r] * sum_k w[r, k] * x[k],
// a thread per row, vector<4xi8> weights converted to f32.
gpu.module @k {
  gpu.func @gemv_q8f(%w: memref<1024x1024xi8, 1>, %x: memref<1024xf32, 1>, %s: memref<1024xf32, 1>, %y: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %z = arith.constant dense<0.0> : vector<4xf32>
    %r = gpu.global_id x upper_bound 1024
    %acc = scf.for %j = %c0 to %c1024 step %c4 iter_args(%a = %z) -> vector<4xf32> {
      %wv = vector.load %w[%r, %j] : memref<1024x1024xi8, 1>, vector<4xi8>
      %xv = vector.load %x[%j] : memref<1024xf32, 1>, vector<4xf32>
      %wf = arith.sitofp %wv : vector<4xi8> to vector<4xf32>
      %n = math.fma %wf, %xv, %a : vector<4xf32>
      scf.yield %n : vector<4xf32>
    }
    %t = vector.reduction <add>, %acc : vector<4xf32> into f32
    %sc = memref.load %s[%r] : memref<1024xf32, 1>
    %o = arith.mulf %t, %sc : f32
    memref.store %o, %y[%r] : memref<1024xf32, 1>
    gpu.return
  }
}
