// GELU, exact form, f32: 0.5 x (1 + erf(x / sqrt(2))).
gpu.module @k {
  gpu.func @gelu_erf(%x: memref<1048576xf32, 1>, %y: memref<1048576xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %gid = gpu.global_id x upper_bound 1048576
    %v = memref.load %x[%gid] : memref<1048576xf32, 1>
    %c = arith.constant 0.70710678 : f32
    %half = arith.constant 0.5 : f32
    %one = arith.constant 1.0 : f32
    %t = arith.mulf %v, %c : f32
    %e = math.erf %t : f32
    %u = arith.addf %e, %one : f32
    %w = arith.mulf %v, %half : f32
    %r = arith.mulf %w, %u : f32
    memref.store %r, %y[%gid] : memref<1048576xf32, 1>
    gpu.return
  }
}
