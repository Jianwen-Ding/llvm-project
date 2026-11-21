// y = x * 2 + z over rows of 32, a thread per row.
gpu.module @k {
  gpu.func @axpy_c32(%x: memref<4096x32xf32, 1>, %z: memref<4096x32xf32, 1>, %y: memref<4096x32xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cT = arith.constant 32 : index
    %two = arith.constant 2.0 : f32
    %r = gpu.global_id x upper_bound 4096
    scf.for %j = %c0 to %cT step %c1 {
      %v = memref.load %x[%r, %j] : memref<4096x32xf32, 1>
      %u = memref.load %z[%r, %j] : memref<4096x32xf32, 1>
      %o = math.fma %v, %two, %u : f32
      memref.store %o, %y[%r, %j] : memref<4096x32xf32, 1>
    }
    gpu.return
  }
}
