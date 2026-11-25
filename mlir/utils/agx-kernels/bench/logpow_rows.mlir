// y = log(x) + pow(x, 0.75) over rows of 256 positive floats, a thread per row.
gpu.module @k {
  gpu.func @logpow_rows(%x: memref<1024x256xf32, 1>, %y: memref<1024x256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %p = arith.constant 0.75 : f32
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %l = math.log %v : f32
      %w = math.powf %v, %p : f32
      %o = arith.addf %l, %w : f32
      memref.store %o, %y[%r, %j] : memref<1024x256xf32, 1>
    }
    gpu.return
  }
}
