// y = sin(x) * cos(2x) over rows of 256, a thread per row.
gpu.module @k {
  gpu.func @sincos_rows(%x: memref<1024x256xf32, 1>, %y: memref<1024x256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %two = arith.constant 2.0 : f32
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %s = math.sin %v : f32
      %v2 = arith.mulf %v, %two : f32
      %c = math.cos %v2 : f32
      %o = arith.mulf %s, %c : f32
      memref.store %o, %y[%r, %j] : memref<1024x256xf32, 1>
    }
    gpu.return
  }
}
