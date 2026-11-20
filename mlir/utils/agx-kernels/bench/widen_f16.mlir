// f16 rows to f32 with a scale and bias, a thread per row of 256.
gpu.module @k {
  gpu.func @widen_f16(%x: memref<1024x256xf16, 1>, %y: memref<1024x256xf32, 1>, %s: f32, %b: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf16, 1>
      %w = arith.extf %v : f16 to f32
      %o = math.fma %w, %s, %b : f32
      memref.store %o, %y[%r, %j] : memref<1024x256xf32, 1>
    }
    gpu.return
  }
}
