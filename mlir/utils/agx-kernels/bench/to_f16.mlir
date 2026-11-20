// f32 rows to f16 with a scale, a thread per row of 256.
gpu.module @k {
  gpu.func @to_f16(%x: memref<1024x256xf32, 1>, %y: memref<1024x256xf16, 1>, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %m = arith.mulf %v, %s : f32
      %h = arith.truncf %m : f32 to f16
      memref.store %h, %y[%r, %j] : memref<1024x256xf16, 1>
    }
    gpu.return
  }
}
