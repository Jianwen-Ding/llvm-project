// Row quantization to int8: q = clamp(round(x * s), -127, 127), a thread per row of 256.
gpu.module @k {
  gpu.func @quant_rows(%x: memref<1024x256xf32, 1>, %q: memref<1024x256xi8, 1>, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %lo = arith.constant -127.0 : f32
    %hi = arith.constant 127.0 : f32
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %m = arith.mulf %v, %s : f32
      %rd = math.roundeven %m : f32
      %c1f = arith.maximumf %rd, %lo : f32
      %c2f = arith.minimumf %c1f, %hi : f32
      %qi = arith.fptosi %c2f : f32 to i8
      memref.store %qi, %q[%r, %j] : memref<1024x256xi8, 1>
    }
    gpu.return
  }
}
