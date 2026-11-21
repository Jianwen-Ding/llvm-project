// agx-cost: --uniform 4=512 --uniform 5=1069547520
// half rows quantized to int8, a runtime length, a thread per row.
gpu.module @k {
  gpu.func @q_f16_i8_dyn(%x: memref<1024x512xf16, 1>, %y: memref<1024x512xi8, 1>, %n: i32, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x512xf16, 1>
      %e = arith.extf %v : f16 to f32
      %m = arith.mulf %e, %s : f32
      %rd = math.roundeven %m : f32
      %lo = arith.constant -127.0 : f32
      %hi = arith.constant 127.0 : f32
      %c1f = arith.maximumf %rd, %lo : f32
      %c2f = arith.minimumf %c1f, %hi : f32
      %o = arith.fptosi %c2f : f32 to i8
      memref.store %o, %y[%r, %j] : memref<1024x512xi8, 1>
    }
    gpu.return
  }
}
