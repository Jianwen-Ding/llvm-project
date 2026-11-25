// agx-cost: --uniform 4=512 --uniform 5=1069547520
// int8 rows dequantized to halves, a runtime length, a thread per row.
gpu.module @k {
  gpu.func @deq_i8_f16_dyn(%x: memref<1024x512xi8, 1>, %y: memref<1024x512xf16, 1>, %n: i32, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x512xi8, 1>
      %f = arith.sitofp %v : i8 to f32
      %m = arith.mulf %f, %s : f32
      %o = arith.truncf %m : f32 to f16
      memref.store %o, %y[%r, %j] : memref<1024x512xf16, 1>
    }
    gpu.return
  }
}
