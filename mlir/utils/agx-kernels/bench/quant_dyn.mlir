// agx-cost: --uniform 4=256 --uniform 5=1069547520
// Row quantization to int8 over a runtime length n (256 here), a thread per row.
gpu.module @k {
  gpu.func @quant_dyn(%x: memref<1024x256xf32, 1>, %q: memref<1024x256xi8, 1>, %n: i32, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lo = arith.constant -127.0 : f32
    %hi = arith.constant 127.0 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
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
