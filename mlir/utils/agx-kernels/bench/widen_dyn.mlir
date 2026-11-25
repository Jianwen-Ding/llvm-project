// agx-cost: --uniform 4=256 --uniform 5=1069547520
// f16 rows widened to f32 with a scale over a runtime length n (256 here), a thread per row.
gpu.module @k {
  gpu.func @widen_dyn(%x: memref<1024x256xf16, 1>, %y: memref<1024x256xf32, 1>, %n: i32, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf16, 1>
      %w = arith.extf %v : f16 to f32
      %o = arith.mulf %w, %s : f32
      memref.store %o, %y[%r, %j] : memref<1024x256xf32, 1>
    }
    gpu.return
  }
}
