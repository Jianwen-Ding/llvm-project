// agx-cost: --uniform 6=512
// Half rows scaled by a half per row, a runtime length, a thread per row.
gpu.module @k {
  gpu.func @hscale_dyn(%x: memref<1024x512xf16, 1>, %w: memref<1024xf16, 1>, %y: memref<1024x512xf16, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %s = memref.load %w[%r] : memref<1024xf16, 1>
    scf.for %j = %c0 to %ni step %c1 {
      %p = memref.load %x[%r, %j] : memref<1024x512xf16, 1>
      %b = arith.mulf %p, %s : f16
      memref.store %b, %y[%r, %j] : memref<1024x512xf16, 1>
    }
    gpu.return
  }
}
