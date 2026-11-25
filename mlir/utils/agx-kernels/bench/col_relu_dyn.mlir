// agx-cost: --uniform 6=256
// y[i, c] = max(x[i, c] * w[i], 0) down columns of a runtime length n (256 here), a thread per column.
gpu.module @k {
  gpu.func @col_relu_dyn(%x: memref<256x1024xf32, 1>, %w: memref<256xf32, 1>, %y: memref<256x1024xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %i = %c0 to %ni step %c1 {
      %v = memref.load %x[%i, %c] : memref<256x1024xf32, 1>
      %wv = memref.load %w[%i] : memref<256xf32, 1>
      %m = arith.mulf %v, %wv : f32
      %z = arith.constant 0.0 : f32
      %o = arith.maximumf %m, %z : f32
      memref.store %o, %y[%i, %c] : memref<256x1024xf32, 1>
    }
    gpu.return
  }
}
