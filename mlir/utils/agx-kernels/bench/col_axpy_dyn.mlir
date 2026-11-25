// agx-cost: --uniform 6=256
// y[i, c] = x[i, c] * w[i] + y[i, c] down columns of a runtime length n (256 here), a thread per column.
gpu.module @k {
  gpu.func @col_axpy_dyn(%x: memref<256x1024xf32, 1>, %w: memref<256xf32, 1>, %y: memref<256x1024xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %i = %c0 to %ni step %c1 {
      %v = memref.load %x[%i, %c] : memref<256x1024xf32, 1>
      %wv = memref.load %w[%i] : memref<256xf32, 1>
      %yv = memref.load %y[%i, %c] : memref<256x1024xf32, 1>
      %o = math.fma %v, %wv, %yv : f32
      memref.store %o, %y[%i, %c] : memref<256x1024xf32, 1>
    }
    gpu.return
  }
}
