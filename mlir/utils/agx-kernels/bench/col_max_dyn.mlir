// agx-cost: --uniform 4=256
// Column maxima (maxnum) down a runtime length n (256 here), a thread per column.
gpu.module @k {
  gpu.func @col_max_dyn(%x: memref<256x1024xf32, 1>, %y: memref<1024xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant -3.0e38 : f32
    %c = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %s = scf.for %i = %c0 to %ni step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%i, %c] : memref<256x1024xf32, 1>
      %b = arith.maxnumf %a, %v : f32
      scf.yield %b : f32
    }
    memref.store %s, %y[%c] : memref<1024xf32, 1>
    gpu.return
  }
}
