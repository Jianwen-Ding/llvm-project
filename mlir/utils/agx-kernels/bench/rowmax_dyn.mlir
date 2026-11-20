// agx-cost: --uniform 4=512
// Row maxima over rows of a runtime length n (512 here), a thread per row.
gpu.module @k {
  gpu.func @rowmax_dyn(%x: memref<1024x512xf32, 1>, %y: memref<1024xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant -3.0e38 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %s = scf.for %j = %c0 to %ni step %c1 iter_args(%a = %zf) -> f32 {
      %p = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %b = arith.maxnumf %a, %p : f32
      scf.yield %b : f32
    }
    %ns = arith.addf %s, %s : f32
    memref.store %ns, %y[%r] : memref<1024xf32, 1>
    gpu.return
  }
}
