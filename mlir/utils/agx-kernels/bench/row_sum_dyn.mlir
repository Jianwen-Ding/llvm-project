// agx-cost: --uniform 4=512
// Row sums over a runtime length n (512 here), a thread per row.
gpu.module @k {
  gpu.func @row_sum_dyn(%x: memref<1024x512xf32, 1>, %y: memref<1024xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant 0.0 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %s = scf.for %j = %c0 to %ni step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %b = arith.addf %a, %v : f32
      scf.yield %b : f32
    }
    memref.store %s, %y[%r] : memref<1024xf32, 1>
    gpu.return
  }
}
