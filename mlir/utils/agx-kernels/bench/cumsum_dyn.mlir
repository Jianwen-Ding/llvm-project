// agx-cost: --uniform 6=512
// Row prefix sums over rows of a runtime length, a thread per row.
gpu.module @k {
  gpu.func @cumsum_dyn(%x: memref<1024x512xf32, 1>, %y: memref<1024x512xf32, 1>, %z: memref<1024x512xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant 0.0 : f32
    %lo = arith.constant -1.0 : f32
    %hi = arith.constant 1.0 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %s = scf.for %j = %c0 to %ni step %c1 iter_args(%a = %zf) -> f32 {
      %p = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %b = arith.addf %a, %p : f32
      memref.store %b, %y[%r, %j] : memref<1024x512xf32, 1>
      scf.yield %b : f32
    }
    gpu.return
  }
}
