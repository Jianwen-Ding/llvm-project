// agx-cost: --uniform 6=512
// Row dots of halves with a half vector, into floats, over rows of a runtime length, a thread per row.
gpu.module @k {
  gpu.func @hdot_dyn(%x: memref<1024x512xf16, 1>, %w: memref<512xf16, 1>, %y: memref<1024xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant 0.0 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %s = scf.for %j = %c0 to %ni step %c1 iter_args(%a = %zf) -> f32 {
      %p = memref.load %x[%r, %j] : memref<1024x512xf16, 1>
      %q = arith.extf %p : f16 to f32
      %v = memref.load %w[%j] : memref<512xf16, 1>
      %u = arith.extf %v : f16 to f32
      %b = math.fma %q, %u, %a : f32
      scf.yield %b : f32
    }
    memref.store %s, %y[%r] : memref<1024xf32, 1>
    gpu.return
  }
}
