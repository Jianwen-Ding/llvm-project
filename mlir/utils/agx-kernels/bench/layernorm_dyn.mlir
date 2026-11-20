// agx-cost: --uniform 8=256
// LayerNorm over rows of a runtime length n (256 here), a thread per row: mean, variance, normalize with gamma / beta.
gpu.module @k {
  gpu.func @layernorm_dyn(%x: memref<1024x256xf32, 1>, %g: memref<256xf32, 1>, %b: memref<256xf32, 1>, %y: memref<1024x256xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %nf = arith.sitofp %n : i32 to f32
    %s = scf.for %j = %c0 to %ni step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %a2 = arith.addf %a, %v : f32
      scf.yield %a2 : f32
    }
    %mean = arith.divf %s, %nf : f32
    %q = scf.for %j = %c0 to %ni step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %d = arith.subf %v, %mean : f32
      %a2 = math.fma %d, %d, %a : f32
      scf.yield %a2 : f32
    }
    %var = arith.divf %q, %nf : f32
    %ve = arith.addf %var, %eps : f32
    %rs = math.rsqrt %ve : f32
    scf.for %j = %c0 to %ni step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %d = arith.subf %v, %mean : f32
      %nv = arith.mulf %d, %rs : f32
      %gv = memref.load %g[%j] : memref<256xf32, 1>
      %bv = memref.load %b[%j] : memref<256xf32, 1>
      %o = math.fma %nv, %gv, %bv : f32
      memref.store %o, %y[%r, %j] : memref<1024x256xf32, 1>
    }
    gpu.return
  }
}
