// LayerNorm, f32, a thread per row of 256: the mean and variance (two passes), then the
// normalized row with gamma / beta, whose loop loads three buffers and stores a fourth.
gpu.module @k {
  gpu.func @layernorm_thread(%x: memref<1024x256xf32, 1>, %g: memref<256xf32, 1>, %b: memref<256xf32, 1>, %y: memref<1024x256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %zf = arith.constant 0.0 : f32
    %n = arith.constant 256.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %r = gpu.global_id x upper_bound 1024
    %s = scf.for %j = %c0 to %c256 step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %a2 = arith.addf %a, %v : f32
      scf.yield %a2 : f32
    }
    %mean = arith.divf %s, %n : f32
    %q = scf.for %j = %c0 to %c256 step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %d = arith.subf %v, %mean : f32
      %a2 = math.fma %d, %d, %a : f32
      scf.yield %a2 : f32
    }
    %var = arith.divf %q, %n : f32
    %ve = arith.addf %var, %eps : f32
    %rs = math.rsqrt %ve : f32
    scf.for %j = %c0 to %c256 step %c1 {
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
