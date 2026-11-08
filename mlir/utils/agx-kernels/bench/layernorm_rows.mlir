// LayerNorm of 1024-float rows, a 256-thread threadgroup per row: four
// elements per thread, the row's sum and sum of squares by gpu.all_reduce
// (the dialect's own lowering: shuffles and a threadgroup buffer), then
// (x - mean) * rsqrt(var + eps) * gamma + beta.
// agx-cost: --wg 256
gpu.module @k {
  gpu.func @layernorm_rows(%x: memref<1024x1024xf32, 1>, %y: memref<1024x1024xf32, 1>,
                           %gamma: memref<1024xf32, 1>, %beta: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0.0 : f32
    %rn = arith.constant 9.765625e-04 : f32
    %eps = arith.constant 1.0e-05 : f32
    %row = gpu.block_id x upper_bound 1024
    %tx = gpu.thread_id x
    %s:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %zero, %q = %zero) -> (f32, f32) {
      %o = arith.muli %i, %c256 : index
      %j = arith.addi %o, %tx : index
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %a2 = arith.addf %a, %v : f32
      %q2 = math.fma %v, %v, %q : f32
      scf.yield %a2, %q2 : f32, f32
    }
    %sum = gpu.all_reduce add %s#0 uniform {} : (f32) -> (f32)
    %sq = gpu.all_reduce add %s#1 uniform {} : (f32) -> (f32)
    %mean = arith.mulf %sum, %rn : f32
    %ex2 = arith.mulf %sq, %rn : f32
    %m2 = arith.mulf %mean, %mean : f32
    %var = arith.subf %ex2, %m2 : f32
    %ve = arith.addf %var, %eps : f32
    %rs = math.rsqrt %ve : f32
    scf.for %i = %c0 to %c4 step %c1 {
      %o = arith.muli %i, %c256 : index
      %j = arith.addi %o, %tx : index
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %g = memref.load %gamma[%j] : memref<1024xf32, 1>
      %b = memref.load %beta[%j] : memref<1024xf32, 1>
      %d = arith.subf %v, %mean : f32
      %n = arith.mulf %d, %rs : f32
      %r = math.fma %n, %g, %b : f32
      memref.store %r, %y[%row, %j] : memref<1024x1024xf32, 1>
    }
    gpu.return
  }
}
