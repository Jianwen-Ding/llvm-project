// One-pass softmax normalizer (log-sum-exp) of 1024-float rows by a thread, vector<4xf32>
// steps: four running maximumf / sums, combined after the loop.
gpu.module @k {
  gpu.func @online_softmax_vec4(%x: memref<1024x1024xf32, 1>, %lse: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %ninf = arith.constant dense<0xFF800000> : vector<4xf32>
    %zf = arith.constant dense<0.0> : vector<4xf32>
    %r = gpu.global_id x upper_bound 1024
    %res:2 = scf.for %j = %c0 to %c1024 step %c4 iter_args(%m = %ninf, %s = %zf) -> (vector<4xf32>, vector<4xf32>) {
      %v = vector.load %x[%r, %j] : memref<1024x1024xf32, 1>, vector<4xf32>
      %m2 = arith.maximumf %m, %v : vector<4xf32>
      %d = arith.subf %m, %m2 : vector<4xf32>
      %sc = math.exp %d : vector<4xf32>
      %e = arith.subf %v, %m2 : vector<4xf32>
      %ev = math.exp %e : vector<4xf32>
      %s2 = math.fma %s, %sc, %ev : vector<4xf32>
      scf.yield %m2, %s2 : vector<4xf32>, vector<4xf32>
    }
    %mm = vector.reduction <maximumf>, %res#0 : vector<4xf32> into f32
    %mv = vector.broadcast %mm : f32 to vector<4xf32>
    %d = arith.subf %res#0, %mv : vector<4xf32>
    %sc = math.exp %d : vector<4xf32>
    %t = arith.mulf %res#1, %sc : vector<4xf32>
    %ss = vector.reduction <add>, %t : vector<4xf32> into f32
    %lg = math.log %ss : f32
    %o = arith.addf %lg, %mm : f32
    memref.store %o, %lse[%r] : memref<1024xf32, 1>
    gpu.return
  }
}
