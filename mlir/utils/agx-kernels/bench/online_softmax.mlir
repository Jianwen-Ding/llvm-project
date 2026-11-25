// One-pass (online) softmax normalizer of 1024-float rows by a SIMD-group: a running
// maximumf m and a running sum s * e^(m - m') + e^(x - m'), merged across lanes.
gpu.module @k {
  gpu.func @online_softmax(%x: memref<1024x1024xf32, 1>, %lse: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zf = arith.constant 0.0 : f32
    %row = gpu.block_id x upper_bound 1024
    %lane = gpu.lane_id upper_bound 32
    %r:2 = scf.for %j = %lane to %c1024 step %c32 iter_args(%m = %ninf, %s = %zf) -> (f32, f32) {
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %m2 = arith.maximumf %m, %v : f32
      %d = arith.subf %m, %m2 : f32
      %sc = math.exp %d : f32
      %e = arith.subf %v, %m2 : f32
      %ev = math.exp %e : f32
      %s2 = math.fma %s, %sc, %ev : f32
      scf.yield %m2, %s2 : f32, f32
    }
    %mm = gpu.subgroup_reduce maximumf %r#0 : (f32) -> f32
    %d = arith.subf %r#0, %mm : f32
    %sc = math.exp %d : f32
    %t = arith.mulf %r#1, %sc : f32
    %ss = gpu.subgroup_reduce add %t : (f32) -> f32
    %lg = math.log %ss : f32
    %o = arith.addf %lg, %mm : f32
    memref.store %o, %lse[%row] : memref<1024xf32, 1>
    gpu.return
  }
}
