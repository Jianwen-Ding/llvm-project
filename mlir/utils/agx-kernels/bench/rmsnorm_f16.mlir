// RMSNorm of 1024-half rows, a SIMD-group per row, vector<4xf16> loads in f32:
// y = x * rsqrt(mean(x^2) + eps) * w.
gpu.module @k {
  gpu.func @rmsnorm_f16(%x: memref<1024x1024xf16, 1>, %w: memref<1024xf16, 1>, %y: memref<1024x1024xf16, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %zf = arith.constant dense<0.0> : vector<4xf32>
    %rn = arith.constant 9.765625e-04 : f32
    %eps = arith.constant 1.0e-06 : f32
    %row = gpu.block_id x upper_bound 1024
    %lane = gpu.lane_id upper_bound 32
    %l4 = arith.muli %lane, %c4 : index
    %s = scf.for %j = %l4 to %c1024 step %c128 iter_args(%a = %zf) -> vector<4xf32> {
      %h = vector.load %x[%row, %j] : memref<1024x1024xf16, 1>, vector<4xf16>
      %v = arith.extf %h : vector<4xf16> to vector<4xf32>
      %q = math.fma %v, %v, %a : vector<4xf32>
      scf.yield %q : vector<4xf32>
    }
    %t = vector.reduction <add>, %s : vector<4xf32> into f32
    %ss = gpu.subgroup_reduce add %t : (f32) -> f32
    %ms = arith.mulf %ss, %rn : f32
    %me = arith.addf %ms, %eps : f32
    %r = math.rsqrt %me : f32
    %rv = vector.broadcast %r : f32 to vector<4xf32>
    scf.for %j = %l4 to %c1024 step %c128 {
      %h = vector.load %x[%row, %j] : memref<1024x1024xf16, 1>, vector<4xf16>
      %g = vector.load %w[%j] : memref<1024xf16, 1>, vector<4xf16>
      %v = arith.extf %h : vector<4xf16> to vector<4xf32>
      %gv = arith.extf %g : vector<4xf16> to vector<4xf32>
      %n = arith.mulf %v, %rv : vector<4xf32>
      %o = arith.mulf %n, %gv : vector<4xf32>
      %oh = arith.truncf %o : vector<4xf32> to vector<4xf16>
      vector.store %oh, %y[%row, %j] : memref<1024x1024xf16, 1>, vector<4xf16>
    }
    gpu.return
  }
}
