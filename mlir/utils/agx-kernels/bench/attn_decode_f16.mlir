// agx-cost: --wg 32 --wg-id 7,0,0
// Attention decode for one query per head over 512 cached f16 keys / values of 128 dims:
// a SIMD-group per head, four dims per lane (vector<4xf16> loads, f32 math), scores by
// gpu.subgroup_reduce add, an online softmax (maximumf, the rescale e^(m - m')) and the
// rescaled weighted sum of values, normalized at the end.
gpu.module @k {
  gpu.func @attn_decode_f16(%q: memref<32x128xf16, 1>, %kc: memref<32x512x128xf16, 1>,
                            %vc: memref<32x512x128xf16, 1>, %o: memref<32x128xf16, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c512 = arith.constant 512 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zf = arith.constant 0.0 : f32
    %zv = arith.constant dense<0.0> : vector<4xf32>
    %scale = arith.constant 0.08838834764 : f32
    %h = gpu.block_id x upper_bound 32
    %lane = gpu.lane_id upper_bound 32
    %d = arith.muli %lane, %c4 : index
    %qh = vector.load %q[%h, %d] : memref<32x128xf16, 1>, vector<4xf16>
    %qv = arith.extf %qh : vector<4xf16> to vector<4xf32>
    %r:3 = scf.for %j = %c0 to %c512 step %c1 iter_args(%m = %ninf, %s = %zf, %acc = %zv) -> (f32, f32, vector<4xf32>) {
      %kh = vector.load %kc[%h, %j, %d] : memref<32x512x128xf16, 1>, vector<4xf16>
      %kv = arith.extf %kh : vector<4xf16> to vector<4xf32>
      %p = arith.mulf %qv, %kv : vector<4xf32>
      %ps = vector.reduction <add>, %p : vector<4xf32> into f32
      %dot = gpu.subgroup_reduce add %ps : (f32) -> f32
      %sc = arith.mulf %dot, %scale : f32
      %m2 = arith.maximumf %m, %sc : f32
      %d1 = arith.subf %m, %m2 : f32
      %e1 = math.exp %d1 : f32
      %d2 = arith.subf %sc, %m2 : f32
      %e2 = math.exp %d2 : f32
      %s2 = math.fma %s, %e1, %e2 : f32
      %vh = vector.load %vc[%h, %j, %d] : memref<32x512x128xf16, 1>, vector<4xf16>
      %vv = arith.extf %vh : vector<4xf16> to vector<4xf32>
      %e1v = vector.broadcast %e1 : f32 to vector<4xf32>
      %e2v = vector.broadcast %e2 : f32 to vector<4xf32>
      %t = arith.mulf %vv, %e2v : vector<4xf32>
      %acc2 = math.fma %acc, %e1v, %t : vector<4xf32>
      scf.yield %m2, %s2, %acc2 : f32, f32, vector<4xf32>
    }
    %one = arith.constant 1.0 : f32
    %inv = arith.divf %one, %r#1 : f32
    %iv = vector.broadcast %inv : f32 to vector<4xf32>
    %out = arith.mulf %r#2, %iv : vector<4xf32>
    %oh = arith.truncf %out : vector<4xf32> to vector<4xf16>
    vector.store %oh, %o[%h, %d] : memref<32x128xf16, 1>, vector<4xf16>
    gpu.return
  }
}
