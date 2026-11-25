// ReLU6 of vector<4xf32> (arith.maximumf with 0, arith.minimumf with 6, as
// MLIR lowers a clamped activation): AIR has no maximum, so the portable
// lowering's maxnum carries a zero select and a NaN select (L15-E399).
// agx-cost: --wg 256
gpu.module @k {
  gpu.func @relu6_vec4(%x: memref<1048576xf32, 1>, %y: memref<1048576xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c4 = arith.constant 4 : index
    %z = arith.constant dense<0.0> : vector<4xf32>
    %six = arith.constant dense<6.0> : vector<4xf32>
    %g = gpu.global_id x upper_bound 262144
    %i = arith.muli %g, %c4 : index
    %v = vector.load %x[%i] : memref<1048576xf32, 1>, vector<4xf32>
    %a = arith.maximumf %v, %z : vector<4xf32>
    %b = arith.minimumf %a, %six : vector<4xf32>
    vector.store %b, %y[%i] : memref<1048576xf32, 1>, vector<4xf32>
    gpu.return
  }
}
