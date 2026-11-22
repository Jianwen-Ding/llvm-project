// agx-cost: --grid 32 --wg 32
// C = A B, 32x32x32: f16 inputs, f32 accumulators, as gpu.subgroup_mma 16x16 tiles on one SIMD-group (each tile four 8x8 AIR matrices).
gpu.module @k {
  gpu.func @k(%a: memref<32x32xf16, 1>, %b: memref<32x32xf16, 1>, %c: memref<32x32xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %z = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c32 step %c16 {
      scf.for %j = %c0 to %c32 step %c16 {
        %acc0 = gpu.subgroup_mma_constant_matrix %z : !gpu.mma_matrix<16x16xf32, "COp">
        %acc = scf.for %k = %c0 to %c32 step %c16 iter_args(%s = %acc0) -> (!gpu.mma_matrix<16x16xf32, "COp">) {
          %A = gpu.subgroup_mma_load_matrix %a[%i, %k] leadDimension 32 : memref<32x32xf16, 1> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %B = gpu.subgroup_mma_load_matrix %b[%k, %j] leadDimension 32: memref<32x32xf16, 1> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %D = gpu.subgroup_mma_compute %A, %B, %s : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          scf.yield %D : !gpu.mma_matrix<16x16xf32, "COp">
        }
        gpu.subgroup_mma_store_matrix %acc, %c[%i, %j] leadDimension 32 : !gpu.mma_matrix<16x16xf32, "COp">, memref<32x32xf32, 1>
      }
    }
    gpu.return
  }
}
