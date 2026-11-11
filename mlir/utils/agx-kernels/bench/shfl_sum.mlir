// Sums of SIMD-groups by an xor shuffle tree, stored by the first lane at
// the group's index i / 32 (a 64-bit shift pair kept, L15-E183).
gpu.module @k {
  gpu.func @shfl_sum(%a: memref<65536xf32, 1>, %o: memref<2048xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 256, 1, 1>} {
    %i = gpu.global_id x
    %v = memref.load %a[%i] : memref<65536xf32, 1>
    %w = arith.constant 32 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %s1, %p1 = gpu.shuffle xor %v, %c16, %w : f32
    %v1 = arith.addf %v, %s1 : f32
    %s2, %p2 = gpu.shuffle xor %v1, %c8, %w : f32
    %v2 = arith.addf %v1, %s2 : f32
    %s3, %p3 = gpu.shuffle xor %v2, %c4, %w : f32
    %v3 = arith.addf %v2, %s3 : f32
    %s4, %p4 = gpu.shuffle xor %v3, %c2, %w : f32
    %v4 = arith.addf %v3, %s4 : f32
    %s5, %p5 = gpu.shuffle xor %v4, %c1, %w : f32
    %v5 = arith.addf %v4, %s5 : f32
    %c32 = arith.constant 32 : index
    %g = arith.divui %i, %c32 : index
    %l = gpu.lane_id
    %c0 = arith.constant 0 : index
    %first = arith.cmpi eq, %l, %c0 : index
    scf.if %first {
      memref.store %v5, %o[%g] : memref<2048xf32, 1>
    }
    gpu.return
  }
}
