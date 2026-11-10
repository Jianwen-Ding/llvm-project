// A SIMD-group sum by butterfly shuffles (gpu.shuffle xor 16 .. 1), lane 0
// (The arrays have 2^20 elements: the global ids are below 2^20.)
// stores it: the reduction step of most GPU reductions.
gpu.module @k {
  gpu.func @warp_reduce(%in: memref<1048576xf32, 1>, %out: memref<1048576xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c0 = arith.constant 0 : index
    %gid = gpu.global_id x upper_bound 1048576
    %v = memref.load %in[%gid] : memref<1048576xf32, 1>
    %w = arith.constant 32 : i32
    %o16 = arith.constant 16 : i32
    %o8 = arith.constant 8 : i32
    %o4 = arith.constant 4 : i32
    %o2 = arith.constant 2 : i32
    %o1 = arith.constant 1 : i32
    %s16, %v16 = gpu.shuffle xor %v, %o16, %w : f32
    %a16 = arith.addf %v, %s16 : f32
    %s8, %v8 = gpu.shuffle xor %a16, %o8, %w : f32
    %a8 = arith.addf %a16, %s8 : f32
    %s4, %v4 = gpu.shuffle xor %a8, %o4, %w : f32
    %a4 = arith.addf %a8, %s4 : f32
    %s2, %v2 = gpu.shuffle xor %a4, %o2, %w : f32
    %a2 = arith.addf %a4, %s2 : f32
    %s1, %v1 = gpu.shuffle xor %a2, %o1, %w : f32
    %a1 = arith.addf %a2, %s1 : f32
    %lane = gpu.lane_id
    %first = arith.cmpi eq, %lane, %c0 : index
    scf.if %first {
      %b = gpu.block_id x
      memref.store %a1, %out[%b] : memref<1048576xf32, 1>
    }
    gpu.return
  }
}
