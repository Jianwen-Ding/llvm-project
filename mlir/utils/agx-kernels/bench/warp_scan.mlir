// An inclusive SIMD-group scan by shuffles up (1, 2, 4, 8, 16), where a
// lane below the offset keeps its value (the shuffle's valid flag).
gpu.module @k {
  gpu.func @warp_scan(%in: memref<1048576xi32, 1>, %out: memref<1048576xi32, 1>) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    %gid = gpu.global_id x upper_bound 1048576
    %v = memref.load %in[%gid] : memref<1048576xi32, 1>
    %w = arith.constant 32 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %z = arith.constant 0 : i32
    %s1, %ok1 = gpu.shuffle up %v, %c1, %w : i32
    %t1 = arith.select %ok1, %s1, %z : i32
    %a1 = arith.addi %v, %t1 : i32
    %s2, %ok2 = gpu.shuffle up %a1, %c2, %w : i32
    %t2 = arith.select %ok2, %s2, %z : i32
    %a2 = arith.addi %a1, %t2 : i32
    %s4, %ok4 = gpu.shuffle up %a2, %c4, %w : i32
    %t4 = arith.select %ok4, %s4, %z : i32
    %a4 = arith.addi %a2, %t4 : i32
    %s8, %ok8 = gpu.shuffle up %a4, %c8, %w : i32
    %t8 = arith.select %ok8, %s8, %z : i32
    %a8 = arith.addi %a4, %t8 : i32
    %s16, %ok16 = gpu.shuffle up %a8, %c16, %w : i32
    %t16 = arith.select %ok16, %s16, %z : i32
    %a16 = arith.addi %a8, %t16 : i32
    memref.store %a16, %out[%gid] : memref<1048576xi32, 1>
    gpu.return
  }
}
