// A binary search per thread (std::lower_bound of keys[gid] in a sorted
// array of N keys, N an argument, here 4096) as an scf.while: twelve trips
// of a data-dependent, divergent loop with a load per trip.
// agx-cost: --uniform 6=4096 --wg 256
gpu.module @k {
  gpu.func @bsearch_while(%keys: memref<1048576xi32, 1>, %sorted: memref<1048576xi32, 1>,
                          %pos: memref<1048576xi32, 1>, %n: index) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %gid = gpu.global_id x upper_bound 1048576
    %key = memref.load %keys[%gid] : memref<1048576xi32, 1>
    %r:2 = scf.while (%lo = %c0, %hi = %n) : (index, index) -> (index, index) {
      %go = arith.cmpi ult, %lo, %hi : index
      scf.condition(%go) %lo, %hi : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %s = arith.addi %l, %h : index
      %mid = arith.shrui %s, %c1 : index
      %v = memref.load %sorted[%mid] : memref<1048576xi32, 1>
      %lt = arith.cmpi slt, %v, %key : i32
      %m1 = arith.addi %mid, %c1 : index
      %nl = arith.select %lt, %m1, %l : index
      %nh = arith.select %lt, %h, %mid : index
      scf.yield %nl, %nh : index, index
    }
    %p = arith.index_cast %r#0 : index to i32
    memref.store %p, %pos[%gid] : memref<1048576xi32, 1>
    gpu.return
  }
}
