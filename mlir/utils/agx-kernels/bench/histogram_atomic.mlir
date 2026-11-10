// A 256-bin byte histogram: bins cleared in threadgroup memory, a
// memref.atomic_rmw addi per byte into them, a barrier, then one device
// atomic_rmw addi per bin.
// agx-cost: --wg 256
gpu.module @k {
  gpu.func @histogram_atomic(%data: memref<1048576xi8, 1>, %hist: memref<256xi32, 1>)
      workgroup(%bins: memref<256xi32, 3>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %z = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %tx = gpu.thread_id x
    %gid = gpu.global_id x upper_bound 1048576
    memref.store %z, %bins[%tx] : memref<256xi32, 3>
    gpu.barrier
    %b = memref.load %data[%gid] : memref<1048576xi8, 1>
    %bi = arith.extui %b : i8 to i32
    %bx = arith.index_castui %bi : i32 to index
    %o = memref.atomic_rmw addi %one, %bins[%bx] : (i32, memref<256xi32, 3>) -> i32
    gpu.barrier
    %c = memref.load %bins[%tx] : memref<256xi32, 3>
    %o2 = memref.atomic_rmw addi %c, %hist[%tx] : (i32, memref<256xi32, 1>) -> i32
    gpu.return
  }
}
