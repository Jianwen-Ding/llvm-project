// agx-cost: --uniform 4=512 --uniform 5=1069547520
// int32 rows plus a constant, a runtime length, a thread per row.
gpu.module @k {
  gpu.func @iadd_i32_dyn(%x: memref<1024x512xi32, 1>, %y: memref<1024x512xi32, 1>, %n: i32, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x512xi32, 1>
      %k = arith.constant 7 : i32
      %o = arith.addi %v, %k : i32
      memref.store %o, %y[%r, %j] : memref<1024x512xi32, 1>
    }
    gpu.return
  }
}
