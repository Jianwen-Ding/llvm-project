// agx-cost: --uniform 4=512
// Row sums of bytes into ints over rows of a runtime length, a thread per row.
gpu.module @k {
  gpu.func @bsum_dyn(%x: memref<1024x512xi8, 1>, %y: memref<1024xi32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant 0 : i32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    %s = scf.for %j = %c0 to %ni step %c1 iter_args(%a = %zf) -> i32 {
      %p = memref.load %x[%r, %j] : memref<1024x512xi8, 1>
      %q = arith.extsi %p : i8 to i32
      %b = arith.addi %a, %q : i32
      scf.yield %b : i32
    }
    memref.store %s, %y[%r] : memref<1024xi32, 1>
    gpu.return
  }
}
