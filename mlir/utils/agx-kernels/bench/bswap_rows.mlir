// Byte-swap each 32-bit word of the rows, a thread per row of 256.
gpu.module @k {
  gpu.func @bswap_rows(%x: memref<1024x256xi32, 1>, %y: memref<1024x256xi32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c8 = arith.constant 8 : i32
    %c24 = arith.constant 24 : i32
    %m1 = arith.constant 65280 : i32
    %m2 = arith.constant 16711680 : i32
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xi32, 1>
      %a = arith.shli %v, %c24 : i32
      %b0 = arith.shli %v, %c8 : i32
      %b = arith.andi %b0, %m2 : i32
      %d0 = arith.shrui %v, %c8 : i32
      %d = arith.andi %d0, %m1 : i32
      %e = arith.shrui %v, %c24 : i32
      %o1 = arith.ori %a, %b : i32
      %o2 = arith.ori %d, %e : i32
      %o = arith.ori %o1, %o2 : i32
      memref.store %o, %y[%r, %j] : memref<1024x256xi32, 1>
    }
    gpu.return
  }
}
