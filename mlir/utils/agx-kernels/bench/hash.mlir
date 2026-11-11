// A 32-bit hash's mix (MurmurHash3's multiplies and rotates, then an xor
// shift): rotates of products (L15-E189).
gpu.module @k {
  gpu.func @hash(%a: memref<65536xi32, 1>, %o: memref<65536xi32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 256, 1, 1>} {
    %i = gpu.global_id x
    %v = memref.load %a[%i] : memref<65536xi32, 1>
    %c1 = arith.constant -862048943 : i32
    %c2 = arith.constant 461845907 : i32
    %c15 = arith.constant 15 : i32
    %c17 = arith.constant 17 : i32
    %c13 = arith.constant 13 : i32
    %c19 = arith.constant 19 : i32
    %c16 = arith.constant 16 : i32
    %k1 = arith.muli %v, %c1 : i32
    %l = arith.shli %k1, %c15 : i32
    %r = arith.shrui %k1, %c17 : i32
    %rot = arith.ori %l, %r : i32
    %k2 = arith.muli %rot, %c2 : i32
    %l2 = arith.shli %k2, %c13 : i32
    %r2 = arith.shrui %k2, %c19 : i32
    %rot2 = arith.ori %l2, %r2 : i32
    %s = arith.shrui %rot2, %c16 : i32
    %x = arith.xori %rot2, %s : i32
    memref.store %x, %o[%i] : memref<65536xi32, 1>
    gpu.return
  }
}
