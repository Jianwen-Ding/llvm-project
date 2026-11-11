// A 256-bin histogram of the low bytes of words: a device atomic at a byte
// index (L15-E179).
gpu.module @k {
  gpu.func @hist(%a: memref<65536xi32, 1>, %h: memref<256xi32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 256, 1, 1>} {
    %c1 = arith.constant 1 : i32
    %c255 = arith.constant 255 : i32
    %i = gpu.global_id x
    %v = memref.load %a[%i] : memref<65536xi32, 1>
    %b = arith.andi %v, %c255 : i32
    %bi = arith.index_castui %b : i32 to index
    %o = memref.atomic_rmw addi %c1, %h[%bi] : (i32, memref<256xi32, 1>) -> i32
    gpu.return
  }
}
