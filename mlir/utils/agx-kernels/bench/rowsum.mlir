// Row sums of a 256 x 512 array, one thread per row: a loop over the columns
// (a byte-offset induction after LSR, L15-E171).
gpu.module @k {
  gpu.func @rowsum(%in: memref<256x512xf32, 1>, %out: memref<256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %zero = arith.constant 0.0 : f32
    %r = gpu.global_id x
    %s = scf.for %j = %c0 to %c512 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %in[%r, %j] : memref<256x512xf32, 1>
      %a = arith.addf %acc, %v : f32
      scf.yield %a : f32
    }
    memref.store %s, %out[%r] : memref<256xf32, 1>
    gpu.return
  }
}
