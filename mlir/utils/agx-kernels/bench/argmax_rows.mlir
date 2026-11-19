// A thread's argmax over its row of 256 floats (the first maximum's index):
// the loop's 64-bit load index, narrowed, and the index_cast's i32 copy of
// the same counter are one register (L15-E431).
// agx-cost: --wg 256
gpu.module @k {
  gpu.func @argmax_rows(%x: memref<1024x256xf32, 1>, %out: memref<1024xi32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %ninf = arith.constant 0xFF800000 : f32
    %z = arith.constant 0 : i32
    %r = gpu.global_id x upper_bound 1024
    %res:2 = scf.for %j = %c0 to %c256 step %c1 iter_args(%m = %ninf, %bi = %z) -> (f32, i32) {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %gt = arith.cmpf ogt, %v, %m : f32
      %ji = arith.index_cast %j : index to i32
      %nm = arith.select %gt, %v, %m : f32
      %nb = arith.select %gt, %ji, %bi : i32
      scf.yield %nm, %nb : f32, i32
    }
    memref.store %res#1, %out[%r] : memref<1024xi32, 1>
    gpu.return
  }
}
