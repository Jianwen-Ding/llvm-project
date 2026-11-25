// A column sum of a 512 x 1024 matrix, a thread per column: a constant trip
// of 512 rows, unrolled by 8 before llc (opt -O2 with the AGX unrolling
// preferences), whose down-counter by 8 beside the row index goes (L15-E439).
// agx-cost: --wg 256
gpu.module @k {
  gpu.func @colsum(%x: memref<512x1024xf32, 1>, %y: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %zf = arith.constant 0.0 : f32
    %j = gpu.global_id x upper_bound 1024
    %s = scf.for %i = %c0 to %c512 step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%i, %j] : memref<512x1024xf32, 1>
      %n = arith.addf %a, %v : f32
      scf.yield %n : f32
    }
    memref.store %s, %y[%j] : memref<1024xf32, 1>
    gpu.return
  }
}
