// vector<2xf16> loads and stores at element 2 i (2-byte aligned: split in
// halves), a multiply-add (L15-E190).
gpu.module @k {
  gpu.func @h2(%a: memref<65536xf16, 1>, %b: memref<65536xf16, 1>, %o: memref<65536xf16, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 128, 1, 1>} {
    %i = gpu.global_id x
    %c2 = arith.constant 2 : index
    %j = arith.muli %i, %c2 : index
    %x = vector.load %a[%j] : memref<65536xf16, 1>, vector<2xf16>
    %y = vector.load %b[%j] : memref<65536xf16, 1>, vector<2xf16>
    %m = arith.mulf %x, %y : vector<2xf16>
    %s = arith.addf %m, %x : vector<2xf16>
    vector.store %s, %o[%j] : memref<65536xf16, 1>, vector<2xf16>
    gpu.return
  }
}
