// A row copy by vector<4xf32>, one thread per row: an scf.for of step 4
// (LSR counts it from -4: a signed 64-bit counter, L15-E174).
gpu.module @k {
  gpu.func @rowcopy_vec4(%a: memref<256x512xf32, 1>, %b: memref<256x512xf32, 1>) kernel
      attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c512 = arith.constant 512 : index
    %r = gpu.global_id x
    scf.for %j = %c0 to %c512 step %c4 {
      %v = vector.load %a[%r, %j] : memref<256x512xf32, 1>, vector<4xf32>
      vector.store %v, %b[%r, %j] : memref<256x512xf32, 1>, vector<4xf32>
    }
    gpu.return
  }
}
