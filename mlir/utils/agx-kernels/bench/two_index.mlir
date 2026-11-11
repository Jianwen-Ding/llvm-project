// A dot product of a row of a at stride 2 and a row of b, one thread per row:
// two accesses whose indices step 2 and 1 (L15-E175).
gpu.module @k {
  gpu.func @two_index(%a: memref<65536xf32, 1>, %b: memref<65536xf32, 1>, %out: memref<256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0.0 : f32
    %r = gpu.global_id x
    %s = scf.for %j = %c0 to %c128 step %c1 iter_args(%acc = %zero) -> f32 {
      %j2 = arith.muli %j, %c2 : index
      %p = arith.muli %r, %c256 : index
      %ia = arith.addi %p, %j2 : index
      %ib = arith.addi %p, %j : index
      %x = memref.load %a[%ia] : memref<65536xf32, 1>
      %y = memref.load %b[%ib] : memref<65536xf32, 1>
      %m = math.fma %x, %y, %acc : f32
      scf.yield %m : f32
    }
    memref.store %s, %out[%r] : memref<256xf32, 1>
    gpu.return
  }
}
