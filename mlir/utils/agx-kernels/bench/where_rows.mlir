// y = c ? a : b per element of rows, c a byte mask, a thread per row of 256.
gpu.module @k {
  gpu.func @where_rows(%c: memref<1024x256xi8, 1>, %a: memref<1024x256xf32, 1>, %b: memref<1024x256xf32, 1>, %y: memref<1024x256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %z = arith.constant 0 : i8
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %m = memref.load %c[%r, %j] : memref<1024x256xi8, 1>
      %av = memref.load %a[%r, %j] : memref<1024x256xf32, 1>
      %bv = memref.load %b[%r, %j] : memref<1024x256xf32, 1>
      %t = arith.cmpi ne, %m, %z : i8
      %o = arith.select %t, %av, %bv : f32
      memref.store %o, %y[%r, %j] : memref<1024x256xf32, 1>
    }
    gpu.return
  }
}
