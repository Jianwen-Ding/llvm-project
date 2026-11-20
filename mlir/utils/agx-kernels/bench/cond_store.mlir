// y[r, j] = x[r, j] where x[r, j] > t (y untouched elsewhere), a thread per row of 256.
gpu.module @k {
  gpu.func @cond_store(%x: memref<1024x256xf32, 1>, %y: memref<1024x256xf32, 1>, %t: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %g = arith.cmpf ogt, %v, %t : f32
      scf.if %g {
        memref.store %v, %y[%r, %j] : memref<1024x256xf32, 1>
      }
    }
    gpu.return
  }
}
