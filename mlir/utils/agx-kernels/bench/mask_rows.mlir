// Row masks: m[r, j] = x[r, j] > t[r] as a byte, a thread per row of 256.
gpu.module @k {
  gpu.func @mask_rows(%x: memref<1024x256xf32, 1>, %t: memref<1024xf32, 1>, %m: memref<1024x256xi8, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %r = gpu.global_id x upper_bound 1024
    %th = memref.load %t[%r] : memref<1024xf32, 1>
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
      %c = arith.cmpf ogt, %v, %th : f32
      %b = arith.extui %c : i1 to i8
      memref.store %b, %m[%r, %j] : memref<1024x256xi8, 1>
    }
    gpu.return
  }
}
