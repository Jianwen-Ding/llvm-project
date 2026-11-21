// y = x * w[j], a thread per row
gpu.module @k {
  gpu.func @row_scale(%x: memref<1024x512xf32, 1>, %w: memref<512xf32, 1>, %z: memref<1024x512xf32, 1>, %y: memref<1024x512xf32, 1>, %s: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %one = arith.constant 1.0 : f32
    %half = arith.constant 0.5 : f32
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c512 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %wv = memref.load %w[%j] : memref<512xf32, 1>
      %o = arith.mulf %v, %wv : f32
      memref.store %o, %y[%r, %j] : memref<1024x512xf32, 1>
    }
    gpu.return
  }
}
