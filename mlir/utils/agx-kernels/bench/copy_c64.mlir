// y = 2x, a thread per row of 64.
gpu.module @k {
  gpu.func @copy_c64(%x: memref<4096x64xf32, 1>, %y: memref<4096x64xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cT = arith.constant 64 : index
    %r = gpu.global_id x upper_bound 4096
    scf.for %j = %c0 to %cT step %c1 {
      %v = memref.load %x[%r, %j] : memref<4096x64xf32, 1>
      %o = arith.addf %v, %v : f32
      memref.store %o, %y[%r, %j] : memref<4096x64xf32, 1>
    }
    gpu.return
  }
}
