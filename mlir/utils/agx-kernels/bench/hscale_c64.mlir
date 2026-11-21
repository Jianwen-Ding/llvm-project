// half rows halved, a thread per row of 64.
gpu.module @k {
  gpu.func @hscale_c64(%x: memref<4096x64xf16, 1>, %y: memref<4096x64xf16, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cT = arith.constant 64 : index
    %r = gpu.global_id x upper_bound 4096
    scf.for %j = %c0 to %cT step %c1 {
      %v = memref.load %x[%r, %j] : memref<4096x64xf16, 1>
      %h = arith.constant 0.5 : f16
      %o = arith.mulf %v, %h : f16
      memref.store %o, %y[%r, %j] : memref<4096x64xf16, 1>
    }
    gpu.return
  }
}
