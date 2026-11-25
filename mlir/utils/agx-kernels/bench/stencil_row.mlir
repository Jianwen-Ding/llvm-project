// y[j] = 0.25 x[j-1] + 0.5 x[j] + 0.25 x[j+1] over each 258-float row (256 outputs), a thread per row.
gpu.module @k {
  gpu.func @stencil_row(%x: memref<1024x258xf32, 1>, %y: memref<1024x256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %q = arith.constant 0.25 : f32
    %h = arith.constant 0.5 : f32
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %j1 = arith.addi %j, %c1 : index
      %j2 = arith.addi %j, %c2 : index
      %a = memref.load %x[%r, %j] : memref<1024x258xf32, 1>
      %b = memref.load %x[%r, %j1] : memref<1024x258xf32, 1>
      %c = memref.load %x[%r, %j2] : memref<1024x258xf32, 1>
      %s0 = arith.mulf %a, %q : f32
      %s1 = math.fma %b, %h, %s0 : f32
      %s2 = math.fma %c, %q, %s1 : f32
      memref.store %s2, %y[%r, %j] : memref<1024x256xf32, 1>
    }
    gpu.return
  }
}
