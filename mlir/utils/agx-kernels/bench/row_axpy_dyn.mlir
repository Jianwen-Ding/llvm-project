// agx-cost: --uniform 6=512 --uniform 7=1069547520
// z = a x + y over rows of a runtime length n (512 here), a thread per row.
gpu.module @k {
  gpu.func @row_axpy_dyn(%x: memref<1024x512xf32, 1>, %y: memref<1024x512xf32, 1>, %z: memref<1024x512xf32, 1>, %n: i32, %a: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %w = memref.load %y[%r, %j] : memref<1024x512xf32, 1>
      %o = math.fma %a, %v, %w : f32
      memref.store %o, %z[%r, %j] : memref<1024x512xf32, 1>
    }
    gpu.return
  }
}
