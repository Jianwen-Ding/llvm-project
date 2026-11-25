// agx-cost: --uniform 6=512
// Rows split into sums and differences with a second row, two outputs, a runtime length, a thread per row.
gpu.module @k {
  gpu.func @split_dyn(%x: memref<1024x512xf32, 1>, %y: memref<1024x512xf32, 1>, %z: memref<1024x512xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zf = arith.constant 0.0 : f32
    %lo = arith.constant -1.0 : f32
    %hi = arith.constant 1.0 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %p = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %q = memref.load %y[%r, %j] : memref<1024x512xf32, 1>
      %a = arith.addf %p, %q : f32
      %b = arith.subf %p, %q : f32
      memref.store %a, %z[%r, %j] : memref<1024x512xf32, 1>
      memref.store %b, %y[%r, %j] : memref<1024x512xf32, 1>
    }
    gpu.return
  }
}
