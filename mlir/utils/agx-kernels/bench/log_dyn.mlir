// agx-cost: --uniform 4=512
// y = log(x) + x over rows of a runtime length n (512 here), a thread per row.
gpu.module @k {
  gpu.func @log_dyn(%x: memref<1024x512xf32, 1>, %y: memref<1024x512xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.0 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %a = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %e = math.log %a : f32
      %o = arith.addf %e, %a : f32
      memref.store %o, %y[%r, %j] : memref<1024x512xf32, 1>
    }
    gpu.return
  }
}

