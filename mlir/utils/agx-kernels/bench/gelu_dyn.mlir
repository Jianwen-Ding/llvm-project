// agx-cost: --uniform 4=512
// y = gelu_tanh(x) over rows of a runtime length n (512 here), a thread per row.
gpu.module @k {
  gpu.func @gelu_dyn(%x: memref<1024x512xf32, 1>, %y: memref<1024x512xf32, 1>, %n: i32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.0 : f32
    %half = arith.constant 0.5 : f32
    %k0 = arith.constant 0.044715 : f32
    %k1 = arith.constant 0.7978845608 : f32
    %r = gpu.global_id x upper_bound 1024
    %ni = arith.index_cast %n : i32 to index
    scf.for %j = %c0 to %ni step %c1 {
      %a = memref.load %x[%r, %j] : memref<1024x512xf32, 1>
      %a2 = arith.mulf %a, %a : f32
      %a3 = arith.mulf %a2, %a : f32
      %t0 = arith.mulf %a3, %k0 : f32
      %t1 = arith.addf %a, %t0 : f32
      %t2 = arith.mulf %t1, %k1 : f32
      %th = math.tanh %t2 : f32
      %u = arith.addf %th, %one : f32
      %hh = arith.mulf %a, %half : f32
      %o = arith.mulf %hh, %u : f32
      memref.store %o, %y[%r, %j] : memref<1024x512xf32, 1>
    }
    gpu.return
  }
}
