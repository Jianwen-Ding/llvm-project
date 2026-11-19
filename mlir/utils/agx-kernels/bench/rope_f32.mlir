// agx-cost: --wg 64 --wg-id 300,0,0
// Rotary position embedding of 1024 tokens x 128 dims (64 pairs): a thread per pair,
// the angle position * inv_freq[i] through precise math.cos and math.sin.
gpu.module @k {
  gpu.func @rope_f32(%x: memref<1024x128xf32, 1>, %f: memref<64xf32, 1>, %y: memref<1024x128xf32, 1>) kernel
      attributes {known_block_size = array<i32: 64, 1, 1>} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %t = gpu.block_id x upper_bound 1024
    %i = gpu.thread_id x upper_bound 64
    %j0 = arith.muli %i, %c2 : index
    %j1 = arith.addi %j0, %c1 : index
    %ti = arith.index_cast %t : index to i32
    %tf = arith.sitofp %ti : i32 to f32
    %fr = memref.load %f[%i] : memref<64xf32, 1>
    %a = arith.mulf %tf, %fr : f32
    %c = math.cos %a : f32
    %s = math.sin %a : f32
    %x0 = memref.load %x[%t, %j0] : memref<1024x128xf32, 1>
    %x1 = memref.load %x[%t, %j1] : memref<1024x128xf32, 1>
    %p = arith.mulf %x0, %c : f32
    %q = arith.mulf %x1, %s : f32
    %y0 = arith.subf %p, %q : f32
    %u = arith.mulf %x0, %s : f32
    %v = arith.mulf %x1, %c : f32
    %y1 = arith.addf %u, %v : f32
    memref.store %y0, %y[%t, %j0] : memref<1024x128xf32, 1>
    memref.store %y1, %y[%t, %j1] : memref<1024x128xf32, 1>
    gpu.return
  }
}
