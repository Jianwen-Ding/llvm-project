// An FIR filter / 1-D convolution with a runtime tap count: y[i] = sum_k
// w[k] x[i + k] for k < K (here 9), a thread per output. x[i + k] and w[k]
// share the counter k as their index (L15-E398).
// agx-cost: --uniform 6=9 --wg 256
gpu.module @k {
  gpu.func @fir_taps(%x: memref<65600xf32, 1>, %w: memref<64xf32, 1>, %y: memref<65536xf32, 1>, %kk: index) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %i = gpu.global_id x upper_bound 65536
    %s = scf.for %k = %c0 to %kk step %c1 iter_args(%a = %zero) -> f32 {
      %ik = arith.addi %i, %k : index
      %xv = memref.load %x[%ik] : memref<65600xf32, 1>
      %wv = memref.load %w[%k] : memref<64xf32, 1>
      %m = math.fma %xv, %wv, %a : f32
      scf.yield %m : f32
    }
    memref.store %s, %y[%i] : memref<65536xf32, 1>
    gpu.return
  }
}
