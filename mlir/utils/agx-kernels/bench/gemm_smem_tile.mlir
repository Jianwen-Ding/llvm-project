// A row-major 2-D GEMM, C = A B, with 16 x 16 tiles of A and B staged in
// threadgroup memory: each step loads one element of each tile per thread,
// waits at a barrier, runs 16 fmas from threadgroup memory and waits again.
// The K loop's bound is an argument (K = 64: four tiles).
// agx-cost: --uniform 6=64 --wg 16,16,1
gpu.module @k {
  gpu.func @gemm_smem_tile(%a: memref<1024x1024xf32, 1>, %b: memref<1024x1024xf32, 1>,
                           %c: memref<1024x1024xf32, 1>, %kdim: index)
      workgroup(%ta: memref<16x16xf32, 3>, %tb: memref<16x16xf32, 3>) kernel
      attributes {known_block_size = array<i32: 16, 16, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %tx = gpu.thread_id x
    %ty = gpu.thread_id y
    %col = gpu.global_id x upper_bound 1024
    %row = gpu.global_id y upper_bound 1024
    %s = scf.for %k0 = %c0 to %kdim step %c16 iter_args(%acc = %zero) -> f32 {
      %ka = arith.addi %k0, %tx : index
      %va = memref.load %a[%row, %ka] : memref<1024x1024xf32, 1>
      memref.store %va, %ta[%ty, %tx] : memref<16x16xf32, 3>
      %kb = arith.addi %k0, %ty : index
      %vb = memref.load %b[%kb, %col] : memref<1024x1024xf32, 1>
      memref.store %vb, %tb[%ty, %tx] : memref<16x16xf32, 3>
      gpu.barrier
      %r = scf.for %kk = %c0 to %c16 step %c1 iter_args(%t = %acc) -> f32 {
        %x = memref.load %ta[%ty, %kk] : memref<16x16xf32, 3>
        %y = memref.load %tb[%kk, %tx] : memref<16x16xf32, 3>
        %f = math.fma %x, %y, %t : f32
        scf.yield %f : f32
      }
      gpu.barrier
      scf.yield %r : f32
    }
    memref.store %s, %c[%row, %col] : memref<1024x1024xf32, 1>
    gpu.return
  }
}
