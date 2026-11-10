// A GEMM on the SIMD-group matrix unit: each of the four SIMD-groups of a
// 128-thread threadgroup accumulates one 8 x 8 tile of C (the threadgroup a
// 16 x 16 tile) over K in steps of 8 with air.simdgroup_matrix_load of A
// and B straight from device memory and
// air.simdgroup_matrix_multiply_accumulate. K is an argument (64 here:
// eight steps).
// agx-cost: --uniform 6=64 --wg 128
gpu.module @k {
  gpu.func @gemm_simd_matrix(%a: memref<1024x1024xf32, 1>, %b: memref<1024x1024xf32, 1>,
                             %c: memref<1024x1024xf32, 1>, %kdim: index) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %bx = gpu.block_id x upper_bound 64
    %by = gpu.block_id y upper_bound 64
    %sg = gpu.subgroup_id : index
    %sr = arith.divui %sg, %c2 : index
    %sc = arith.remui %sg, %c2 : index
    %r0 = arith.muli %by, %c16 : index
    %r1 = arith.muli %sr, %c8 : index
    %row = arith.addi %r0, %r1 : index
    %q0 = arith.muli %bx, %c16 : index
    %q1 = arith.muli %sc, %c8 : index
    %col = arith.addi %q0, %q1 : index
    %init = air.simdgroup_matrix_fill %zero : vector<64xf32>
    %acc = scf.for %k = %c0 to %kdim step %c8 iter_args(%t = %init) -> vector<64xf32> {
      %ma = air.simdgroup_matrix_load %a[%row, %k] : memref<1024x1024xf32, 1> -> vector<64xf32>
      %mb = air.simdgroup_matrix_load %b[%k, %col] : memref<1024x1024xf32, 1> -> vector<64xf32>
      %d = air.simdgroup_matrix_multiply_accumulate %ma, %mb, %t : vector<64xf32>
      scf.yield %d : vector<64xf32>
    }
    air.simdgroup_matrix_store %acc, %c[%row, %col] : vector<64xf32> -> memref<1024x1024xf32, 1>
    gpu.return
  }
}
