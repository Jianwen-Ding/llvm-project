// A naive matrix product, one output per thread: a loop over k with two
// loads and an fma per step.
// agx-cost: --uniform 6=64
gpu.module @k {
  gpu.func @matmul(%a: memref<1048576xf32, 1>, %b: memref<1048576xf32, 1>,
                   %c: memref<1048576xf32, 1>, %n: index) kernel
      attributes {known_block_size = array<i32: 16, 16, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %col = gpu.global_id x upper_bound 1024
    %row = gpu.global_id y upper_bound 1024
    %rown = arith.muli %row, %n : index
    %s = scf.for %k = %c0 to %n step %c1 iter_args(%acc = %zero) -> f32 {
      %ia = arith.addi %rown, %k : index
      %kn = arith.muli %k, %n : index
      %ib = arith.addi %kn, %col : index
      %x = memref.load %a[%ia] : memref<1048576xf32, 1>
      %y = memref.load %b[%ib] : memref<1048576xf32, 1>
      %r = math.fma %x, %y, %acc : f32
      scf.yield %r : f32
    }
    %ic = arith.addi %rown, %col : index
    memref.store %s, %c[%ic] : memref<1048576xf32, 1>
    gpu.return
  }
}
