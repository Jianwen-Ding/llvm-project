// y = a * x + y by a grid-stride loop over n elements (n = 4096 here: four
// steps of a grid of 4 x 256 threads); the loop's induction variable is the
// accesses' index.
// agx-cost: --uniform 5=4096 --uniform 7=4
gpu.module @k {
  gpu.func @grid_stride(%x: memref<1048576xf32, 1>, %y: memref<1048576xf32, 1>,
                        %a: f32, %n: index) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %gid = gpu.global_id x upper_bound 1048576
    %bd = gpu.block_dim x
    %gd = gpu.grid_dim x
    %stride = arith.muli %bd, %gd : index
    scf.for %i = %gid to %n step %stride {
      %xv = memref.load %x[%i] : memref<1048576xf32, 1>
      %yv = memref.load %y[%i] : memref<1048576xf32, 1>
      %r = math.fma %a, %xv, %yv : f32
      memref.store %r, %y[%i] : memref<1048576xf32, 1>
    }
    gpu.return
  }
}
