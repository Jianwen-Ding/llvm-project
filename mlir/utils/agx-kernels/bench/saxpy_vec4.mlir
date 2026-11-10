// y = a x + y on vector<4xf32> (16-byte vector.load / vector.store), a
// grid-stride loop over N / 4 vectors, N an argument (16384 here: with
// a 1024-thread grid, four iterations per thread).
// agx-cost: --uniform 5=16384 --uniform 7=4 --grid 1024 --wg 256
gpu.module @k {
  gpu.func @saxpy_vec4(%x: memref<1048576xf32, 1>, %y: memref<1048576xf32, 1>,
                       %a: f32, %n: index) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c4 = arith.constant 4 : index
    %gid = gpu.global_id x upper_bound 262144
    %g = gpu.grid_dim x
    %bd = gpu.block_dim x
    %t = arith.muli %g, %bd : index
    %stride = arith.muli %t, %c4 : index
    %i0 = arith.muli %gid, %c4 : index
    %va = vector.broadcast %a : f32 to vector<4xf32>
    scf.for %i = %i0 to %n step %stride {
      %vx = vector.load %x[%i] : memref<1048576xf32, 1>, vector<4xf32>
      %vy = vector.load %y[%i] : memref<1048576xf32, 1>, vector<4xf32>
      %r = math.fma %va, %vx, %vy : vector<4xf32>
      vector.store %r, %y[%i] : memref<1048576xf32, 1>, vector<4xf32>
    }
    gpu.return
  }
}
