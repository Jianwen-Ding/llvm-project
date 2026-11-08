// A threadgroup sum of 256 threads: SIMD-group sums, their partials in
// threadgroup memory, then the first SIMD-group's sum of those.
gpu.module @k {
  gpu.func @block_reduce(%in: memref<1048576xf32, 1>, %out: memref<1048576xf32, 1>)
      workgroup(%part: memref<8xf32, 3>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %gid = gpu.global_id x upper_bound 1048576
    %v = memref.load %in[%gid] : memref<1048576xf32, 1>
    %s = gpu.subgroup_reduce add %v : (f32) -> f32
    %lane = gpu.lane_id
    %sg = gpu.subgroup_id : index
    %first = arith.cmpi eq, %lane, %c0 : index
    scf.if %first {
      memref.store %s, %part[%sg] : memref<8xf32, 3>
    }
    gpu.barrier
    %n = gpu.num_subgroups : index
    %sg0 = arith.cmpi eq, %sg, %c0 : index
    scf.if %sg0 {
      %in_range = arith.cmpi ult, %lane, %n : index
      %p = scf.if %in_range -> f32 {
        %x = memref.load %part[%lane] : memref<8xf32, 3>
        scf.yield %x : f32
      } else {
        scf.yield %zero : f32
      }
      %t = gpu.subgroup_reduce add %p : (f32) -> f32
      scf.if %first {
        %b = gpu.block_id x
        memref.store %t, %out[%b] : memref<1048576xf32, 1>
      }
    }
    gpu.return
  }
}
