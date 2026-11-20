// Row softmax of f32 rows of 1024 by a SIMD-group, constant trips: max, sum of exp, normalized store.
gpu.module @k {
  gpu.func @softmax_f32v(%x: memref<1024x1024xf32, 1>, %y: memref<1024x1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zf = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %row = gpu.block_id x upper_bound 1024
    %lane = gpu.lane_id upper_bound 32
    %m = scf.for %j = %lane to %c1024 step %c32 iter_args(%a = %ninf) -> f32 {
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %n = arith.maximumf %a, %v : f32
      scf.yield %n : f32
    }
    %mm = gpu.subgroup_reduce maximumf %m : (f32) -> f32
    %s = scf.for %j = %lane to %c1024 step %c32 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %d = arith.subf %v, %mm : f32
      %e = math.exp %d : f32
      %n = arith.addf %a, %e : f32
      scf.yield %n : f32
    }
    %ss = gpu.subgroup_reduce add %s : (f32) -> f32
    %inv = arith.divf %one, %ss : f32
    scf.for %j = %lane to %c1024 step %c32 {
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %d = arith.subf %v, %mm : f32
      %e = math.exp %d : f32
      %p = arith.mulf %e, %inv : f32
      memref.store %p, %y[%row, %j] : memref<1024x1024xf32, 1>
    }
    gpu.return
  }
}
