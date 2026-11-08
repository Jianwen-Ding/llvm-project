// Row softmax with a SIMD-group per row of N (an argument, here 256)
// floats: a strided loop for the row's maximum, gpu.subgroup_reduce
// maxnumf, a loop summing exp(x - max), gpu.subgroup_reduce add, and a
// loop writing exp(x - max) / sum: three passes over the row.
// agx-cost: --uniform 2=256 --wg 128
gpu.module @k {
  gpu.func @softmax_rows(%x: memref<1024x1024xf32, 1>, %n: index) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %bx = gpu.block_id x upper_bound 256
    %sg = gpu.subgroup_id : index
    %lane = gpu.lane_id
    %row0 = arith.muli %bx, %c4 : index
    %row = arith.addi %row0, %sg : index
    %m = scf.for %j = %lane to %n step %c32 iter_args(%a = %ninf) -> f32 {
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %r = arith.maxnumf %a, %v : f32
      scf.yield %r : f32
    }
    %mx = gpu.subgroup_reduce maxnumf %m : (f32) -> f32
    %s = scf.for %j = %lane to %n step %c32 iter_args(%a = %zero) -> f32 {
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %d = arith.subf %v, %mx : f32
      %e = math.exp %d : f32
      %r = arith.addf %a, %e : f32
      scf.yield %r : f32
    }
    %sum = gpu.subgroup_reduce add %s : (f32) -> f32
    %inv = arith.divf %one, %sum : f32
    scf.for %j = %lane to %n step %c32 {
      %v = memref.load %x[%row, %j] : memref<1024x1024xf32, 1>
      %d = arith.subf %v, %mx : f32
      %e = math.exp %d : f32
      %r = arith.mulf %e, %inv : f32
      memref.store %r, %x[%row, %j] : memref<1024x1024xf32, 1>
    }
    gpu.return
  }
}
