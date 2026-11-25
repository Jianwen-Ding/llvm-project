// A causally masked row reduction (attention's j <= i mask), a SIMD-group per
// row of 1024 scores: the running maximum of the unmasked scores, then the
// sum of their exponentials. The mask compares the 64-bit index j with the
// row (L15-E426).
// agx-cost: --wg 32 --wg-id 600,0,0
gpu.module @k {
  gpu.func @causal_rowmax(%s: memref<1048576xf32, 1>, %o: memref<1048576xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    %row = gpu.block_id x upper_bound 1024
    %lane = gpu.lane_id upper_bound 32
    %base = arith.muli %row, %c1024 : index
    %m = scf.for %j = %lane to %c1024 step %c32 iter_args(%a = %ninf) -> f32 {
      %idx = arith.addi %base, %j : index
      %v = memref.load %s[%idx] : memref<1048576xf32, 1>
      %ok = arith.cmpi ule, %j, %row : index
      %vm = arith.select %ok, %v, %ninf : f32
      %n = arith.maxnumf %a, %vm : f32
      scf.yield %n : f32
    }
    %mx = gpu.subgroup_reduce maxnumf %m : (f32) -> f32
    %sum = scf.for %j = %lane to %c1024 step %c32 iter_args(%a = %zero) -> f32 {
      %idx = arith.addi %base, %j : index
      %v = memref.load %s[%idx] : memref<1048576xf32, 1>
      %ok = arith.cmpi ule, %j, %row : index
      %d = arith.subf %v, %mx : f32
      %e = math.exp %d : f32
      %em = arith.select %ok, %e, %zero : f32
      %n = arith.addf %a, %em : f32
      scf.yield %n : f32
    }
    %tot = gpu.subgroup_reduce add %sum : (f32) -> f32
    %oidx = arith.addi %base, %lane : index
    memref.store %tot, %o[%oidx] : memref<1048576xf32, 1>
    gpu.return
  }
}
