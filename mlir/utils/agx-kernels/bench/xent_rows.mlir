// Cross-entropy loss per row of 1000 logits, a SIMD-group per row: the
// maximum (maximumf), the sum of exp(x - max), and lane 0's log(sum) + max -
// x[label]. The loops' trip umin(1000 - lane, 1000) is lane-bounded: 32-bit,
// the umin gone (L15-E435).
// agx-cost: --wg 32 --wg-id 5,0,0
gpu.module @k {
  gpu.func @xent_rows(%x: memref<1024x1000xf32, 1>, %lab: memref<1024xi32, 1>, %loss: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zf = arith.constant 0.0 : f32
    %row = gpu.block_id x upper_bound 1024
    %lane = gpu.thread_id x upper_bound 32
    %m = scf.for %j = %lane to %c1000 step %c32 iter_args(%a = %ninf) -> f32 {
      %v = memref.load %x[%row, %j] : memref<1024x1000xf32, 1>
      %n = arith.maximumf %a, %v : f32
      scf.yield %n : f32
    }
    %mm = gpu.subgroup_reduce maximumf %m : (f32) -> f32
    %s = scf.for %j = %lane to %c1000 step %c32 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%row, %j] : memref<1024x1000xf32, 1>
      %d = arith.subf %v, %mm : f32
      %e = math.exp %d : f32
      %n = arith.addf %a, %e : f32
      scf.yield %n : f32
    }
    %ss = gpu.subgroup_reduce add %s : (f32) -> f32
    %is0 = arith.cmpi eq, %lane, %c0 : index
    scf.if %is0 {
      %l = memref.load %lab[%row] : memref<1024xi32, 1>
      %li = arith.index_cast %l : i32 to index
      %xl = memref.load %x[%row, %li] : memref<1024x1000xf32, 1>
      %lg = math.log %ss : f32
      %t = arith.addf %lg, %mm : f32
      %r = arith.subf %t, %xl : f32
      memref.store %r, %loss[%row] : memref<1024xf32, 1>
    }
    gpu.return
  }
}
