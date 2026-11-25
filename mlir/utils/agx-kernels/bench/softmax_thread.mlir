// Softmax of 64-float rows, a thread per row: max, sum of exp, normalized store.
gpu.module @k {
  gpu.func @softmax_thread(%x: memref<4096x64xf32, 1>, %y: memref<4096x64xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %ninf = arith.constant 0xFF800000 : f32
    %zf = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %r = gpu.global_id x upper_bound 4096
    %m = scf.for %j = %c0 to %c64 step %c1 iter_args(%a = %ninf) -> f32 {
      %v = memref.load %x[%r, %j] : memref<4096x64xf32, 1>
      %n = arith.maximumf %a, %v : f32
      scf.yield %n : f32
    }
    %s = scf.for %j = %c0 to %c64 step %c1 iter_args(%a = %zf) -> f32 {
      %v = memref.load %x[%r, %j] : memref<4096x64xf32, 1>
      %d = arith.subf %v, %m : f32
      %e = math.exp %d : f32
      %n = arith.addf %a, %e : f32
      scf.yield %n : f32
    }
    %inv = arith.divf %one, %s : f32
    scf.for %j = %c0 to %c64 step %c1 {
      %v = memref.load %x[%r, %j] : memref<4096x64xf32, 1>
      %d = arith.subf %v, %m : f32
      %e = math.exp %d : f32
      %p = arith.mulf %e, %inv : f32
      memref.store %p, %y[%r, %j] : memref<4096x64xf32, 1>
    }
    gpu.return
  }
}
