// y = clamp(x * 3 + b[j], -32768, 32767) in int16, a thread per row of 256.
gpu.module @k {
  gpu.func @i16_rows(%x: memref<1024x256xi16, 1>, %b: memref<256xi16, 1>, %y: memref<1024x256xi16, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %k3 = arith.constant 3 : i32
    %lo = arith.constant -32768 : i32
    %hi = arith.constant 32767 : i32
    %r = gpu.global_id x upper_bound 1024
    scf.for %j = %c0 to %c256 step %c1 {
      %v = memref.load %x[%r, %j] : memref<1024x256xi16, 1>
      %bv = memref.load %b[%j] : memref<256xi16, 1>
      %v32 = arith.extsi %v : i16 to i32
      %b32 = arith.extsi %bv : i16 to i32
      %m = arith.muli %v32, %k3 : i32
      %a = arith.addi %m, %b32 : i32
      %c1x = arith.maxsi %a, %lo : i32
      %c2x = arith.minsi %c1x, %hi : i32
      %o = arith.trunci %c2x : i32 to i16
      memref.store %o, %y[%r, %j] : memref<1024x256xi16, 1>
    }
    gpu.return
  }
}
