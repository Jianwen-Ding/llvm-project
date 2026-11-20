// int8 row dot products with an int8 vector in i32, a thread per row of 512, scalar byte loads.
gpu.module @k {
  gpu.func @dot_i8_scalar(%a: memref<1024x512xi8, 1>, %b: memref<512xi8, 1>, %out: memref<1024xi32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %zero = arith.constant 0 : i32
    %r = gpu.global_id x upper_bound 1024
    %s = scf.for %j = %c0 to %c512 step %c1 iter_args(%acc = %zero) -> i32 {
      %x = memref.load %a[%r, %j] : memref<1024x512xi8, 1>
      %y = memref.load %b[%j] : memref<512xi8, 1>
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %t = arith.addi %acc, %m : i32
      scf.yield %t : i32
    }
    memref.store %s, %out[%r] : memref<1024xi32, 1>
    gpu.return
  }
}
