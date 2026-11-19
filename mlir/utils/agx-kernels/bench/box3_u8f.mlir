// A 3-tap box filter of bytes to floats: (x[i] + x[i+1] + x[i+2]) / 3 with the
// sum in float, exact (an integer below 2^24): done in integers and converted
// once (L15-E407).
// agx-cost: --wg 256
gpu.module @k {
  gpu.func @box3_u8f(%in: memref<1048584xi8, 1>, %out: memref<1048576xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %g = gpu.global_id x upper_bound 1048576
    %g1 = arith.addi %g, %c1 : index
    %g2 = arith.addi %g, %c2 : index
    %a = memref.load %in[%g] : memref<1048584xi8, 1>
    %b = memref.load %in[%g1] : memref<1048584xi8, 1>
    %c = memref.load %in[%g2] : memref<1048584xi8, 1>
    %fa = arith.uitofp %a : i8 to f32
    %fb = arith.uitofp %b : i8 to f32
    %fc = arith.uitofp %c : i8 to f32
    %s1 = arith.addf %fa, %fb : f32
    %s = arith.addf %s1, %fc : f32
    %k = arith.constant 0.33333334 : f32
    %m = arith.mulf %s, %k : f32
    memref.store %m, %out[%g] : memref<1048576xf32, 1>
    gpu.return
  }
}
