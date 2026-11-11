// A 3 x 3 convolution over a 258 x 258 image, its loops fully unrolled
// (immediate offsets with the known grid's bounds, L15-E192).
gpu.module @k {
  gpu.func @conv3x3(%in: memref<258x258xf32, 1>, %w: memref<3x3xf32, 1>, %out: memref<256x256xf32, 1>) kernel
      attributes {known_block_size = array<i32: 16, 16, 1>, known_grid_size = array<i32: 16, 16, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %zero = arith.constant 0.0 : f32
    %x = gpu.global_id x
    %y = gpu.global_id y
    %s = scf.for %dy = %c0 to %c3 step %c1 iter_args(%acc = %zero) -> f32 {
      %s2 = scf.for %dx = %c0 to %c3 step %c1 iter_args(%acc2 = %acc) -> f32 {
        %yy = arith.addi %y, %dy : index
        %xx = arith.addi %x, %dx : index
        %v = memref.load %in[%yy, %xx] : memref<258x258xf32, 1>
        %k = memref.load %w[%dy, %dx] : memref<3x3xf32, 1>
        %m = math.fma %v, %k, %acc2 : f32
        scf.yield %m : f32
      }
      scf.yield %s2 : f32
    }
    memref.store %s, %out[%y, %x] : memref<256x256xf32, 1>
    gpu.return
  }
}
