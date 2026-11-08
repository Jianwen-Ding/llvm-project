// A 2-D bias add with relu, f16 in and out, f32 arithmetic: one element of a
// 512 x 1024 array per thread.
gpu.module @k {
  gpu.func @bias_relu_2d_f16(%in: memref<512x1024xf16, 1>, %bias: memref<1024xf16, 1>,
                             %out: memref<512x1024xf16, 1>) kernel
      attributes {known_block_size = array<i32: 32, 8, 1>,
                  known_grid_size = array<i32: 32, 64, 1>} {
    %zero = arith.constant 0.0 : f32
    %c = gpu.global_id x
    %r = gpu.global_id y
    %v = memref.load %in[%r, %c] : memref<512x1024xf16, 1>
    %b = memref.load %bias[%c] : memref<1024xf16, 1>
    %vf = arith.extf %v : f16 to f32
    %bf = arith.extf %b : f16 to f32
    %s = arith.addf %vf, %bf : f32
    %m = arith.maximumf %s, %zero : f32
    %h = arith.truncf %m : f32 to f16
    memref.store %h, %out[%r, %c] : memref<512x1024xf16, 1>
    gpu.return
  }
}
