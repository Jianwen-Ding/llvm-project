// SIMD-group reductions of f16 (whole and over clusters of 8) and of i64,
// and an i64 shuffle (L15-E198).
gpu.module @k {
  gpu.func @red16(%a: memref<65536xf16, 1>, %b: memref<65536xi64, 1>, %o: memref<65536xf16, 1>, %p: memref<65536xi64, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 256, 1, 1>} {
    %i = gpu.global_id x
    %v = memref.load %a[%i] : memref<65536xf16, 1>
    %r = gpu.subgroup_reduce add %v : (f16) -> f16
    %q = gpu.subgroup_reduce add %v cluster(size = 8) : (f16) -> f16
    %s = arith.addf %r, %q : f16
    memref.store %s, %o[%i] : memref<65536xf16, 1>
    %w = memref.load %b[%i] : memref<65536xi64, 1>
    %c = arith.constant 3 : i32
    %n = arith.constant 32 : i32
    %x, %ok = gpu.shuffle xor %w, %c, %n : i64
    %m = gpu.subgroup_reduce maxui %w : (i64) -> i64
    %t = arith.addi %x, %m : i64
    memref.store %t, %p[%i] : memref<65536xi64, 1>
    gpu.return
  }
}
