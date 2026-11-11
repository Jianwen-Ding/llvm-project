// Reductions over clusters of 8 (add) and 4 (maxnumf) lanes: xor shuffles
// within each cluster and AIR's quad-group maximum (L15-E197).
gpu.module @k {
  gpu.func @cluster(%a: memref<65536xf32, 1>, %o: memref<65536xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 256, 1, 1>} {
    %i = gpu.global_id x
    %v = memref.load %a[%i] : memref<65536xf32, 1>
    %r = gpu.subgroup_reduce add %v cluster(size = 8) : (f32) -> f32
    %m = gpu.subgroup_reduce maxnumf %v cluster(size = 4) : (f32) -> f32
    %s = arith.addf %r, %m : f32
    memref.store %s, %o[%i] : memref<65536xf32, 1>
    gpu.return
  }
}
