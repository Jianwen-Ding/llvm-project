// A 5-point Jacobi step on the interior of a 1026 x 1026 grid (one
// halo cell each side), 16 x 16 threads per threadgroup, 2-D memrefs:
// out = 0.25 (n + s + e + w) - h2 f.
gpu.module @k {
  gpu.func @stencil5_2d(%u: memref<1026x1026xf32, 1>, %f: memref<1026x1026xf32, 1>,
                        %out: memref<1026x1026xf32, 1>, %h2: f32) kernel
      attributes {known_block_size = array<i32: 16, 16, 1>} {
    %c1 = arith.constant 1 : index
    %q = arith.constant 0.25 : f32
    %gx = gpu.global_id x upper_bound 1024
    %gy = gpu.global_id y upper_bound 1024
    %x = arith.addi %gx, %c1 : index
    %y = arith.addi %gy, %c1 : index
    %xm = arith.subi %x, %c1 : index
    %xp = arith.addi %x, %c1 : index
    %ym = arith.subi %y, %c1 : index
    %yp = arith.addi %y, %c1 : index
    %n = memref.load %u[%ym, %x] : memref<1026x1026xf32, 1>
    %s = memref.load %u[%yp, %x] : memref<1026x1026xf32, 1>
    %w = memref.load %u[%y, %xm] : memref<1026x1026xf32, 1>
    %e = memref.load %u[%y, %xp] : memref<1026x1026xf32, 1>
    %fv = memref.load %f[%y, %x] : memref<1026x1026xf32, 1>
    %a = arith.addf %n, %s : f32
    %b = arith.addf %w, %e : f32
    %c = arith.addf %a, %b : f32
    %hf = arith.mulf %h2, %fv : f32
    %r0 = arith.mulf %c, %q : f32
    %r = arith.subf %r0, %hf : f32
    memref.store %r, %out[%y, %x] : memref<1026x1026xf32, 1>
    gpu.return
  }
}
