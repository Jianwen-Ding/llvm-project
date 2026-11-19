// A zero-padded 3 x 3 convolution of a 1024 x 1024 image (nine taps, each
// load guarded by its bounds test, the weights read in the loop): the guarded
// loads stay in flight across the guards' ifs, waited for at the taps' sums
// (L15-E427, L15-E428).
// agx-cost: --wg 16,16
gpu.module @k {
  gpu.func @conv3x3_pad(%x: memref<1024x1024xf32, 1>, %w: memref<9xf32, 1>, %y: memref<1024x1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 16, 16, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0.0 : f32
    %gx = gpu.global_id x upper_bound 1024
    %gy = gpu.global_id y upper_bound 1024
    %s = scf.for %dy = %c0 to %c3 step %c1 iter_args(%a = %zero) -> f32 {
      %s2 = scf.for %dx = %c0 to %c3 step %c1 iter_args(%b = %a) -> f32 {
        %yy0 = arith.addi %gy, %dy : index
        %xx0 = arith.addi %gx, %dx : index
        %yy = arith.subi %yy0, %c1 : index
        %xx = arith.subi %xx0, %c1 : index
        %y0 = arith.cmpi sge, %yy0, %c1 : index
        %y1 = arith.cmpi slt, %yy, %c1024 : index
        %x0 = arith.cmpi sge, %xx0, %c1 : index
        %x1 = arith.cmpi slt, %xx, %c1024 : index
        %ya = arith.andi %y0, %y1 : i1
        %xa = arith.andi %x0, %x1 : i1
        %in = arith.andi %ya, %xa : i1
        %v = scf.if %in -> f32 {
          %l = memref.load %x[%yy, %xx] : memref<1024x1024xf32, 1>
          scf.yield %l : f32
        } else {
          scf.yield %zero : f32
        }
        %k3 = arith.muli %dy, %c3 : index
        %k = arith.addi %k3, %dx : index
        %wk = memref.load %w[%k] : memref<9xf32, 1>
        %n = math.fma %v, %wk, %b : f32
        scf.yield %n : f32
      }
      scf.yield %s2 : f32
    }
    memref.store %s, %y[%gy, %gx] : memref<1024x1024xf32, 1>
    gpu.return
  }
}
