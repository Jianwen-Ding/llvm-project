// A row's maximumf of 256 floats in vector<4xf32> steps, a thread per row
// (torch-mlir's softmax maximum, NaN-propagating, -0 below +0): the NaN and
// the zeros' sign carried beside an ordered compare-select (L15-E444).
// agx-cost: --wg 256
gpu.module @k {
  gpu.func @rowmax_vec4(%x: memref<1024x256xf32, 1>, %out: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %ninf = arith.constant dense<0xFF800000> : vector<4xf32>
    %r = gpu.global_id x upper_bound 1024
    %m = scf.for %j = %c0 to %c256 step %c4 iter_args(%a = %ninf) -> vector<4xf32> {
      %v = vector.load %x[%r, %j] : memref<1024x256xf32, 1>, vector<4xf32>
      %n = arith.maximumf %a, %v : vector<4xf32>
      scf.yield %n : vector<4xf32>
    }
    %s = vector.reduction <maximumf>, %m : vector<4xf32> into f32
    memref.store %s, %out[%r] : memref<1024xf32, 1>
    gpu.return
  }
}
