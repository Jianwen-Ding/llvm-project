// SiLU of f16 activations plus an f16 bias, four halves per thread as
// vector<4xf16> loads and stores (8 bytes), computed in f32:
// y = (x + b) * sigmoid(x + b), as in a transformer MLP's f16 path.
gpu.module @k {
  gpu.func @silu_f16x4(%x: memref<1048576xf16, 1>, %bias: memref<4096xf16, 1>,
                       %y: memref<1048576xf16, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c4 = arith.constant 4 : index
    %c4095 = arith.constant 4095 : index
    %one = arith.constant dense<1.0> : vector<4xf32>
    %gid = gpu.global_id x upper_bound 262144
    %i = arith.muli %gid, %c4 : index
    %bi = arith.andi %i, %c4095 : index
    %v = vector.load %x[%i] : memref<1048576xf16, 1>, vector<4xf16>
    %b = vector.load %bias[%bi] : memref<4096xf16, 1>, vector<4xf16>
    %vb = arith.addf %v, %b : vector<4xf16>
    %f = arith.extf %vb : vector<4xf16> to vector<4xf32>
    %n = arith.negf %f : vector<4xf32>
    %e = math.exp %n : vector<4xf32>
    %d = arith.addf %e, %one : vector<4xf32>
    %s = arith.divf %f, %d : vector<4xf32>
    %h = arith.truncf %s : vector<4xf32> to vector<4xf16>
    vector.store %h, %y[%i] : memref<1048576xf16, 1>, vector<4xf16>
    gpu.return
  }
}
