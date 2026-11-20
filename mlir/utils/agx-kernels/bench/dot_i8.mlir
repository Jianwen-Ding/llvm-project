// int8 GEMV: y[r] = sum_k a[r, k] * x[k] in i32, a thread per row, vector<4xi8> loads.
gpu.module @k {
  gpu.func @dot_i8(%a: memref<1024x1024xi8, 1>, %x: memref<1024xi8, 1>, %y: memref<1024xi32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %z = arith.constant dense<0> : vector<4xi32>
    %r = gpu.global_id x upper_bound 1024
    %s = scf.for %j = %c0 to %c1024 step %c4 iter_args(%acc = %z) -> vector<4xi32> {
      %u = vector.load %a[%r, %j] : memref<1024x1024xi8, 1>, vector<4xi8>
      %v = vector.load %x[%j] : memref<1024xi8, 1>, vector<4xi8>
      %ue = arith.extsi %u : vector<4xi8> to vector<4xi32>
      %ve = arith.extsi %v : vector<4xi8> to vector<4xi32>
      %p = arith.muli %ue, %ve : vector<4xi32>
      %n = arith.addi %acc, %p : vector<4xi32>
      scf.yield %n : vector<4xi32>
    }
    %t = vector.reduction <add>, %s : vector<4xi32> into i32
    memref.store %t, %y[%r] : memref<1024xi32, 1>
    gpu.return
  }
}
