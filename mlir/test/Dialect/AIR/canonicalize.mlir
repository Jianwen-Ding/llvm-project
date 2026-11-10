// RUN: mlir-opt %s --canonicalize --split-input-file | FileCheck %s

// Element-wise operations of matrix fills are fills of the operation.
// CHECK-LABEL: func.func @arith
// CHECK-SAME: (%[[X:.*]]: f32, %[[Y:.*]]: f32, %[[H:.*]]: f16, %[[P:.*]]: i1)
func.func @arith(%x: f32, %y: f32, %h: f16, %p: i1) -> (vector<64xf32>, vector<64xf32>, vector<64xf16>, vector<64xf32>) {
  %a = air.simdgroup_matrix_fill %x : vector<64xf32>
  %b = air.simdgroup_matrix_fill %y : vector<64xf32>
  // CHECK: %[[S:.*]] = arith.addf %[[X]], %[[Y]] fastmath<contract> : f32
  // CHECK: %[[C:.*]] = air.simdgroup_matrix_fill %[[S]] : vector<64xf32>
  %c = arith.addf %a, %b fastmath<contract> : vector<64xf32>
  // CHECK: %[[E:.*]] = arith.extf %[[H]] : f16 to f32
  // CHECK: air.simdgroup_matrix_fill %[[E]] : vector<64xf32>
  %hf = air.simdgroup_matrix_fill %h : vector<64xf16>
  %e = arith.extf %hf : vector<64xf16> to vector<64xf32>
  // CHECK: %[[T:.*]] = arith.truncf %[[X]] : f32 to f16
  // CHECK: air.simdgroup_matrix_fill %[[T]] : vector<64xf16>
  %t = arith.truncf %a : vector<64xf32> to vector<64xf16>
  // A scalar condition stays.
  // CHECK: %[[SEL:.*]] = arith.select %[[P]], %[[X]], %[[Y]] : f32
  // CHECK: air.simdgroup_matrix_fill %[[SEL]] : vector<64xf32>
  %s = arith.select %p, %a, %b : vector<64xf32>
  return %c, %e, %t, %s : vector<64xf32>, vector<64xf32>, vector<64xf16>, vector<64xf32>
}

// -----

// A matrix that is not a fill keeps the operation on the matrices.
// CHECK-LABEL: func.func @mixed
func.func @mixed(%m: memref<8x8xf32>, %x: f32, %i: index) -> vector<64xf32> {
  %l = air.simdgroup_matrix_load %m[%i, %i] : memref<8x8xf32> -> vector<64xf32>
  %a = air.simdgroup_matrix_fill %x : vector<64xf32>
  // CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<64xf32>
  %c = arith.mulf %l, %a : vector<64xf32>
  return %c : vector<64xf32>
}

// -----

// So do element-wise operations of other vectors.
// CHECK-LABEL: func.func @other_vectors
func.func @other_vectors(%v: vector<64xf32>, %w: vector<4xf32>) -> (vector<64xf32>, vector<4xf32>) {
  // CHECK: arith.negf %{{.*}} : vector<64xf32>
  %n = arith.negf %v : vector<64xf32>
  // CHECK: arith.negf %{{.*}} : vector<4xf32>
  %m = arith.negf %w : vector<4xf32>
  return %n, %m : vector<64xf32>, vector<4xf32>
}
