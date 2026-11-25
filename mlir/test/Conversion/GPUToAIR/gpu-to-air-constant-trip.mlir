// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 --split-input-file | FileCheck %s
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 constant-trip-loops=0" --split-input-file | FileCheck %s --check-prefix=OFF

// A loop from a lane position to a constant bound by a constant step that
// every lane runs the same number of times (here 64: ceil((2048 - l) / 32)
// for l in [0, 32)) runs over that count from 0, iv = lane + 32 k, so that
// LLVM sees a constant trip count (L15-E511).
gpu.module @lane_rows {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-DAG: %[[T:.*]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK-DAG: %[[S:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: llvm.icmp "ult" %[[K:.*]], %[[T]] : i32
  // CHECK: llvm.mul %[[K]], %[[S]]
  // OFF-LABEL: llvm.func @k(
  // OFF-NOT: llvm.mlir.constant(64 : i32)
  // OFF: llvm.intr.umin
  gpu.func @k(%x: memref<2048xf32, 1>, %o: memref<32xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c32 = arith.constant 32 : index
    %c2048 = arith.constant 2048 : index
    %z = arith.constant 0.0 : f32
    %lane = gpu.lane_id upper_bound 32
    %s = scf.for %j = %lane to %c2048 step %c32 iter_args(%a = %z) -> f32 {
      %v = memref.load %x[%j] : memref<2048xf32, 1>
      %n = arith.addf %a, %v : f32
      scf.yield %n : f32
    }
    memref.store %s, %o[%lane] : memref<32xf32, 1>
    gpu.return
  }
}

// -----

// 1000 elements: lane 0 runs 32 times, lane 31 31 times; the loop stays.
gpu.module @uneven {
  // CHECK-LABEL: llvm.func @k(
  // CHECK: llvm.intr.umin
  // CHECK: llvm.icmp "ult"
  gpu.func @k(%x: memref<1000xf32, 1>, %o: memref<32xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %z = arith.constant 0.0 : f32
    %lane = gpu.lane_id upper_bound 32
    %s = scf.for %j = %lane to %c1000 step %c32 iter_args(%a = %z) -> f32 {
      %v = memref.load %x[%j] : memref<1000xf32, 1>
      %n = arith.addf %a, %v : f32
      scf.yield %n : f32
    }
    memref.store %s, %o[%lane] : memref<32xf32, 1>
    gpu.return
  }
}

// -----

// Four lanes' vectors (lane * 4 in [0, 124], step 128 over 1024: 8 trips
// each); a loop whose iv is compared (a window) stays.
gpu.module @scaled_and_compared {
  // CHECK-LABEL: llvm.func @k(
  // CHECK: %[[T:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: llvm.icmp "ult" %{{.*}}, %[[T]] : i32
  // CHECK: llvm.intr.umin
  gpu.func @k(%x: memref<1024xf32, 1>, %o: memref<32xf32, 1>, %w: index) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %z = arith.constant 0.0 : f32
    %lane = gpu.lane_id upper_bound 32
    %l4 = arith.muli %lane, %c4 : index
    %s = scf.for %j = %l4 to %c1024 step %c128 iter_args(%a = %z) -> f32 {
      %v = memref.load %x[%j] : memref<1024xf32, 1>
      %n = arith.addf %a, %v : f32
      scf.yield %n : f32
    }
    %t = scf.for %j = %lane to %c1024 step %c32 iter_args(%a = %s) -> f32 {
      %v = memref.load %x[%j] : memref<1024xf32, 1>
      %ok = arith.cmpi ule, %j, %w : index
      %vm = arith.select %ok, %v, %z : f32
      %n = arith.addf %a, %vm : f32
      scf.yield %n : f32
    }
    memref.store %t, %o[%lane] : memref<32xf32, 1>
    gpu.return
  }
}
