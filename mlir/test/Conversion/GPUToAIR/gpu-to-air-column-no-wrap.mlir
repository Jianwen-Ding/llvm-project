// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 | FileCheck %s
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 column-no-wrap=0" | FileCheck %s --check-prefix=OFF

// A column's access, in a loop that steps an outer dimension and stores
// nothing, has its linear index i * 1024 + c without wrapping (an access out
// of bounds is undefined): LLVM then steps one pointer and reads an unrolled
// loop's rows at immediate offsets from it (L15-E641). Not in a loop that
// stores (its bounds would version it), nor a row's (its word indices).

// CHECK-LABEL: llvm.func @col_sum(
// CHECK: %[[M:.*]] = llvm.mul %{{.*}}, %{{.*}} overflow<nuw> : i32
// CHECK: llvm.add %[[M]], %{{.*}} overflow<nuw> : i32
// OFF-LABEL: llvm.func @col_sum(
// OFF: %[[M:.*]] = llvm.mul %{{.*}}, %{{.*}} : i32
// OFF-NEXT: llvm.add %[[M]], %{{.*}} : i32

// CHECK-LABEL: llvm.func @col_scale(
// CHECK: %[[M:.*]] = llvm.mul %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: llvm.add %[[M]], %{{.*}} : i32

// CHECK-LABEL: llvm.func @row_sum(
// CHECK: %[[M:.*]] = llvm.mul %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: llvm.add %[[M]], %{{.*}} : i32

gpu.module @column_no_wrap {
  gpu.func @col_sum(%x: memref<256x1024xf32, 1>, %y: memref<1024xf32, 1>, %n: index) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0.0 : f32
    %c = gpu.global_id x upper_bound 1024
    %s = scf.for %i = %c0 to %n step %c1 iter_args(%a = %z) -> f32 {
      %v = memref.load %x[%i, %c] : memref<256x1024xf32, 1>
      %b = arith.addf %a, %v : f32
      scf.yield %b : f32
    }
    memref.store %s, %y[%c] : memref<1024xf32, 1>
    gpu.return
  }
  gpu.func @col_scale(%x: memref<256x1024xf32, 1>, %n: index, %k: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c = gpu.global_id x upper_bound 1024
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %x[%i, %c] : memref<256x1024xf32, 1>
      %b = arith.mulf %v, %k : f32
      memref.store %b, %x[%i, %c] : memref<256x1024xf32, 1>
    }
    gpu.return
  }
  gpu.func @row_sum(%x: memref<256x1024xf32, 1>, %y: memref<256xf32, 1>, %n: index) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0.0 : f32
    %r = gpu.global_id x upper_bound 256
    %s = scf.for %j = %c0 to %n step %c1 iter_args(%a = %z) -> f32 {
      %v = memref.load %x[%r, %j] : memref<256x1024xf32, 1>
      %b = arith.addf %a, %v : f32
      scf.yield %b : f32
    }
    memref.store %s, %y[%r] : memref<256xf32, 1>
    gpu.return
  }
}
