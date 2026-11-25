// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 --split-input-file | FileCheck %s
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 speculate-guarded-loads=0" --split-input-file | FileCheck %s --check-prefix=OFF

// A scan's guarded load, in = j < 256; v = in ? x[r, j] : 0: the load runs
// unconditionally at j clamped into the row (umin(j, 255)), r below its
// size by its upper bound, and a select takes it (L15-E736); the select's
// only use, a compare in an and with in, reads the load itself, the and a
// select on in (false where in is, whatever the compare: a plain and of a
// poison compare would be poison). Only the loop's own branch is left.
// CHECK-LABEL: llvm.func @find_first
// CHECK: %[[F:.*]] = llvm.mlir.constant(false) : i1
// CHECK: llvm.intr.umin
// CHECK: %[[L:.*]] = llvm.load
// CHECK-NOT: llvm.select
// CHECK: %[[C:.*]] = llvm.fcmp "ole" %[[L]],
// CHECK-NEXT: %[[G:.*]] = llvm.select %{{.*}}, %[[C]], %[[F]] : i1, i1
// CHECK-NEXT: llvm.cond_br %[[G]]
// CHECK: llvm.return
// OFF-LABEL: llvm.func @find_first
// OFF-NOT: llvm.intr.umin
// OFF: llvm.return
gpu.module @k {
  gpu.func @find_first(%x: memref<1024x256xf32, 1>, %o: memref<1024xi32, 1>, %t: f32) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %r = gpu.global_id x upper_bound 1024
    %res = scf.while (%j = %c0) : (index) -> index {
      %in = arith.cmpi ult, %j, %c256 : index
      %v = scf.if %in -> (f32) {
        %l = memref.load %x[%r, %j] : memref<1024x256xf32, 1>
        scf.yield %l : f32
      } else {
        %z = arith.constant 0.0 : f32
        scf.yield %z : f32
      }
      %le = arith.cmpf ole, %v, %t : f32
      %go = arith.andi %in, %le : i1
      scf.condition(%go) %j : index
    } do {
    ^bb0(%j: index):
      %j1 = arith.addi %j, %c1 : index
      scf.yield %j1 : index
    }
    %ri = arith.index_cast %res : index to i32
    memref.store %ri, %o[%r] : memref<1024xi32, 1>
    gpu.return
  }
}

// -----

// Not an if whose load needs two clamps (a padded window's row and column):
// its branch on positions is cheaper than the clamps.
// CHECK-LABEL: llvm.func @padded
// CHECK-NOT: llvm.intr.umin
// CHECK: llvm.cond_br
// CHECK: llvm.return
gpu.module @k {
  gpu.func @padded(%x: memref<64x64xf32, 1>, %y: memref<64x64xf32, 1>, %i: index, %j: index) kernel
      attributes {known_block_size = array<i32: 64, 1, 1>} {
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0.0 : f32
    %a = arith.cmpi ult, %i, %c64 : index
    %b = arith.cmpi ult, %j, %c64 : index
    %in = arith.andi %a, %b : i1
    %v = scf.if %in -> (f32) {
      %l = memref.load %x[%i, %j] : memref<64x64xf32, 1>
      scf.yield %l : f32
    } else {
      scf.yield %zero : f32
    }
    %c0 = arith.constant 0 : index
    memref.store %v, %y[%c0, %c0] : memref<64x64xf32, 1>
    gpu.return
  }
}

// -----

// Not a region with a division (not speculatable: a zero divisor).
// CHECK-LABEL: llvm.func @divides
// CHECK-NOT: llvm.intr.umin
// CHECK: llvm.cond_br
// CHECK: llvm.return
gpu.module @k {
  gpu.func @divides(%x: memref<64xi32, 1>, %y: memref<1xi32, 1>, %d: i32) kernel
      attributes {known_block_size = array<i32: 64, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c0i = arith.constant 0 : i32
    %i = gpu.thread_id x
    %in = arith.cmpi ne, %d, %c0i : i32
    %v = scf.if %in -> (i32) {
      %l = memref.load %x[%i] : memref<64xi32, 1>
      %q = arith.divui %l, %d : i32
      scf.yield %q : i32
    } else {
      scf.yield %c0i : i32
    }
    memref.store %v, %y[%c0] : memref<1xi32, 1>
    gpu.return
  }
}
