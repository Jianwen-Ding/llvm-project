// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 --split-input-file | FileCheck %s

// System values are calls of AIR's functions, components of the
// three-dimensional ones; kernels carry "air-kernel"; memrefs are bare
// pointers in AIR's address spaces. The functions of system values touch no
// memory, so that LLVM folds their calls together.
gpu.module @ids {
  // CHECK-DAG: llvm.func @air.thread_index_in_simdgroup() -> i32 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, {{.*}}>, no_unwind, will_return}
  // CHECK-DAG: llvm.func @air.thread_position_in_threadgroup.v3i32() -> vector<3xi32> attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, {{.*}}>, no_unwind, will_return}
  // CHECK-LABEL: llvm.func @ids(
  // CHECK-SAME: %{{.*}}: !llvm.ptr<1>)
  // CHECK-SAME: passthrough = ["air-kernel"]
  gpu.func @ids(%out: memref<8xi32, 1>) kernel {
    // CHECK: %[[T:.*]] = llvm.call @air.thread_position_in_threadgroup.v3i32() : () -> vector<3xi32>
    // CHECK: llvm.extractelement %[[T]]
    %t = gpu.thread_id x
    // CHECK: llvm.call @air.threadgroup_position_in_grid.v3i32()
    %b = gpu.block_id y
    // gpu.global_id is block_id * block_dim + thread_id (populateGpuRewritePatterns).
    // CHECK: llvm.call @air.threadgroup_position_in_grid.v3i32()
    // CHECK: llvm.call @air.threads_per_threadgroup.v3i32()
    // CHECK: llvm.call @air.thread_position_in_threadgroup.v3i32()
    %g = gpu.global_id z
    // A lane is in [0, 32), and there are at most 32 SIMD-groups.
    // CHECK: llvm.call @air.thread_index_in_simdgroup() : () -> (i32 {llvm.range = #llvm.constant_range<i32, 0, 32>})
    %l = gpu.lane_id
    // CHECK: llvm.call @air.simdgroup_index_in_threadgroup() : () -> (i32 {llvm.range = #llvm.constant_range<i32, 0, 32>})
    %s = gpu.subgroup_id : index
    // CHECK: llvm.mlir.constant(32 : i64)
    %n = gpu.subgroup_size : index
    %x = arith.addi %t, %b : index
    %y = arith.addi %g, %l : index
    %z = arith.addi %s, %n : index
    %w = arith.addi %x, %y : index
    %v = arith.addi %w, %z : index
    %i = arith.index_cast %v : index to i32
    %c0 = arith.constant 0 : index
    memref.store %i, %out[%c0] : memref<8xi32, 1>
    gpu.return
  }
}

// -----

// The barrier orders device and threadgroup memory, or the memory its
// memfence names (none for `memfence []`); a subgroup-scope barrier is AIR's
// SIMD-group barrier. Shuffles, ballots and reductions are AIR's SIMD-group
// functions, convergent; the barriers may touch any memory.
gpu.module @simd {
  // CHECK-DAG: llvm.func @air.wg.barrier(i32, i32) attributes {convergent, no_unwind, will_return}
  // CHECK-DAG: llvm.func @air.simd_shuffle_xor.u.i32(i32, i16) -> i32 attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, {{.*}}>, no_unwind, will_return}
  // CHECK-LABEL: llvm.func @simd(
  gpu.func @simd(%out: memref<8xf32, 1>, %v: f32, %p: i1) kernel {
    // CHECK: %[[F:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK: %[[S:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.wg.barrier(%[[F]], %[[S]]) : (i32, i32) -> ()
    gpu.barrier
    // CHECK: %[[N:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: llvm.call @air.wg.barrier(%[[N]], %{{.*}}) : (i32, i32) -> ()
    gpu.barrier memfence []
    // CHECK: %[[W:.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @air.wg.barrier(%[[W]], %{{.*}}) : (i32, i32) -> ()
    gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: %[[G:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.simdgroup.barrier(%[[G]], %{{.*}}) : (i32, i32) -> ()
    gpu.barrier scope <subgroup> memfence [#gpu.address_space<global>]
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    // A constant offset within the SIMD-group: AIR's shuffle_xor.
    // CHECK: %[[MASK:.*]] = llvm.mlir.constant(1 : i16) : i16
    // CHECK: llvm.call @air.simd_shuffle_xor.u.i32(%{{.*}}, %[[MASK]]) : (i32, i16) -> i32
    %sh, %ok = gpu.shuffle xor %v, %c1, %c32 : f32
    // CHECK: llvm.call @air.simd_ballot.i64(%{{.*}}) : (i1) -> i64
    %m = gpu.ballot %p : i64
    // CHECK: llvm.call @air.simd_sum.f32(%{{.*}}) : (f32) -> f32
    %r = gpu.subgroup_reduce add %sh : (f32) -> f32
    %c0 = arith.constant 0 : index
    memref.store %r, %out[%c0] : memref<8xf32, 1>
    %mf = arith.uitofp %m : i64 to f32
    memref.store %mf, %out[%c0] : memref<8xf32, 1>
    gpu.return
  }
}

// -----

// SIMD-group matrices are AIR's: a pointer to the element at the indices, the
// strides from the memref (swapped for transpose), 8 x 8 dimensions.
gpu.module @matrix {
  // CHECK-LABEL: llvm.func @mm(
  gpu.func @mm(%a: memref<64xf32, 1>, %c: memref<64xf32, 1>) kernel {
    %i0 = arith.constant 0 : index
    %A = memref.reinterpret_cast %a to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<64xf32, 1> to memref<8x8xf32, strided<[8, 1]>, 1>
    %C = memref.reinterpret_cast %c to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<64xf32, 1> to memref<8x8xf32, strided<[8, 1]>, 1>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_load.v64f32.p1f32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, vector<2xi64>, vector<2xi64>, vector<2xi64>) -> vector<64xf32>
    %m = air.simdgroup_matrix_load %A[%i0, %i0] transpose : memref<8x8xf32, strided<[8, 1]>, 1> -> vector<64xf32>
    %one = arith.constant 1.0 : f32
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(%{{.*}}) : (f32) -> vector<64xf32>
    %f = air.simdgroup_matrix_fill %one : vector<64xf32>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32
    %d = air.simdgroup_matrix_multiply_accumulate %m, %m, %f : vector<64xf32>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_store.v64f32.p1f32(
    air.simdgroup_matrix_store %d, %C[%i0, %i0] : vector<64xf32> -> memref<8x8xf32, strided<[8, 1]>, 1>
    %h = arith.truncf %m : vector<64xf32> to vector<64xf16>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<64xf16>, vector<64xf16>, vector<64xf32>) -> vector<64xf32>
    %e = air.simdgroup_matrix_multiply_accumulate %h, %h, %d : vector<64xf16>, vector<64xf32>
    air.simdgroup_matrix_store %e, %C[%i0, %i0] : vector<64xf32> -> memref<8x8xf32, strided<[8, 1]>, 1>
    gpu.return
  }
}

// -----

// AIR has no generic address space: a memref without a memory space is device
// memory, except an alloca, which is the thread's.
gpu.module @default_space {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-SAME: %{{.*}}: !llvm.ptr<1>
  gpu.func @k(%out: memref<4xf32>) kernel {
    %c0 = arith.constant 0 : index
    // CHECK: llvm.alloca %{{.*}} x f32 : (i64) -> !llvm.ptr
    %tmp = memref.alloca() : memref<4xf32>
    %v = memref.load %tmp[%c0] : memref<4xf32>
    // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<1>
    memref.store %v, %out[%c0] : memref<4xf32>
    gpu.return
  }
}

// -----

// A known threadgroup size is an AGX hint on the kernel; the threadgroup
// size and SIMD-group count are constants then, and the thread positions
// and SIMD-group index are bounded by it.
gpu.module @block_size {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-SAME: passthrough = ["air-kernel", ["agx-max-threads-per-threadgroup", "128"]]
  gpu.func @k(%out: memref<8xindex, 1>) kernel attributes {known_block_size = array<i32: 64, 2, 1>} {
    // CHECK-NOT: @air.threads_per_threadgroup
    // CHECK: llvm.mlir.constant(2 : i64) : i64
    %d = gpu.block_dim y
    // CHECK: llvm.call @air.thread_position_in_threadgroup.v3i32() : () -> (vector<3xi32> {llvm.range = #llvm.constant_range<i32, 0, 64>})
    %t = gpu.thread_id x
    // CHECK: llvm.mlir.constant(4 : i64) : i64
    %n = gpu.num_subgroups : index
    // CHECK: llvm.call @air.simdgroup_index_in_threadgroup() : () -> (i32 {llvm.range = #llvm.constant_range<i32, 0, 4>})
    %s = gpu.subgroup_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    memref.store %d, %out[%c0] : memref<8xindex, 1>
    memref.store %t, %out[%c1] : memref<8xindex, 1>
    memref.store %n, %out[%c2] : memref<8xindex, 1>
    memref.store %s, %out[%c3] : memref<8xindex, 1>
    gpu.return
  }
}

// -----

// A threadgroup of one SIMD-group: its index is 0. upper_bound bounds a
// value too.
gpu.module @one_simdgroup {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xindex, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    // CHECK-NOT: @air.simdgroup_index_in_threadgroup
    // CHECK: llvm.mlir.constant(0 : i64) : i64
    %s = gpu.subgroup_id : index
    // CHECK: llvm.call @air.threadgroup_position_in_grid.v3i32() : () -> vector<3xi32>
    %b = gpu.block_id x upper_bound 16
    // CHECK: llvm.call @air.thread_index_in_simdgroup() : () -> (i32 {llvm.range = #llvm.constant_range<i32, 0, 16>})
    %l = gpu.lane_id upper_bound 16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %s, %out[%c0] : memref<8xindex, 1>
    memref.store %b, %out[%c1] : memref<8xindex, 1>
    memref.store %l, %out[%c2] : memref<8xindex, 1>
    gpu.return
  }
}

// -----

// Shuffles by constant offsets within the SIMD-group are AIR's shuffle_up,
// shuffle_down and shuffle_xor (xor within any power of two): the lanes a
// shift leaves keep their value, which is what gpu.shuffle's valid flag
// says; other shuffles read the lane they compute.
gpu.module @shuffles {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xi32, 1>, %v: i32, %o: i32) kernel {
    %c2 = arith.constant 2 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    // CHECK: %[[LANE:.*]] = llvm.call @air.thread_index_in_simdgroup()
    // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: llvm.icmp "uge" %[[LANE]], %[[TWO]] : i32
    // CHECK: llvm.call @air.simd_shuffle_up.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    %up, %okup = gpu.shuffle up %v, %c2, %c32 : i32
    // CHECK: %[[LANE2:.*]] = llvm.call @air.thread_index_in_simdgroup()
    // CHECK: %[[B:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.icmp "ult" %[[LANE2]], %[[B]] : i32
    // CHECK: llvm.call @air.simd_shuffle_down.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    %down, %okdown = gpu.shuffle down %v, %c16, %c32 : i32
    // CHECK: llvm.call @air.simd_shuffle_xor.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    %x8, %okx8 = gpu.shuffle xor %v, %c2, %c8 : i32
    // Within segments of 8 lanes, a shift needs the lane arithmetic.
    // CHECK: llvm.icmp "sge"
    // CHECK: llvm.call @air.simd_shuffle.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    %d8, %okd8 = gpu.shuffle down %v, %c2, %c8 : i32
    // CHECK: llvm.call @air.simd_shuffle.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    %xo, %okxo = gpu.shuffle xor %v, %o, %c32 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2i = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    memref.store %up, %out[%c0] : memref<8xi32, 1>
    memref.store %down, %out[%c1] : memref<8xi32, 1>
    memref.store %x8, %out[%c2i] : memref<8xi32, 1>
    memref.store %d8, %out[%c3] : memref<8xi32, 1>
    memref.store %xo, %out[%c4] : memref<8xi32, 1>
    gpu.return
  }
}

// -----

// A reduction over clusters of 4 lanes is AIR's quad-group reduction; one of
// 32 is the SIMD-group's.
gpu.module @clusters {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xf32, 1>, %v: f32, %i: i32, %o: memref<8xi32, 1>) kernel {
    // CHECK: llvm.call @air.quad_sum.f32(%{{.*}}) : (f32) -> f32
    %q = gpu.subgroup_reduce add %v cluster(size = 4) : (f32) -> f32
    // CHECK: llvm.call @air.simd_max.u.i32(%{{.*}}) : (i32) -> i32
    %m = gpu.subgroup_reduce maxui %i cluster(size = 32) : (i32) -> i32
    %c0 = arith.constant 0 : index
    memref.store %q, %out[%c0] : memref<8xf32, 1>
    memref.store %m, %o[%c0] : memref<8xi32, 1>
    gpu.return
  }
}

// -----

// gpu.global_id is AIR's 32-bit thread_position_in_grid where the dimension
// has fewer than 2^32 threads (upper_bound, or known_block_size times
// known_grid_size); without a bound, block_id * block_dim + thread_id in
// index, since the 32-bit register would wrap.
gpu.module @global_id {
  // CHECK-LABEL: llvm.func @bounded(
  gpu.func @bounded(%out: memref<8xindex, 1>) kernel {
    // CHECK: llvm.call @air.thread_position_in_grid.v3i32()
    // CHECK-NOT: @air.threadgroup_position_in_grid
    %g = gpu.global_id x upper_bound 65536
    %c0 = arith.constant 0 : index
    memref.store %g, %out[%c0] : memref<8xindex, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @known(
  gpu.func @known(%out: memref<8xindex, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>,
                  known_grid_size = array<i32: 4096, 1, 1>} {
    // CHECK: llvm.call @air.thread_position_in_grid.v3i32()
    %g = gpu.global_id x
    %c0 = arith.constant 0 : index
    memref.store %g, %out[%c0] : memref<8xindex, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @unbounded(
  gpu.func @unbounded(%out: memref<8xindex, 1>) kernel {
    // CHECK-NOT: @air.thread_position_in_grid
    // CHECK: llvm.call @air.threadgroup_position_in_grid.v3i32()
    // CHECK: llvm.call @air.threads_per_threadgroup.v3i32()
    // CHECK: llvm.call @air.thread_position_in_threadgroup.v3i32()
    // CHECK: llvm.mul %{{.*}} : i64
    // CHECK: llvm.add %{{.*}} : i64
    %g = gpu.global_id x
    %c0 = arith.constant 0 : index
    memref.store %g, %out[%c0] : memref<8xindex, 1>
    gpu.return
  }
}
