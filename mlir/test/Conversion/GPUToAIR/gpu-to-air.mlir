// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 --split-input-file | FileCheck %s
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 narrow-loops=0 linearize-indices=0 simdgroup-all-reduce=0" --split-input-file | FileCheck %s --check-prefix=OFF

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

// A known threadgroup size is an AGX hint on the kernel (its thread count
// and its shape); the threadgroup size and SIMD-group count are constants
// then, and the thread positions and SIMD-group index are bounded by it.
gpu.module @block_size {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-SAME: passthrough = ["air-kernel", ["agx-max-threads-per-threadgroup", "128"], ["agx-threads-per-threadgroup", "64,2,1"]]
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

// maximumf / minimumf of f32 (NaN if any lane's is, -0.0 below +0.0, which
// AIR's float maximum, a maxnum, does not give): the signed i32 maximum of
// the keys b ^ ((b >>s 31) & 0x7fffffff), decoded. A value reduced once: a
// NaN's key replaced by the quiet NaN's (by 0xffffffff's, INT32_MIN, whose
// decoded NaN then selects the quiet NaN), no or of "is NaN". A value that
// two reductions share: its own key, and a NaN where the group's or of
// "is NaN" is set.
gpu.module @ordered_float_reduce {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xf32, 1>, %v: f32) kernel {
    // CHECK: %[[F:.*]] = llvm.freeze
    // CHECK: llvm.fcmp "uno" %[[F]], %[[F]]
    // CHECK: %[[B:.*]] = llvm.bitcast %[[F]] : f32 to i32
    // CHECK: %[[S:.*]] = llvm.ashr %[[B]]
    // CHECK: %[[M:.*]] = llvm.and %[[S]]
    // CHECK: %[[K:.*]] = llvm.xor %[[B]], %[[M]]
    // CHECK: llvm.call @air.simd_max.s.i32(%[[K]])
    // CHECK: llvm.call @air.simd_or.s.i32
    // CHECK: llvm.select
    %mx = gpu.subgroup_reduce maximumf %v : (f32) -> f32
    // CHECK: llvm.call @air.quad_min.s.i32
    // CHECK: llvm.call @air.quad_or.s.i32
    %mn = gpu.subgroup_reduce minimumf %v cluster(size = 4) : (f32) -> f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %mx, %out[%c0] : memref<8xf32, 1>
    memref.store %mn, %out[%c1] : memref<8xf32, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @single(
  gpu.func @single(%out: memref<8xf32, 1>, %v: f32, %w: f32) kernel {
    // CHECK: %[[F:.*]] = llvm.freeze
    // CHECK: %[[N:.*]] = llvm.fcmp "uno" %[[F]], %[[F]]
    // CHECK: %[[B:.*]] = llvm.bitcast %[[F]] : f32 to i32
    // CHECK: %[[S:.*]] = llvm.ashr %[[B]]
    // CHECK: %[[M:.*]] = llvm.and %[[S]]
    // CHECK: %[[K0:.*]] = llvm.xor %[[B]], %[[M]]
    // CHECK: %[[K:.*]] = llvm.select %[[N]], %{{.*}}, %[[K0]] : i1, i32
    // CHECK: llvm.call @air.simd_max.s.i32(%[[K]])
    // CHECK-NOT: _or.s.i32
    %mx = gpu.subgroup_reduce maximumf %v : (f32) -> f32
    // CHECK: %[[Q:.*]] = llvm.call @air.quad_min.s.i32
    // CHECK-NOT: _or.s.i32
    // CHECK: %[[R:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: %[[RN:.*]] = llvm.fcmp "uno" %[[R]], %[[R]]
    // CHECK: llvm.select %[[RN]], %{{.*}}, %[[R]]
    // CHECK-NOT: _or.s.i32
    %mn = gpu.subgroup_reduce minimumf %w cluster(size = 4) : (f32) -> f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %mx, %out[%c0] : memref<8xf32, 1>
    memref.store %mn, %out[%c1] : memref<8xf32, 1>
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
  // The known grid is an AGX hint too (the backend bounds positions by it
  // where an address needs the bound, L15-E190).
  // CHECK-LABEL: llvm.func @known(
  // CHECK-SAME: ["agx-threadgroups-per-grid", "4096,1,1"], ["agx-threads-per-grid", "1048576,1,1"]
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

// -----

// The index of a memref access of a static dimension of at most 2^32
// elements is below 2^32 (out of bounds is undefined), so the arithmetic
// that computes it is 32-bit and the index its zero extension
// (narrow-indices). Larger or dynamic dimensions keep index arithmetic.
gpu.module @narrow_indices {
  // CHECK-LABEL: llvm.func @narrow(
  // CHECK-SAME: %[[A:.*]]: !llvm.ptr<1>, %[[I:.*]]: i64, %[[J:.*]]: i64, %[[LD:.*]]: i64)
  gpu.func @narrow(%a: memref<1024x16xf32, 1>, %i: index, %j: index, %ld: index) kernel {
    // CHECK-DAG: %[[I32:.*]] = llvm.trunc %[[I]] : i64 to i32
    // CHECK-DAG: %[[LD32:.*]] = llvm.trunc %[[LD]] : i64 to i32
    // CHECK-DAG: %[[J32:.*]] = llvm.trunc %[[J]] : i64 to i32
    // CHECK: %[[M:.*]] = llvm.mul %[[I32]], %[[LD32]] : i32
    // CHECK: %[[S:.*]] = llvm.add %[[M]], %[[J32]] : i32
    // CHECK: %[[L:.*]] = llvm.mul %[[S]], %{{.*}} : i32
    // CHECK: llvm.zext %[[L]] : i32 to i64
    %m = arith.muli %i, %ld : index
    %r = arith.addi %m, %j : index
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%r, %c0] : memref<1024x16xf32, 1>
    memref.store %v, %a[%c0, %c0] : memref<1024x16xf32, 1>
    gpu.return
  }
  // A sum of ids and constants that are not negative (no partial sum or
  // product wraps in index) is at most the index, below 2^32: its i32
  // additions do not wrap unsigned. One of an argument (perhaps negative)
  // may.
  // CHECK-LABEL: llvm.func @nuw(
  gpu.func @nuw(%a: memref<65538xf32, 1>, %k: index) kernel {
    // CHECK: %[[X:.*]] = llvm.add %{{.*}}, %{{.*}} overflow<nuw> : i32
    // CHECK: llvm.zext %[[X]] : i32 to i64
    // CHECK: %[[Y:.*]] = llvm.add %{{.*}}, %{{.*}} : i32
    // CHECK: llvm.zext %[[Y]] : i32 to i64
    %c2 = arith.constant 2 : index
    %i = gpu.thread_id x
    %x = arith.addi %i, %c2 : index
    %v = memref.load %a[%x] : memref<65538xf32, 1>
    %y = arith.addi %i, %k : index
    memref.store %v, %a[%y] : memref<65538xf32, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @wide(
  gpu.func @wide(%a: memref<8589934592xf32, 1>, %i: index, %ld: index) kernel {
    // CHECK-NOT: llvm.trunc
    // CHECK: llvm.mul %{{.*}} : i64
    %m = arith.muli %i, %ld : index
    %v = memref.load %a[%m] : memref<8589934592xf32, 1>
    %c0 = arith.constant 0 : index
    memref.store %v, %a[%c0] : memref<8589934592xf32, 1>
    gpu.return
  }
}

// -----

// An scf.for whose iterations index a static dimension of S <= 2^31
// elements by x + iv (or x - iv) keeps its offsets iv - lb below S (out of
// bounds is undefined), so it runs over an unsigned i32 offset from 0 to
// umin(ub - lb, S) by umin(step, S), and iv is lb + the offset
// (narrow-loops); index arithmetic of iv is then 32-bit.
gpu.module @narrow_loops {
  // CHECK-LABEL: llvm.func @grid_stride(
  // CHECK-SAME: %[[A:.*]]: !llvm.ptr<1>, %[[LB:.*]]: i64, %[[UB:.*]]: i64, %[[ST:.*]]: i64)
  // OFF-LABEL: llvm.func @grid_stride(
  gpu.func @grid_stride(%a: memref<1024xf32, 1>, %lb: index, %ub: index, %st: index) kernel {
    // CHECK-DAG: %[[LB32:.*]] = llvm.trunc %[[LB]] : i64 to i32
    // CHECK-DAG: %[[GT:.*]] = llvm.icmp "sgt" %[[UB]], %[[LB]] : i64
    // CHECK-DAG: %[[D:.*]] = llvm.sub %[[UB]], %[[LB]] : i64
    // CHECK-DAG: %[[DM:.*]] = llvm.intr.umin(%[[D]], %{{.*}})
    // CHECK-DAG: %[[E:.*]] = llvm.select %[[GT]], %[[DM]], %{{.*}} : i1, i64
    // CHECK-DAG: %[[STM:.*]] = llvm.intr.umin(%[[ST]], %{{.*}})
    // CHECK-DAG: %[[E32:.*]] = llvm.trunc %[[E]] : i64 to i32
    // CHECK-DAG: %[[ST32:.*]] = llvm.trunc %[[STM]] : i64 to i32
    // CHECK: ^bb1(%[[O:.*]]: i32):
    // CHECK: llvm.icmp "ult" %[[O]], %[[E32]] : i32
    // CHECK: %[[I:.*]] = llvm.add %[[LB32]], %[[O]] : i32
    // CHECK: llvm.zext %[[I]] : i32 to i64
    // CHECK: llvm.add %[[O]], %[[ST32]] overflow<nuw> : i32
    // OFF: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
    scf.for %i = %lb to %ub step %st {
      %v = memref.load %a[%i] : memref<1024xf32, 1>
      %f = arith.addf %v, %v : f32
      memref.store %f, %a[%i] : memref<1024xf32, 1>
    }
    gpu.return
  }
  // A constant extent bounds the offsets too; the results are the loop's.
  // CHECK-LABEL: llvm.func @extent(
  gpu.func @extent(%a: memref<1048576xf32, 1>, %x: index, %c: i1) kernel {
    // CHECK: %[[C:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: ^bb1(%[[O:.*]]: i32, %[[ACC:.*]]: f32):
    // CHECK: llvm.icmp "ult" %[[O]], %[[C]] : i32
    // CHECK: llvm.add %{{.*}}, %[[O]] : i32
    // CHECK: llvm.add %[[O]], %{{.*}} overflow<nuw> : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0.0 : f32
    %s = scf.for %k = %c0 to %c64 step %c1 iter_args(%acc = %zero) -> f32 {
      %y = scf.if %c -> f32 {
        scf.yield %acc : f32
      } else {
        scf.yield %zero : f32
      }
      %j = arith.addi %x, %k : index
      %v = memref.load %a[%j] : memref<1048576xf32, 1>
      %r = arith.addf %y, %v : f32
      scf.yield %r : f32
    }
    memref.store %s, %a[%x] : memref<1048576xf32, 1>
    gpu.return
  }
  // An access after an op with regions might not be reached, one of a
  // dynamic dimension or by 2 * iv bounds nothing: index loops.
  // CHECK-LABEL: llvm.func @unbounded(
  gpu.func @unbounded(%a: memref<1024xf32, 1>, %ub: index, %c: i1) kernel {
    // CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
    // CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
    // CHECK-NOT: llvm.icmp "ult"
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f = arith.constant 1.0 : f32
    scf.for %i = %c0 to %ub step %c1 {
      scf.if %c {
        gpu.barrier
      }
      memref.store %f, %a[%i] : memref<1024xf32, 1>
    }
    scf.for %i = %c0 to %ub step %c1 {
      %j = arith.addi %i, %i : index
      memref.store %f, %a[%j] : memref<1024xf32, 1>
    }
    gpu.return
  }
  // An AIR matrix load or store indexes its memref as memref.load does
  // (base[i + r, j + c] in bounds): k < 64 bounds the loop.
  // CHECK-LABEL: llvm.func @matrix(
  // CHECK: ^bb1(%[[O:.*]]: i32, %{{.*}}: vector<64xf32>):
  // CHECK: llvm.icmp "ult" %[[O]], %{{.*}} : i32
  // OFF-LABEL: llvm.func @matrix(
  // OFF: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
  gpu.func @matrix(%a: memref<8x64xf32, 1>, %c: memref<8x8xf32, 1>, %kdim: index) kernel {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %init = air.simdgroup_matrix_fill %zero : vector<64xf32>
    %acc = scf.for %k = %c0 to %kdim step %c8 iter_args(%t = %init) -> vector<64xf32> {
      %m = air.simdgroup_matrix_load %a[%c0, %k] : memref<8x64xf32, 1> -> vector<64xf32>
      %d = air.simdgroup_matrix_multiply_accumulate %m, %m, %t : vector<64xf32>
      scf.yield %d : vector<64xf32>
    }
    air.simdgroup_matrix_store %acc, %c[%c0, %c0] : vector<64xf32> -> memref<8x8xf32, 1>
    gpu.return
  }
}

// -----

// An access of a static multi-dimensional memref whose elements are within
// 2^32 of its base has the 32-bit index offset + sum(index_k * stride_k)
// into the memref as one dimension (linearize-indices).
gpu.module @linearize_indices {
  // CHECK-LABEL: llvm.func @strided(
  // CHECK-SAME: %[[A:.*]]: !llvm.ptr<3>, %[[I:.*]]: i64, %[[J:.*]]: i64)
  // OFF-LABEL: llvm.func @strided(
  gpu.func @strided(%a: memref<4x8xf32, strided<[16, 2], offset: 3>, 3>, %i: index, %j: index) kernel {
    // CHECK-DAG: %[[I32:.*]] = llvm.trunc %[[I]] : i64 to i32
    // CHECK-DAG: %[[J32:.*]] = llvm.trunc %[[J]] : i64 to i32
    // CHECK: %[[A1:.*]] = llvm.mul %[[I32]], %{{.*}} : i32
    // CHECK: %[[A2:.*]] = llvm.add %[[A1]], %{{.*}} : i32
    // CHECK: %[[A3:.*]] = llvm.mul %[[J32]], %{{.*}} : i32
    // CHECK: %[[L:.*]] = llvm.add %[[A2]], %[[A3]] : i32
    // CHECK: %[[LW:.*]] = llvm.zext %[[L]] : i32 to i64
    // CHECK: llvm.getelementptr inbounds|nuw %{{.*}}[%[[LW]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    // OFF: llvm.mul %[[I:.*]], %{{.*}} : i64
    %v = memref.load %a[%i, %j] : memref<4x8xf32, strided<[16, 2], offset: 3>, 3>
    %f = arith.addf %v, %v : f32
    memref.store %f, %a[%j, %i] : memref<4x8xf32, strided<[16, 2], offset: 3>, 3>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @too_large(
  gpu.func @too_large(%a: memref<65536x65537xf32, 1>, %i: index, %j: index) kernel {
    // CHECK: llvm.mul %{{.*}}, %{{.*}} : i64
    %v = memref.load %a[%i, %j] : memref<65536x65537xf32, 1>
    %f = arith.addf %v, %v : f32
    memref.store %f, %a[%j, %i] : memref<65536x65537xf32, 1>
    gpu.return
  }
}

// -----

// A uniform gpu.all_reduce in a kernel of whole SIMD-groups (known_block_size
// a multiple of 32) is the SIMD-group reduction, the partials in threadgroup
// memory, one barrier and each SIMD-group's reduction of the partials (lanes
// past the SIMD-group count read the identity of an addition, partial 0 of a
// maximum), with a second barrier where it may run again
// (simdgroup-all-reduce). One SIMD-group needs only its reduction; other
// sizes keep the generic lowering.
gpu.module @simdgroup_all_reduce {
  // CHECK: llvm.mlir.global internal @[[BUF:.*]]() {addr_space = 3 : i32} : !llvm.array<4 x f32>
  // CHECK-LABEL: llvm.func @sum(
  // OFF-LABEL: llvm.func @sum(
  gpu.func @sum(%a: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    // CHECK-DAG: %[[ID:.*]] = llvm.mlir.constant(-0.000000e+00 : f32) : f32
    // CHECK: %[[P:.*]] = llvm.call @air.simd_sum.f32(%{{.*}}) : (f32) -> f32
    // CHECK: llvm.call @air.simdgroup_index_in_threadgroup() : () -> (i32 {llvm.range = #llvm.constant_range<i32, 0, 4>})
    // CHECK: llvm.store %[[P]], %{{.*}} : f32, !llvm.ptr<3>
    // CHECK: llvm.call @air.wg.barrier
    // CHECK: %[[IN:.*]] = llvm.icmp "ult" %{{.*}}, %{{.*}} : i64
    // CHECK: %[[X:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> f32
    // CHECK: %[[Y:.*]] = llvm.select %[[IN]], %[[X]], %[[ID]] : i1, f32
    // CHECK: %[[R:.*]] = llvm.call @air.simd_sum.f32(%[[Y]]) : (f32) -> f32
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.store %[[R]], %{{.*}} : f32, !llvm.ptr<1>
    // OFF: llvm.call @air.simd_shuffle.u.i32
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xf32, 1>
    %s = gpu.all_reduce add %v uniform {} : (f32) -> (f32)
    memref.store %s, %a[%c0] : memref<1024xf32, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @max_loop(
  gpu.func @max_loop(%a: memref<1024xi32, 1>, %n: index) kernel
      attributes {known_block_size = array<i32: 16, 4, 1>} {
    // CHECK: llvm.call @air.simd_max.s.i32
    // CHECK: llvm.call @air.wg.barrier
    // CHECK: %[[X:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
    // CHECK: %[[R:.*]] = llvm.call @air.simd_max.s.i32(%[[X]]) : (i32) -> i32
    // CHECK: llvm.call @air.wg.barrier
    // CHECK: llvm.store %[[R]]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %a[%i] : memref<1024xi32, 1>
      %s = gpu.all_reduce maxsi %v uniform {} : (i32) -> (i32)
      memref.store %s, %a[%i] : memref<1024xi32, 1>
    }
    gpu.return
  }
  // CHECK-LABEL: llvm.func @one_group(
  gpu.func @one_group(%a: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 32, 1, 1>} {
    // CHECK: %[[R:.*]] = llvm.call @air.simd_product.f32(%{{.*}}) : (f32) -> f32
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.store %[[R]]
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xf32, 1>
    %s = gpu.all_reduce mul %v uniform {} : (f32) -> (f32)
    memref.store %s, %a[%c0] : memref<1024xf32, 1>
    gpu.return
  }
  // Consecutive reductions whose results are used only after the last of
  // them share one round: both partials stored under one lane-0 guard, one
  // barrier, then both reductions of the partials.
  // CHECK-LABEL: llvm.func @two_sums(
  gpu.func @two_sums(%a: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    // CHECK: %[[P:.*]] = llvm.call @air.simd_sum.f32(%{{.*}}) : (f32) -> f32
    // CHECK: %[[Q:.*]] = llvm.call @air.simd_sum.f32(%{{.*}}) : (f32) -> f32
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.store %[[P]], %{{.*}} : f32, !llvm.ptr<3>
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.store %[[Q]], %{{.*}} : f32, !llvm.ptr<3>
    // CHECK: llvm.call @air.wg.barrier
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: %[[R:.*]] = llvm.call @air.simd_sum.f32(%{{.*}}) : (f32) -> f32
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: %[[S:.*]] = llvm.call @air.simd_sum.f32(%{{.*}}) : (f32) -> f32
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.fadd %[[R]], %[[S]]
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xf32, 1>
    %s = gpu.all_reduce add %v uniform {} : (f32) -> (f32)
    %v2 = arith.mulf %v, %v : f32
    %q = gpu.all_reduce add %v2 uniform {} : (f32) -> (f32)
    %t = arith.addf %s, %q : f32
    memref.store %t, %a[%c0] : memref<1024xf32, 1>
    gpu.return
  }
  // maximumf of f32: the SIMD-group reductions are subgroup_reduce's
  // (the ordered keys' signed maximum), no shuffles.
  // CHECK-LABEL: llvm.func @maximum(
  gpu.func @maximum(%a: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 256, 1, 1>} {
    // CHECK-NOT: llvm.call @air.simd_shuffle
    // CHECK: llvm.call @air.simd_max.s.i32
    // CHECK: llvm.call @air.wg.barrier
    // CHECK-NOT: llvm.call @air.simd_shuffle
    // CHECK: llvm.call @air.simd_max.s.i32
    // CHECK-NOT: llvm.call @air.simd_shuffle
    // CHECK: llvm.return
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xf32, 1>
    %s = gpu.all_reduce maximumf %v uniform {} : (f32) -> (f32)
    memref.store %s, %a[%c0] : memref<1024xf32, 1>
    gpu.return
  }
  // Integers of 8, 16 and 64 bits as well (exact in any order): their partials
  // in threadgroup memory and one barrier, instead of the generic lowering.
  // CHECK-LABEL: llvm.func @sum_i64(
  gpu.func @sum_i64(%a: memref<1024xi64, 1>) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    // CHECK: llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<3>
    // CHECK: llvm.call @air.wg.barrier
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i64
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.return
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xi64, 1>
    %s = gpu.all_reduce add %v uniform {} : (i64) -> (i64)
    memref.store %s, %a[%c0] : memref<1024xi64, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @max_i16(
  gpu.func @max_i16(%a: memref<1024xi16, 1>) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    // CHECK: llvm.store %{{.*}}, %{{.*}} : i16, !llvm.ptr<3>
    // CHECK: llvm.call @air.wg.barrier
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i16
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.return
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xi16, 1>
    %s = gpu.all_reduce maxsi %v uniform {} : (i16) -> (i16)
    memref.store %s, %a[%c0] : memref<1024xi16, 1>
    gpu.return
  }
  // A reduction whose result feeds the next one keeps its own round.
  // CHECK-LABEL: llvm.func @dependent(
  gpu.func @dependent(%a: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 128, 1, 1>} {
    // CHECK: llvm.call @air.wg.barrier
    // CHECK: llvm.call @air.wg.barrier
    // CHECK-NOT: llvm.call @air.wg.barrier
    // CHECK: llvm.return
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xf32, 1>
    %s = gpu.all_reduce add %v uniform {} : (f32) -> (f32)
    %d = arith.subf %v, %s : f32
    %q = gpu.all_reduce add %d uniform {} : (f32) -> (f32)
    memref.store %q, %a[%c0] : memref<1024xf32, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @partial(
  gpu.func @partial(%a: memref<1024xf32, 1>) kernel
      attributes {known_block_size = array<i32: 48, 1, 1>} {
    // CHECK: llvm.call @air.simd_shuffle.u.i32
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<1024xf32, 1>
    %s = gpu.all_reduce add %v uniform {} : (f32) -> (f32)
    memref.store %s, %a[%c0] : memref<1024xf32, 1>
    gpu.return
  }
}

// -----

// A reduction over clusters of 8 or 16 lanes is AIR's quad-group reduction
// (lanes 1 and 2) and xor shuffles at 4 (and 8); over clusters of 2, one xor
// shuffle at 1: the order AIR's whole-group reductions add in.
gpu.module @clustered_reduce {
  // CHECK-LABEL: llvm.func @cluster8(
  gpu.func @cluster8(%a: memref<64xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xf32, 1>
    // CHECK: llvm.call @air.quad_sum.f32
    // CHECK: %[[M4:.*]] = llvm.mlir.constant(4 : i16) : i16
    // CHECK: llvm.call @air.simd_shuffle_xor.u.i32(%{{.*}}, %[[M4]])
    // CHECK: llvm.fadd
    // CHECK-NOT: simd_shuffle
    // CHECK: llvm.store
    %r = gpu.subgroup_reduce add %v cluster(size = 8) : (f32) -> f32
    memref.store %r, %a[%c0] : memref<64xf32, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @cluster2(
  gpu.func @cluster2(%a: memref<64xi32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xi32, 1>
    // CHECK: %[[M1:.*]] = llvm.mlir.constant(1 : i16) : i16
    // CHECK: llvm.call @air.simd_shuffle_xor.u.i32(%{{.*}}, %[[M1]])
    // CHECK: llvm.intr.umax
    // CHECK-NOT: simd_shuffle
    // CHECK: llvm.store
    %r = gpu.subgroup_reduce maxui %v cluster(size = 2) : (i32) -> i32
    memref.store %r, %a[%c0] : memref<64xi32, 1>
    gpu.return
  }
}

// -----

// Reductions AIR has no op for: of i64 (64-bit xor shuffles at 1 .. 16, the
// shuffle lowering splitting each), and over clusters of 8 of f16 (AIR's
// quad-group reduction, then a shuffle at 4).
gpu.module @shuffle_reduce {
  // CHECK-LABEL: llvm.func @i64max(
  gpu.func @i64max(%a: memref<64xi64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xi64, 1>
    // CHECK-COUNT-10: llvm.call @air.simd_shuffle_xor.u.i32
    %r = gpu.subgroup_reduce maxui %v : (i64) -> i64
    memref.store %r, %a[%c0] : memref<64xi64, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @f16cluster8(
  gpu.func @f16cluster8(%a: memref<64xf16, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xf16, 1>
    // CHECK: llvm.call @air.quad_sum.f16
    // CHECK: llvm.call @air.simd_shuffle_xor
    // CHECK: llvm.fadd %{{.*}} : f16
    %r = gpu.subgroup_reduce add %v cluster(size = 8) : (f16) -> f16
    memref.store %r, %a[%c0] : memref<64xf16, 1>
    gpu.return
  }
  // i8 and i16: AIR's i32 reduction of the values extended (signed for a
  // signed minimum or maximum), truncated; no shuffles.
  // CHECK-LABEL: llvm.func @narrow(
  gpu.func @narrow(%a: memref<64xi16, 1>, %b: memref<64xi8, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xi16, 1>
    %w = memref.load %b[%c0] : memref<64xi8, 1>
    // CHECK-NOT: simd_shuffle
    // CHECK: %[[S:.*]] = llvm.sext %{{.*}} : i16 to i32
    // CHECK: %[[M:.*]] = llvm.call @air.quad_min.s.i32(%[[S]])
    // CHECK: llvm.trunc %[[M]] : i32 to i16
    %r = gpu.subgroup_reduce minsi %v cluster(size = 4) : (i16) -> i16
    // CHECK-NOT: simd_shuffle
    // CHECK: %[[Z:.*]] = llvm.zext %{{.*}} : i8 to i32
    // CHECK: %[[A:.*]] = llvm.call @air.simd_sum.s.i32(%[[Z]])
    // CHECK: llvm.trunc %[[A]] : i32 to i8
    // CHECK-NOT: simd_shuffle
    %q = gpu.subgroup_reduce add %w : (i8) -> i8
    memref.store %r, %a[%c0] : memref<64xi16, 1>
    memref.store %q, %b[%c0] : memref<64xi8, 1>
    gpu.return
  }
}

// -----

// expm1 and log1p accurate for small arguments (MathToLLVM's exp(x) - 1 and
// log(1 + x) cancel to 0 there): for |x| < 1/2, (u - 1) x / log(u) with
// u = exp(x), and log(u) x / (u - 1) with u = 1 + x (x where u is 1); from
// 1/2, u - 1 and log(u). Half in float.
gpu.module @accurate_m1 {
  // CHECK-LABEL: llvm.func @expm1(
  gpu.func @expm1(%a: memref<8xf32, 1>, %h: memref<8xf16, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf32, 1>
    // CHECK: %[[U:.*]] = llvm.intr.exp(%[[X:.*]]) : (f32) -> f32
    // CHECK: %[[UM1:.*]] = llvm.fsub %[[U]], %{{.*}} : f32
    // CHECK: %[[P:.*]] = llvm.fmul %[[UM1]], %[[X]] : f32
    // CHECK: %[[L:.*]] = llvm.intr.log(%[[U]]) : (f32) -> f32
    // CHECK: llvm.fdiv %[[P]], %[[L]] : f32
    %r = math.expm1 %v : f32
    memref.store %r, %a[%c0] : memref<8xf32, 1>
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK: llvm.intr.log(%{{.*}}) : (f32) -> f32
    // CHECK: llvm.fptrunc %{{.*}} : f32 to f16
    %w = memref.load %h[%c0] : memref<8xf16, 1>
    %q = math.log1p %w : f16
    memref.store %q, %h[%c0] : memref<8xf16, 1>
    gpu.return
  }
}

// -----

// asinh, acosh, atanh, cbrt and erf, which MathToLLVM does not lower: the
// first three through an accurate log1p (no log(x + sqrt(x^2 +- 1))), cbrt
// as exp2(log2|x| / 3) and a Newton step, erf by the f32 polynomial; half in
// float.
gpu.module @inverse_hyperbolic {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%a: memref<8xf32, 1>, %h: memref<8xf16, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf32, 1>
    // CHECK-NOT: math.
    // CHECK: llvm.intr.log
    // CHECK: llvm.intr.log2
    // CHECK: llvm.intr.exp2
    %s = math.asinh %v : f32
    %t = math.cbrt %s : f32
    %u = math.erf %t : f32
    %w = memref.load %h[%c0] : memref<8xf16, 1>
    %x = math.atanh %w : f16
    %y = math.acosh %x : f16
    memref.store %u, %a[%c0] : memref<8xf32, 1>
    memref.store %y, %h[%c0] : memref<8xf16, 1>
    gpu.return
  }
}

// -----

// math.erfc (upstream's approximation, its last step without a flushed
// correction: exp(-a^2) times fma(r, t, r)), math.isinf (|x| == inf) and
// math.ipowi (a loop of squarings; 1, +-1 or 0 for a negative exponent).
gpu.module @erfc_isinf_ipowi {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%a: memref<8xf32, 1>, %h: memref<8xf16, 1>, %n: memref<8xi32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf32, 1>
    // CHECK-NOT: math.
    // CHECK: %[[S:.*]] = llvm.fmul %[[A:.*]], %[[A]]
    // CHECK: llvm.intr.exp
    // CHECK: %[[T:.*]] = llvm.intr.fma(%{{.*}}, %[[A]], %[[S]])
    // CHECK: %[[C:.*]] = llvm.intr.fma(%{{.*}}, %[[T]], %{{.*}})
    // CHECK: llvm.fmul %{{.*}}, %[[C]]
    %e = math.erfc %v : f32
    memref.store %e, %a[%c0] : memref<8xf32, 1>
    %w = memref.load %h[%c0] : memref<8xf16, 1>
    // CHECK: llvm.intr.fabs
    // CHECK: llvm.fcmp "oeq" %{{.*}}, %{{.*}} : f16
    %i = math.isinf %w : f16
    %j = arith.extui %i : i1 to i32
    %b = memref.load %n[%c0] : memref<8xi32, 1>
    // CHECK: llvm.br
    // CHECK: llvm.mul
    // CHECK: llvm.lshr
    %p = math.ipowi %b, %j : i32
    memref.store %p, %n[%c0] : memref<8xi32, 1>
    gpu.return
  }
}

// -----

// A binary search over index (lo from 0, hi any index, while lo < hi, a load
// of a static 2^20-element memref at mid = (lo + hi) >> 1 every iteration):
// its values stay below 2^31 (or the first load is out of bounds), so the
// loop runs in i32.
gpu.module @binary_search {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%keys: memref<1048576xi32, 1>, %sorted: memref<1048576xi32, 1>, %n: index, %m: index) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %key = memref.load %keys[%c0] : memref<1048576xi32, 1>
    // CHECK: llvm.icmp "ult" %{{.*}}, %{{.*}} : i32
    // CHECK: llvm.add %{{.*}}, %{{.*}} : i32
    // CHECK: llvm.lshr %{{.*}}, %{{.*}} : i32
    %r:2 = scf.while (%lo = %c0, %hi = %n) : (index, index) -> (index, index) {
      %go = arith.cmpi ult, %lo, %hi : index
      scf.condition(%go) %lo, %hi : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %s = arith.addi %l, %h : index
      %mid = arith.shrui %s, %c1 : index
      %v = memref.load %sorted[%mid] : memref<1048576xi32, 1>
      %lt = arith.cmpi slt, %v, %key : i32
      %m1 = arith.addi %mid, %c1 : index
      %nl = arith.select %lt, %m1, %l : index
      %nh = arith.select %lt, %h, %mid : index
      scf.yield %nl, %nh : index, index
    }
    // lo from an argument: not narrowed (lo + hi may wrap).
    // CHECK: llvm.icmp "ult" %{{.*}}, %{{.*}} : i64
    %q:2 = scf.while (%lo = %m, %hi = %n) : (index, index) -> (index, index) {
      %go = arith.cmpi ult, %lo, %hi : index
      scf.condition(%go) %lo, %hi : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %s = arith.addi %l, %h : index
      %mid = arith.shrui %s, %c1 : index
      %v = memref.load %sorted[%mid] : memref<1048576xi32, 1>
      %lt = arith.cmpi slt, %v, %key : i32
      %m1 = arith.addi %mid, %c1 : index
      %nl = arith.select %lt, %m1, %l : index
      %nh = arith.select %lt, %h, %mid : index
      scf.yield %nl, %nh : index, index
    }
    // a signed test: not narrowed
    // CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
    %u:2 = scf.while (%lo = %c0, %hi = %n) : (index, index) -> (index, index) {
      %go = arith.cmpi slt, %lo, %hi : index
      scf.condition(%go) %lo, %hi : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %s = arith.addi %l, %h : index
      %mid = arith.shrui %s, %c1 : index
      %v = memref.load %sorted[%mid] : memref<1048576xi32, 1>
      %lt = arith.cmpi slt, %v, %key : i32
      %m1 = arith.addi %mid, %c1 : index
      %nl = arith.select %lt, %m1, %l : index
      %nh = arith.select %lt, %h, %mid : index
      scf.yield %nl, %nh : index, index
    }
    %p0 = arith.addi %r#0, %q#0 : index
    %p = arith.addi %p0, %u#1 : index
    %pi = arith.index_cast %p : index to i32
    memref.store %pi, %keys[%c0] : memref<1048576xi32, 1>
    gpu.return
  }
}

// -----

// math.exp and math.log of f64: fdlibm's algorithms in double operations
// (the AIR emulation of doubles has no exp or log).
gpu.module @exp_log_f64 {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    // CHECK-NOT: llvm.intr.exp
    // CHECK: llvm.intr.roundeven
    // CHECK: llvm.fdiv
    // CHECK: llvm.shl
    // CHECK-NOT: llvm.intr.log
    // CHECK: llvm.fdiv
    // CHECK: llvm.sitofp
    %e = math.exp %v : f64
    %l = math.log %e : f64
    memref.store %l, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// The other f64 exponentials, logarithms and hyperbolic functions build on
// the f64 exp, log and expm1 above (no llvm.intr call is left for them).
gpu.module @more_f64_functions {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    // CHECK-NOT: llvm.intr.exp
    // CHECK-NOT: llvm.intr.log
    // CHECK-NOT: llvm.intr.cosh
    // CHECK-NOT: llvm.intr.sinh
    // CHECK-NOT: llvm.intr.tanh
    // CHECK: llvm.return
    %e1 = math.expm1 %v : f64
    %e2 = math.log1p %e1 : f64
    %e3 = math.exp2 %e2 : f64
    %e4 = math.log2 %e3 : f64
    %e5 = math.log10 %e4 : f64
    %e6 = math.cosh %e5 : f64
    %e7 = math.sinh %e6 : f64
    %e8 = math.tanh %e7 : f64
    memref.store %e8, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}
