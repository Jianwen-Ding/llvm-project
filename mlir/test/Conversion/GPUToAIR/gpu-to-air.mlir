// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 --split-input-file | FileCheck %s
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 narrow-loops=0 linearize-indices=0 simdgroup-all-reduce=0" --split-input-file | FileCheck %s --check-prefix=OFF
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 approximate-half-functions=0" --split-input-file | FileCheck %s --check-prefix=HALFOFF
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 expm1-f64-direct=0" --split-input-file | FileCheck %s --check-prefix=E1OFF
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 f64-polynomial-fma=1" --split-input-file | FileCheck %s --check-prefix=PFMA
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 log1p-f64-direct=0" --split-input-file | FileCheck %s --check-prefix=L1OFF

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
// says; other shuffles read the lane they compute (L15-E285: shifts within
// narrower segments and idx of a constant lane too, with a select of the own
// value).
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
    // Within segments of 8 lanes, a shift by a constant is the same
    // shuffle_down, and the thread's own value where the source lane is past
    // its segment's end; an idx of a constant lane, the shuffle of that lane
    // and the own value outside its segment (no per-thread lane). The tests:
    // lane % 8 < 8 - 2, and lane & -8 == 2 & -8 (L15-E306).
    // CHECK: %[[L8:.*]] = llvm.call @air.thread_index_in_simdgroup()
    // CHECK: %[[SEVEN:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK: %[[P8:.*]] = llvm.and %[[L8]], %[[SEVEN]]
    // CHECK: %[[SIX:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK: %[[OK8:.*]] = llvm.icmp "ult" %[[P8]], %[[SIX]]
    // CHECK: %[[D8:.*]] = llvm.call @air.simd_shuffle_down.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    // CHECK: llvm.select %[[OK8]], %[[D8]], %{{.*}} : i1, i32
    %d8, %okd8 = gpu.shuffle down %v, %c2, %c8 : i32
    // CHECK: %[[S8:.*]] = llvm.and %{{.*}}, %{{.*}} : i32
    // CHECK: %[[Z:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[OKI:.*]] = llvm.icmp "eq" %[[S8]], %[[Z]]
    // CHECK: %[[I8:.*]] = llvm.call @air.simd_shuffle.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    // CHECK: llvm.select %[[OKI]], %[[I8]], %{{.*}} : i1, i32
    %i8, %oki8 = gpu.shuffle idx %v, %c2, %c8 : i32
    // A uniform mask (a kernel argument): AIR's shuffle_xor of it, and the
    // own value where lane ^ mask leaves the segment (L15-E286).
    // CHECK: %[[M:.*]] = llvm.trunc %{{.*}} : i32 to i16
    // CHECK: llvm.xor
    // CHECK: %[[XO:.*]] = llvm.call @air.simd_shuffle_xor.u.i32(%{{.*}}, %[[M]]) : (i32, i16) -> i32
    // CHECK: llvm.select %{{.*}}, %[[XO]], %{{.*}} : i1, i32
    %xo, %okxo = gpu.shuffle xor %v, %o, %c32 : i32
    // Down by a uniform offset of either sign: shuffle_down by it, or
    // shuffle_up by its negation, by its sign (L15-E296).
    // CHECK: %[[DN:.*]] = llvm.call @air.simd_shuffle_down.u.i32
    // CHECK: %[[UP:.*]] = llvm.call @air.simd_shuffle_up.u.i32
    // CHECK: llvm.select %{{.*}}, %[[DN]], %[[UP]] : i1, i32
    %du, %okdu = gpu.shuffle down %v, %o, %c8 : i32
    // A per-thread mask: the general path, the lane it computes.
    // CHECK: %[[LD:.*]] = llvm.call @air.thread_index_in_simdgroup()
    // CHECK: llvm.call @air.simd_shuffle.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    %ln = gpu.lane_id
    %lo = arith.index_cast %ln : index to i32
    %xd, %okxd = gpu.shuffle xor %v, %lo, %c32 : i32
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
    %c5 = arith.constant 5 : index
    memref.store %i8, %out[%c5] : memref<8xi32, 1>
    %c6 = arith.constant 6 : index
    memref.store %xd, %out[%c6] : memref<8xi32, 1>
    %c7 = arith.constant 7 : index
    memref.store %du, %out[%c7] : memref<8xi32, 1>
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
  // A loop by 1 between i32 values cast to index (a CSR row's) is the loop
  // over i32 between them, compared signed for signed casts; an iv that
  // indexes a memref each iteration is not negative, cast unsigned.
  // CHECK-LABEL: llvm.func @csr_row(
  // CHECK-SAME: %[[B:[^:]*]]: i32, %[[E:[^:]*]]: i32)
  // CHECK: llvm.br ^bb1(%[[B]], %{{.*}} : i32, f32)
  // CHECK: ^bb1(%[[J:.*]]: i32, %{{.*}}: f32):
  // CHECK: llvm.icmp "slt" %[[J]], %[[E]] : i32
  // CHECK: llvm.zext %[[J]] : i32 to i64
  // CHECK: llvm.add %[[J]], %{{.*}} : i32
  // OFF-LABEL: llvm.func @csr_row(
  // OFF: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
  gpu.func @csr_row(%a: memref<1048576xf32, 1>, %o: memref<1xf32, 1>, %b: i32, %e: i32) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %lb = arith.index_cast %b : i32 to index
    %ub = arith.index_cast %e : i32 to index
    %s = scf.for %j = %lb to %ub step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %a[%j] : memref<1048576xf32, 1>
      %r = arith.addf %acc, %v : f32
      scf.yield %r : f32
    }
    memref.store %s, %o[%c0] : memref<1xf32, 1>
    gpu.return
  }
  // From a constant (-8: signed), the iv's index arithmetic is 32-bit;
  // unsigned casts give an unsigned loop.
  // CHECK-LABEL: llvm.func @i32_bounds(
  // CHECK-SAME: %[[N:[^:]*]]: i32, %[[U:[^:]*]]: i32)
  // CHECK: %[[M8:.*]] = llvm.mlir.constant(-8 : i32) : i32
  // CHECK: llvm.br ^bb1(%[[M8]] : i32)
  // CHECK: ^bb1(%[[J:.*]]: i32):
  // CHECK: llvm.icmp "slt" %[[J]], %[[N]] : i32
  // CHECK: %[[J1:.*]] = llvm.add %[[J]], %{{.*}} : i32
  // CHECK: llvm.mul %[[J1]], %[[J1]] : i32
  // CHECK: llvm.add %[[J]], %{{.*}} overflow<nsw> : i32
  // CHECK: ^bb4(%[[K:.*]]: i32):
  // CHECK: llvm.icmp "ult" %[[K]], %[[U]] : i32
  // CHECK: llvm.add %[[K]], %{{.*}} overflow<nuw> : i32
  gpu.func @i32_bounds(%a: memref<1048576xf32, 1>, %n: i32, %u: i32) kernel {
    %cm8 = arith.constant -8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f = arith.constant 1.0 : f32
    %ub = arith.index_cast %n : i32 to index
    scf.for %j = %cm8 to %ub step %c1 {
      %k = arith.addi %j, %c1 : index
      %k2 = arith.muli %k, %k : index
      memref.store %f, %a[%k2] : memref<1048576xf32, 1>
    }
    %uu = arith.index_castui %u : i32 to index
    scf.for %j = %c0 to %uu step %c1 {
      memref.store %f, %a[%j] : memref<1048576xf32, 1>
    }
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

// Reductions AIR has no op for: an i64 product (64-bit xor shuffles at 1 ..
// 16, the shuffle lowering splitting each; i64 minima, maxima, sums and
// bitwise reductions are i32 reductions, L15-E442 / E443), and over clusters
// of 8 of f16 (AIR's quad-group reduction, then a shuffle at 4).
gpu.module @shuffle_reduce {
  // CHECK-LABEL: llvm.func @i64mul(
  gpu.func @i64mul(%a: memref<64xi64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xi64, 1>
    // CHECK-COUNT-10: llvm.call @air.simd_shuffle_xor.u.i32
    %r = gpu.subgroup_reduce mul %v : (i64) -> i64
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
  // maximumf / minimumf of f16: the f32 ordered-key reduction of the values
  // extended, truncated; no shuffles.
  // CHECK-LABEL: llvm.func @f16maximum(
  gpu.func @f16maximum(%a: memref<64xf16, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xf16, 1>
    // CHECK-NOT: simd_shuffle
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK: llvm.call @air.simd_max.s.i32
    // CHECK: llvm.fptrunc %{{.*}} : f32 to f16
    // CHECK-NOT: simd_shuffle
    %r = gpu.subgroup_reduce maximumf %v : (f16) -> f16
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
// float. asinh and acosh take one log, of 1 + t below 4096 and of |x| (x)
// above, for log1p's and log |x| + ln 2's two (L15-E308).
gpu.module @inverse_hyperbolic {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%a: memref<8xf32, 1>, %h: memref<8xf16, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf32, 1>
    // CHECK-NOT: math.
    // CHECK: %[[NEAR:.*]] = llvm.fcmp "olt" %[[ABS:.*]], %{{.*}} : f32
    // CHECK: %[[ARG:.*]] = llvm.select %[[NEAR]], %{{.*}}, %[[ABS]] : i1, f32
    // CHECK: %[[LOG:.*]] = llvm.intr.log(%[[ARG]]) : (f32) -> f32
    // CHECK-NOT: llvm.intr.log(
    // CHECK: llvm.select %[[NEAR]], %{{.*}}, %{{.*}} : i1, f32
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
    // y 2^k1 on y's exponent field, then one multiplication by 2^k2
    // (L15-E326)
    // CHECK-NEXT: llvm.fsub
    // CHECK-NEXT: llvm.fsub
    // CHECK-NEXT: %[[Y:.*]] = llvm.fsub
    // CHECK-NEXT: %[[K:.*]] = llvm.fptosi
    // CHECK: %[[K1:.*]] = llvm.ashr %[[K]]
    // CHECK: %[[YB:.*]] = llvm.bitcast %[[Y]] : f64 to i64
    // CHECK: %[[KS:.*]] = llvm.shl %[[K1]]
    // CHECK: %[[SB:.*]] = llvm.add %[[YB]], %[[KS]] : i64
    // CHECK: %[[S:.*]] = llvm.bitcast %[[SB]] : i64 to f64
    // CHECK: %[[NAN:.*]] = llvm.fcmp "uno" %[[Y]], %[[Y]]
    // CHECK: %[[SY:.*]] = llvm.select %[[NAN]], %[[Y]], %[[S]]
    // CHECK: llvm.fmul %[[SY]], %{{.*}} : f64
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

// -----

// f64 cosh and sinh: one exp, of |x| below 709 and of |x| / 2 from it
// (sinh's expm1(|x|) from it too) (L15-E309).
gpu.module @f64_cosh_sinh {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    // CHECK: %[[A:.*]] = llvm.intr.fabs(%{{.*}}) : (f64) -> f64
    // CHECK: %[[BELOW:.*]] = llvm.fcmp "olt" %[[A]], %{{.*}} : f64
    // CHECK: %[[H:.*]] = llvm.fmul %[[A]], %{{.*}} : f64
    // CHECK: llvm.select %[[BELOW]], %[[A]], %[[H]] : i1, f64
    // CHECK-NOT: llvm.intr.fabs
    // CHECK: llvm.select %[[BELOW]]
    %e = math.cosh %v : f64
    memref.store %e, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @s(
  gpu.func @s(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    // CHECK: %[[A:.*]] = llvm.intr.fabs(%{{.*}}) : (f64) -> f64
    // CHECK: %[[BELOW:.*]] = llvm.fcmp "olt" %[[A]], %{{.*}} : f64
    // CHECK: %[[H:.*]] = llvm.fmul %[[A]], %{{.*}} : f64
    // CHECK: llvm.select %[[BELOW]], %[[A]], %[[H]] : i1, f64
    // CHECK: llvm.select %[[BELOW]]
    %e = math.sinh %v : f64
    memref.store %e, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// arith.floordivsi by 2^k is the arithmetic shift by k; ceildivsi by 2^k that
// shift plus 1 where a low bit is set (L15-E290).
gpu.module @pow2_floor_ceil_div {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xi32, 1>, %v: i32) kernel {
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c6 = arith.constant 6 : i32
    // CHECK: %[[F:.*]] = llvm.ashr %{{.*}}, %{{.*}} : i32
    // CHECK-NOT: llvm.sdiv
    %f = arith.floordivsi %v, %c8 : i32
    // CHECK: %[[S:.*]] = llvm.ashr %{{.*}}, %{{.*}} : i32
    // CHECK: %[[L:.*]] = llvm.and %{{.*}}, %{{.*}} : i32
    // CHECK: llvm.icmp "ne" %[[L]]
    // CHECK: llvm.add %[[S]]
    // CHECK-NOT: llvm.sdiv
    %g = arith.ceildivsi %v, %c16 : i32
    // Another positive constant: s ^ ((x ^ s) / d), s the sign splat
    // (L15-E293).
    // CHECK: %[[SG:.*]] = llvm.ashr %{{.*}}, %{{.*}} : i32
    // CHECK: %[[XS:.*]] = llvm.xor %{{.*}}, %[[SG]] : i32
    // CHECK: %[[Q:.*]] = llvm.sdiv %[[XS]], %{{.*}} : i32
    // CHECK: llvm.xor %[[Q]], %[[SG]] : i32
    %h = arith.floordivsi %v, %c6 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %f, %out[%c0] : memref<8xi32, 1>
    memref.store %g, %out[%c1] : memref<8xi32, 1>
    memref.store %h, %out[%c2] : memref<8xi32, 1>
    gpu.return
  }
}

// -----

// A shuffle by the induction variable of a loop with uniform bounds (a
// broadcast of each lane in turn) is a uniform shuffle of that lane, not a
// per-thread source lane (L15-E297).
gpu.module @loop_uniform_shuffle {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xf32, 1>, %v: f32) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %w = arith.constant 32 : i32
    %zero = arith.constant 0.0 : f32
    // CHECK: %[[L:.*]] = llvm.trunc %{{.*}} : i32 to i16
    // CHECK: llvm.call @air.simd_shuffle.u.i32(%{{.*}}, %[[L]])
    // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, f32
    %s = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %zero) -> f32 {
      %ki = arith.index_cast %k : index to i32
      %b, %ok = gpu.shuffle idx %v, %ki, %w : f32
      %acc2 = arith.addf %acc, %b : f32
      scf.yield %acc2 : f32
    }
    memref.store %s, %out[%c0] : memref<8xf32, 1>
    gpu.return
  }
}

// -----

// gpu.rotate: a shuffle down by the offset and up by width - offset, both AIR
// shifts, and a select by the lane's position in its segment; valid below
// width. gpu.subgroup_broadcast: AIR's shuffle of the lane, or its broadcast
// of the first active lane (L15-E298).
gpu.module @rotate_broadcast {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xf32, 1>, %v: f32, %l: i32) kernel {
    // CHECK: llvm.call @air.simd_shuffle_down.u.i32
    // CHECK: llvm.call @air.simd_shuffle_up.u.i32
    // CHECK: llvm.select
    %r, %ok = gpu.rotate %v, 1, 8 : f32
    // CHECK: llvm.call @air.simd_shuffle.u.i32
    %b = gpu.subgroup_broadcast %v, specific_lane %l : f32
    // CHECK: llvm.call @air.simd_broadcast_first.u.i32
    %f = gpu.subgroup_broadcast %v, first_active_lane : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %r, %out[%c0] : memref<8xf32, 1>
    memref.store %b, %out[%c1] : memref<8xf32, 1>
    memref.store %f, %out[%c2] : memref<8xf32, 1>
    gpu.return
  }
}

// -----

// A shuffle offset a loop carries from a uniform start through uniform
// updates (a scan's doubling offset) is uniform too (L15-E299).
gpu.module @loop_carried_offset {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%out: memref<8xi32, 1>, %v: i32) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %one = arith.constant 1 : i32
    %w = arith.constant 32 : i32
    // CHECK: llvm.call @air.simd_shuffle_up.u.i32
    // CHECK-NOT: llvm.call @air.simd_shuffle.u.i32
    %r:2 = scf.for %k = %c0 to %c5 step %c1 iter_args(%x = %v, %off = %one) -> (i32, i32) {
      %u, %ok = gpu.shuffle up %x, %off, %w : i32
      %x2 = arith.addi %x, %u : i32
      %off2 = arith.shli %off, %one : i32
      scf.yield %x2, %off2 : i32, i32
    }
    memref.store %r#0, %out[%c0] : memref<8xi32, 1>
    gpu.return
  }
}

// -----

// f64 sinh and tanh: one division each of their own beside those of the exp
// and expm1 they use (sinh's t^2 or t by t + 1, tanh's t or 2 by t + 2, the
// dividend selected: L15-E323), and the expm1 fdlibm's, with one division
// (L15-E814): 4 divisions in all; 8 with exp, log and Kahan's division
// (expm1-f64-direct=0), 10 before L15-E323.
gpu.module @f64_sinh_tanh_one_division {
  // CHECK-LABEL: llvm.func @k(
  // E1OFF-LABEL: gpu.module @f64_sinh_tanh_one_division
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    // CHECK-COUNT-4: llvm.fdiv
    // CHECK-NOT: llvm.fdiv
    // CHECK: llvm.return
    // E1OFF-COUNT-8: llvm.fdiv
    // E1OFF-NOT: llvm.fdiv
    // E1OFF: llvm.return
    %s = math.sinh %v : f64
    %t = math.tanh %s : f64
    memref.store %t, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 exp's polynomial in fdlibm's multiplies and adds, not fmas (the
// emulated fma costs more than both: L15-E815); f64-polynomial-fma=1: fmas.
gpu.module @f64_polynomial_mul_add {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: llvm.intr.fma
  // CHECK: llvm.return
  // PFMA-LABEL: gpu.module @f64_polynomial_mul_add
  // PFMA-COUNT-4: llvm.intr.fma
  // PFMA-NOT: llvm.intr.fma
  // PFMA: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = math.exp %v : f64
    memref.store %e, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 log1p as fdlibm's (L15-E818): one division of doubles (log's s), the
// correction c / u in f32 by the hardware's reciprocal; log1p-f64-direct=0:
// log(1 + x) and Kahan's division, two.
gpu.module @f64_log1p {
  // CHECK-LABEL: llvm.func @k(
  // CHECK: llvm.fdiv {{.*}} fastmath<afn> : f32
  // CHECK: llvm.fdiv {{.*}} : f64
  // CHECK-NOT: llvm.fdiv
  // CHECK: llvm.return
  // L1OFF-LABEL: gpu.module @f64_log1p
  // L1OFF-COUNT-2: llvm.fdiv {{.*}} : f64
  // L1OFF-NOT: llvm.fdiv
  // L1OFF: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = math.log1p %v : f64
    memref.store %e, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 asinh, acosh, atanh and cbrt (L15-E820; none lowered before): fdlibm's
// forms from f64 log1p and sqrt: asinh 1 division and log1p's, acosh log1p's,
// atanh 1 and log1p's, cbrt 3 (8 of doubles; log1p's f32 corrections 3).
gpu.module @f64_inverse_hyperbolic {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: math.
  // CHECK-COUNT-8: llvm.fdiv {{.*}} : f64
  // CHECK-NOT: llvm.fdiv {{.*}} : f64
  // CHECK-NOT: math.
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = math.asinh %v : f64
    %f = math.acosh %e : f64
    %g = math.atanh %f : f64
    %h = math.cbrt %g : f64
    memref.store %h, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 atan, atan2, asin and acos (L15-E821; llvm.atan.f64 and the others had
// no emulation): fdlibm's forms, atan one division (its operands selected),
// atan2 y / x and its atan's, asin and acos their rational's and the square
// root's correction (7), no intrinsic call of theirs left.
gpu.module @f64_inverse_trig {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: llvm.intr.{{a(tan|sin|cos)}}
  // CHECK-COUNT-7: llvm.fdiv {{.*}} : f64
  // CHECK-NOT: llvm.fdiv
  // CHECK-NOT: llvm.intr.{{a(tan|sin|cos)}}
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = math.atan %v : f64
    %f = math.atan2 %e, %v : f64
    %g = math.asin %f : f64
    %h = math.acos %g : f64
    memref.store %h, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 sin, cos and tan (L15-E822; llvm.sin.f64 and the others had no
// emulation): fdlibm's kernels on x reduced by pi/2, Cody and Waite's below
// 2^19 pi/2, Payne and Hanek's from it in a branch of its own (its window of
// 2/pi's bits, its product's normalization and error fma); tan's one
// division. No intrinsic call of theirs left.
gpu.module @f64_trig {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: llvm.intr.{{(sin|cos|tan)}}(
  // CHECK: llvm.cond_br
  // CHECK: llvm.intr.ctlz
  // CHECK: llvm.intr.fma
  // CHECK: llvm.intr.roundeven
  // CHECK: llvm.cond_br
  // CHECK: llvm.cond_br
  // CHECK: llvm.fdiv {{.*}} : f64
  // CHECK-NOT: llvm.fdiv
  // CHECK-NOT: llvm.intr.{{(sin|cos|tan)}}(
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = math.sin %v : f64
    %f = math.cos %e : f64
    %g = math.tan %f : f64
    memref.store %g, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 pow (L15-E823; llvm.pow.f64 had no emulation): fdlibm's e_pow.c,
// branch-free: log2|x| in two parts (a reciprocal), times y in two parts,
// 2^z by exp's rational (a division); no intrinsic call of its left.
gpu.module @f64_pow {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: llvm.intr.pow
  // CHECK-COUNT-2: llvm.fdiv {{.*}} : f64
  // CHECK-NOT: llvm.fdiv
  // CHECK-NOT: llvm.intr.pow
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = math.powf %v, %v : f64
    memref.store %e, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 erf and erfc (L15-E824; before, left unlowered): fdlibm's s_erf.c with
// its four rationals as one of selected coefficients (1/x^2, the quotient),
// the tail's two exponentials as one (its division) and r / |x|: 4 divisions
// each, no math op left.
gpu.module @f64_erf {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: math.
  // CHECK-COUNT-8: llvm.fdiv {{.*}} : f64
  // CHECK-NOT: llvm.fdiv
  // CHECK-NOT: math.
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = math.erf %v : f64
    %f = math.erfc %e : f64
    memref.store %f, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// f64 math on vectors (L15-E825): one scalar op an element, which the f64
// rewrites expand (before, MathToLLVM's vector intrinsic was scalarized into
// llvm.exp.f64 calls that the AIR target's f64 emulation does not take).
gpu.module @f64_vector_math {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: llvm.intr.exp
  // CHECK-COUNT-2: llvm.intr.roundeven({{.*}}) : (f64) -> f64
  // CHECK-NOT: llvm.intr.exp
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0] : memref<8xf64, 1>, vector<2xf64>
    %e = math.exp %v : vector<2xf64>
    vector.store %e, %a[%c0] : memref<8xf64, 1>, vector<2xf64>
    gpu.return
  }
}

// -----

// gpu.subgroup_reduce of f64 (L15-E826): AIR has no SIMD-group reduction of
// doubles (it was air.simd_sum.s.i64 of the double, an integer sum of its
// bits that the backend could not select): a butterfly of 32-bit shuffles of
// both halves and the emulated adds, as for i64.
gpu.module @f64_subgroup_reduce {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: air.simd_sum
  // CHECK-COUNT-5: llvm.fadd {{.*}} : f64
  // CHECK-NOT: air.simd_sum
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf64, 1>
    %e = gpu.subgroup_reduce add %v : (f64) -> f64
    memref.store %e, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// vector.reduction of f64 (L15-E827): its scalar operations in order (LLVM's
// ordered llvm.vector.reduce.fadd.v4f64 had no f64 emulation in the AIR
// target).
gpu.module @f64_vector_reduction {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: llvm.intr.vector.reduce
  // CHECK-COUNT-3: llvm.fadd {{.*}} : f64
  // CHECK-NOT: llvm.intr.vector.reduce
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0] : memref<8xf64, 1>, vector<4xf64>
    %e = vector.reduction <add>, %v : vector<4xf64> into f64
    memref.store %e, %a[%c0] : memref<8xf64, 1>
    gpu.return
  }
}

// -----

// n-D vectors (L15-E828): shape casts, transposes and contractions lowered to
// 1-D vectors first, as ConvertVectorToLLVMPass does (before, they were left
// in the output); the contraction's outer products as fmuladds.
#nd_map0 = affine_map<(i, j, k) -> (i, k)>
#nd_map1 = affine_map<(i, j, k) -> (k, j)>
#nd_map2 = affine_map<(i, j, k) -> (i, j)>
gpu.module @nd_vectors {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: vector.
  // CHECK: llvm.intr.fmuladd
  // CHECK-NOT: vector.
  // CHECK: llvm.return
  gpu.func @k(%a: memref<64xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0] : memref<64xf32, 1>, vector<4xf32>
    %m = vector.shape_cast %v : vector<4xf32> to vector<2x2xf32>
    %t = vector.transpose %m, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
    %z = arith.constant dense<0.0> : vector<2x2xf32>
    %r = vector.contract {indexing_maps = [#nd_map0, #nd_map1, #nd_map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %m, %t, %z : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
    %f = vector.shape_cast %r : vector<2x2xf32> to vector<4xf32>
    vector.store %f, %a[%c0] : memref<64xf32, 1>, vector<4xf32>
    gpu.return
  }
}

// -----

// complex functions with no ComplexToLLVM pattern (L15-E829): ComplexToStandard's
// forms on the parts (complex.exp: e^re cos(im), e^re sin(im)); the ops
// ComplexToLLVM lowers stay its own.
gpu.module @complex_functions {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-NOT: complex.
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xcomplex<f32>, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xcomplex<f32>, 1>
    %e = complex.exp %v : complex<f32>
    %f = complex.mul %e, %v : complex<f32>
    memref.store %f, %a[%c0] : memref<8xcomplex<f32>, 1>
    gpu.return
  }
}

// -----

// gpu.dynamic_shared_memory (L15-E830; before, illegal): an external-size
// threadgroup global the backend places after the static ones.
gpu.module @dynamic_shared_memory {
  // CHECK: llvm.mlir.global internal @__dynamic_shmem__0() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  // CHECK-LABEL: llvm.func @k(
  // CHECK: llvm.mlir.addressof @__dynamic_shmem__0 : !llvm.ptr<3>
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %sh = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %v = memref.view %sh[%c0][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<8xf32, #gpu.address_space<workgroup>>
    %x = memref.load %a[%c0] : memref<8xf32, 1>
    memref.store %x, %v[%c0] : memref<8xf32, #gpu.address_space<workgroup>>
    gpu.return
  }
}

// -----

// An i1 in memory (L15-E831; AIR has none): a byte, loaded and truncated,
// stored zero-extended, its GEPs over i8.
gpu.module @i1_memory {
  // CHECK-LABEL: llvm.func @k(
  // CHECK: llvm.getelementptr {{.*}} -> !llvm.ptr<1>, i8
  // CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> i8
  // CHECK: llvm.trunc {{.*}} : i8 to i1
  // CHECK: llvm.zext {{.*}} : i1 to i8
  // CHECK: llvm.store {{.*}} : i8, !llvm.ptr<1>
  // CHECK-NOT: -> i1
  // CHECK: llvm.return
  gpu.func @k(%a: memref<8xi1, 1>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v = memref.load %a[%c1] : memref<8xi1, 1>
    %t = arith.constant true
    %e = arith.xori %v, %t : i1
    memref.store %e, %a[%c0] : memref<8xi1, 1>
    gpu.return
  }
}

// -----

// f16 functions computed in f32 with the hardware's approximations of the
// steps for which every f16 input's result is the same (L15-E330): cbrt all
// of them, asinh its square root only; off with approximate-half-functions=0.
gpu.module @half_approximations {
  // CHECK-LABEL: llvm.func @k(
  // HALFOFF-LABEL: gpu.module @half_approximations
  // HALFOFF-NOT: fastmath<afn>
  // HALFOFF: llvm.return
  gpu.func @k(%a: memref<8xf16, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf16, 1>
    // CHECK: llvm.intr.log2(%{{.*}}) fastmath<afn> : (f32) -> f32
    // CHECK: llvm.intr.exp2(%{{.*}}) fastmath<afn> : (f32) -> f32
    // CHECK: llvm.fdiv %{{.*}}, %{{.*}} fastmath<afn> : f32
    // CHECK: llvm.intr.sqrt(%{{.*}}) fastmath<afn> : (f32) -> f32
    // CHECK-NOT: fastmath<afn>
    // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : f32
    // CHECK: llvm.return
    %c = math.cbrt %v : f16
    %s = math.asinh %c : f16
    memref.store %s, %a[%c0] : memref<8xf16, 1>
    gpu.return
  }
}

// -----

// f16 erfc (widened to f32): its two divisions by the reciprocal (afn);
// f32 erfc keeps the correctly rounded ones (L15-E337).
gpu.module @half_erfc {
  // CHECK-LABEL: llvm.func @k(
  gpu.func @k(%a: memref<8xf16, 1>, %b: memref<8xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf16, 1>
    %w = memref.load %b[%c0] : memref<8xf32, 1>
    // CHECK: llvm.fdiv %{{.*}}, %{{.*}} fastmath<afn> : f32
    // CHECK: llvm.fdiv %{{.*}}, %{{.*}} fastmath<afn> : f32
    // CHECK: llvm.fptrunc
    // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : f32
    // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : f32
    // CHECK: llvm.return
    %e = math.erfc %v : f16
    %f = math.erfc %w : f32
    memref.store %e, %a[%c0] : memref<8xf16, 1>
    memref.store %f, %b[%c0] : memref<8xf32, 1>
    gpu.return
  }
}

// -----

// The upper_bounds of positions read in the kernel's entry block before any
// op with regions or a call hold for every execution: AGX hints, one
// "x,y,z" per AIR system value, 0 where unknown (the backend's loop
// analyses bound the position registers by them, L15-E391). A read after
// a loop gives none.
gpu.module @upper_bounds {
  // CHECK-LABEL: llvm.func @entry(
  // CHECK-SAME: ["agx-upper-bounds:air.threadgroup_position_in_grid", "256,0,0"]
  // CHECK-SAME: ["agx-upper-bounds:air.thread_position_in_grid", "0,1024,0"]
  gpu.func @entry(%out: memref<8xindex, 1>) kernel {
    %b = gpu.block_id x upper_bound 256
    %g = gpu.global_id y upper_bound 1024
    %s = arith.addi %b, %g : index
    %c0 = arith.constant 0 : index
    memref.store %s, %out[%c0] : memref<8xindex, 1>
    gpu.return
  }
  // CHECK-LABEL: llvm.func @after_loop(
  // CHECK-NOT: agx-upper-bounds
  // CHECK: llvm.return
  gpu.func @after_loop(%out: memref<8xindex, 1>, %n: index) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      memref.store %i, %out[%c0] : memref<8xindex, 1>
    }
    %b = gpu.block_id x upper_bound 256
    memref.store %b, %out[%c1] : memref<8xindex, 1>
    gpu.return
  }
}

// -----

// i64 minimum / maximum reductions: two i32 AIR reductions, the high words'
// in the operation's signedness, then the unsigned low words' of the lanes
// whose high word is the result (L15-E442), no shuffles.
gpu.module @k {
  // CHECK-LABEL: llvm.func @reduce_i64
  gpu.func @reduce_i64(%b: memref<64xi64, 1>, %o: memref<64xi64, 1>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %w = memref.load %b[%c0] : memref<64xi64, 1>
    // CHECK-NOT: shuffle
    // CHECK: %[[H:.*]] = llvm.call @air.simd_max.s.i32(%[[HI:.*]]) : (i32) -> i32
    // CHECK: %[[EQ:.*]] = llvm.icmp "eq" %[[HI]], %[[H]] : i32
    // CHECK: %[[LO:.*]] = llvm.select %[[EQ]], %{{.*}}, %{{.*}} : i1, i32
    // CHECK: llvm.call @air.simd_max.u.i32(%[[LO]]) : (i32) -> i32
    %m = gpu.subgroup_reduce maxsi %w : (i64) -> i64
    // CHECK: llvm.call @air.quad_min.u.i32
    // CHECK: llvm.call @air.quad_min.u.i32
    // CHECK-NOT: shuffle
    %n = gpu.subgroup_reduce minui %w cluster(size = 4) : (i64) -> i64
    // A sum: the high words', and the low words' two 16-bit halves' (no
    // carry lost: below 2^21 over 32 lanes); xor: each word's (L15-E443).
    // CHECK-COUNT-3: llvm.call @air.simd_sum.s.i32
    // CHECK-COUNT-2: llvm.call @air.simd_xor.s.i32
    // CHECK-NOT: shuffle
    %s = gpu.subgroup_reduce add %w : (i64) -> i64
    %x = gpu.subgroup_reduce xor %w : (i64) -> i64
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    memref.store %m, %o[%c0] : memref<64xi64, 1>
    memref.store %n, %o[%c1] : memref<64xi64, 1>
    memref.store %s, %o[%c2] : memref<64xi64, 1>
    memref.store %x, %o[%c3] : memref<64xi64, 1>
    gpu.return
  }
}

// -----

// Views of memrefs (L15-E840): a subview, an expand_shape and a
// collapse_shape as their sources' pointers with offsets computed in i64 (the
// subview's from its dynamic offsets, r * 64 + q), no memref op left.
gpu.module @memref_views {
  // CHECK-LABEL: gpu.module @memref_views
  // CHECK: llvm.func @k(
  // CHECK: %[[RQ:.*]] = llvm.mul %arg2, %{{.*}} overflow<nsw> : i64
  // CHECK: llvm.add %[[RQ]], %arg3 : i64
  // CHECK-NOT: memref.
  // CHECK: llvm.return
  gpu.func @k(%a: memref<16x64xf32, 1>, %b: memref<256xf32, 1>, %r: index, %q: index) kernel {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %s = memref.subview %a[%r, %q] [4, 8] [2, 3] : memref<16x64xf32, 1> to memref<4x8xf32, strided<[128, 3], offset: ?>, 1>
    %v = memref.load %s[%c1, %c2] : memref<4x8xf32, strided<[128, 3], offset: ?>, 1>
    %e = memref.expand_shape %b [[0, 1]] output_shape [16, 16] : memref<256xf32, 1> into memref<16x16xf32, 1>
    %w = memref.load %e[%c2, %c1] : memref<16x16xf32, 1>
    %c = memref.collapse_shape %a [[0, 1]] : memref<16x64xf32, 1> into memref<1024xf32, 1>
    %x = arith.addf %v, %w : f32
    memref.store %x, %c[%q] : memref<1024xf32, 1>
    gpu.return
  }
}

// -----

// A SIMD-group reduction of a vector as reductions of its elements, and a
// constant mask as a constant (L15-E840).
gpu.module @subgroup_reduce_vector {
  // CHECK-LABEL: gpu.module @subgroup_reduce_vector
  // CHECK: llvm.func @k(
  // CHECK: llvm.mlir.constant(dense<[true, false]> : vector<2xi1>)
  // CHECK: llvm.extractelement
  // CHECK: llvm.call @air.simd_sum.f32(
  // CHECK: llvm.extractelement
  // CHECK: llvm.call @air.simd_sum.f32(
  // CHECK: llvm.intr.masked.store
  gpu.func @k(%a: memref<64xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0] : memref<64xf32, 1>, vector<2xf32>
    %r = gpu.subgroup_reduce add %v : (vector<2xf32>) -> vector<2xf32>
    %m = vector.constant_mask [1] : vector<2xi1>
    vector.maskedstore %a[%c0], %m, %r : memref<64xf32, 1>, vector<2xi1>, vector<2xf32>
    gpu.return
  }
}

// -----

// The affine dialect's ops (an outlined kernel's index maps, loads and
// stores) to arith and memref before the conversion (L15-E842): d0 * 2 + 1,
// min(d0 floordiv 3, 50) by a compare and select.
gpu.module @affine_ops {
  // CHECK-LABEL: gpu.module @affine_ops
  // CHECK: llvm.func @k(
  // CHECK-NOT: affine.
  // CHECK: llvm.icmp "slt"
  // CHECK: llvm.select
  // CHECK: llvm.load
  // CHECK: llvm.store
  // CHECK-NOT: affine.
  // CHECK: llvm.return
  gpu.func @k(%a: memref<256xf32, 1>, %i: index) kernel {
    %c1 = arith.constant 1 : index
    %j = affine.apply affine_map<(d0)[s0] -> (d0 * 2 + s0)>(%i)[%c1]
    %k = affine.min affine_map<(d0) -> (d0 floordiv 3, 50)>(%i)
    %v = affine.load %a[%j] : memref<256xf32, 1>
    affine.store %v, %a[%k] : memref<256xf32, 1>
    gpu.return
  }
}

// -----

// A 1-D transfer that may run out of bounds as a masked load (its mask from
// the memref's size), an in-bounds one as a plain store, and cf.assert as a
// trap where its condition fails, not a call of puts and abort (L15-E843).
gpu.module @assert_transfer {
  // CHECK-LABEL: gpu.module @assert_transfer
  // CHECK-NOT: llvm.func @abort
  // CHECK: llvm.func @k(
  // CHECK: llvm.icmp "sgt" {{.*}} : vector<4xi32>
  // CHECK: llvm.intr.masked.load
  // CHECK: llvm.cond_br %{{.*}}, ^[[OK:.*]], ^[[FAIL:.*]]
  // CHECK: ^[[OK]]:
  // CHECK: llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>
  // CHECK: ^[[FAIL]]:
  // CHECK-NEXT: llvm.intr.trap
  // CHECK-NEXT: llvm.unreachable
  gpu.func @k(%a: memref<256xf32, 1>, %i: index) kernel {
    %p = arith.constant 0.0 : f32
    %v = vector.transfer_read %a[%i], %p : memref<256xf32, 1>, vector<4xf32>
    %e = vector.extract %v[0] : f32 from vector<4xf32>
    %c = arith.cmpf oge, %e, %p : f32
    cf.assert %c, "negative"
    vector.transfer_write %v, %a[%i] {in_bounds = [true]} : vector<4xf32>, memref<256xf32, 1>
    gpu.return
  }
}

// -----

// Conversions of small floats in integer operations (L15-E844): an f8E4M3FN
// code extended to f32 (its denormals by uitofp and a scale), f32 rounded to
// f8E5M2 and f64 to f4E2M1FN on their own bits (no double rounding), the
// codes stored as integers.
gpu.module @small_floats {
  // CHECK-LABEL: gpu.module @small_floats
  // CHECK: llvm.func @k(
  // CHECK: llvm.load {{.*}} -> i8
  // CHECK: llvm.uitofp {{.*}} : i32 to f32
  // CHECK: llvm.fmul
  // CHECK: llvm.store {{.*}} : f32, !llvm.ptr<1>
  // CHECK: llvm.trunc {{.*}} : i32 to i8
  // CHECK: llvm.store {{.*}} : i8, !llvm.ptr<1>
  // CHECK: llvm.load {{.*}} -> f64
  // CHECK-NOT: llvm.fptrunc
  // CHECK: llvm.trunc {{.*}} : i64 to i4
  // CHECK: llvm.store {{.*}} : i4, !llvm.ptr<1>
  gpu.func @k(%a: memref<64xf8E4M3FN, 1>, %b: memref<64xf32, 1>, %c: memref<64xf8E5M2, 1>, %d: memref<64xf64, 1>, %e: memref<64xf4E2M1FN, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<64xf8E4M3FN, 1>
    %x = arith.extf %v : f8E4M3FN to f32
    memref.store %x, %b[%c0] : memref<64xf32, 1>
    %t = arith.truncf %x : f32 to f8E5M2
    memref.store %t, %c[%c0] : memref<64xf8E5M2, 1>
    %w = memref.load %d[%c0] : memref<64xf64, 1>
    %u = arith.truncf %w : f64 to f4E2M1FN
    memref.store %u, %e[%c0] : memref<64xf4E2M1FN, 1>
    gpu.return
  }
}
