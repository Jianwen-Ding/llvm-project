# 'air' Dialect: Apple GPUs through AIR

[TOC]

The `air` dialect targets Apple GPUs through AIR, the LLVM IR form of Apple's Metal shading language that the AIR
target (`llvm/lib/Target/AIR`) writes. Kernels are written in the standard dialects (`gpu`, `scf`, `arith`, `math`,
`memref`, `vector`) plus the few `air` operations for what those cannot say. AIR abstracts over the chip: a kernel
names no chip, and the chip-specific code generation lives in the LLVM backends.

## Operations

### SIMD-group matrices

An 8x8 SIMD-group matrix is a `vector<64xT>` (T is f16, bf16 or f32) shared by the 32 threads of a SIMD-group. Its
elements are not individually addressable: only the operations below, element-wise arithmetic (`arith`, `math`),
`arith.extf` / `arith.truncf` and `arith.select` / `scf` values of matrices are allowed. Every thread of the
SIMD-group executes them with the same operands.

- `air.simdgroup_matrix_load %m[%i, %j] [transpose]`: the matrix `m[i + r, j + c]` (`m[i + c, j + r]` with
  `transpose`) of a 2-D memref with any strides.
- `air.simdgroup_matrix_store %x, %m[%i, %j] [transpose]`: the reverse.
- `air.simdgroup_matrix_multiply_accumulate %a, %b, %c`: `A * B + C`. A and B have one element type and C and the
  result another: the same, or f16 or bf16 A and B with f32 C (`: vector<64xf16>, vector<64xf32>`).
- `air.simdgroup_matrix_fill %v`: every element `v`.

`--canonicalize` folds element-wise operations of fills: an `arith` or `math` operation whose matrix operands are
all `air.simdgroup_matrix_fill`s (and whose other operands are scalars) is the fill of the operation on their values,
`arith.addf(fill(x), fill(y))` is `fill(arith.addf(x, y))`, as every element of a fill is its value.

## What the conversion knows

`convert-gpu-to-air` declares each AIR function with what its calls do, so that LLVM can fold, hoist and sink them:
the system values (`air.thread_position_in_grid` ...) touch no memory, the SIMD-group functions (shuffles, ballots,
reductions, matrices) are `convergent` as well (the matrix loads read and the stores write their pointer operand
only), and the barriers are `convergent` calls that may touch any memory.

From the launch bounds the kernel knows (`known_block_size`, `known_grid_size`, the ops' `upper_bound`):

- a threadgroup or grid size it fixes is a constant (`gpu.block_dim`, `gpu.grid_dim`), as is the SIMD-group count
  (`gpu.num_subgroups`, `ceil(threads / 32)`) and, for a threadgroup of one SIMD-group, `gpu.subgroup_id`;
- the other system values carry their `range` (`llvm.range`): a lane is in `[0, 32)`, a thread position below the
  threadgroup size, a SIMD-group index below the SIMD-group count.

`gpu.global_id` is AIR's `thread_position_in_grid`, one 32-bit read that a memory access takes as its 32-bit index,
where the dimension has fewer than 2^32 threads (the op's `upper_bound`, or `known_block_size` times
`known_grid_size`); without such a bound it is `block_id * block_dim + thread_id` in `index`, since the 32-bit
register would wrap (alive2: `tests/alive-tv/gpu/global-id-32bit*`).

The index of a `memref.load` or `memref.store` of a static dimension of at most 2^32 elements is below 2^32
(an access out of bounds is undefined), so it is its value modulo 2^32, which additions, subtractions and
multiplications commute with: the conversion computes such an index's arithmetic in `i32` and zero-extends the
result (`narrow-indices`, on; alive2: `tests/alive-tv/gpu/narrow-memref-index`). AGX's integer ALU is 32-bit,
where a 64-bit multiplication takes several instructions.

An `scf.for` over `index` whose iterations each reach a `memref.load` or `memref.store` of a static dimension of
S <= 2^31 elements indexed by `x + iv` or `x - iv` (`x` loop invariant) keeps its offsets `iv - lb` below S: the
accessed indices step by `step` within `[0, S)`, so a step of S or more allows one iteration. A constant extent
`ub - lb <= 2^31` bounds them too. The conversion runs such a loop over an unsigned `i32` offset from 0 to
`umin(ub - lb, S)` (0 for `ub <= lb`) by `umin(step, S)`, with `iv = lb + offset` (`narrow-loops`, on; alive2:
`tests/alive-tv/gpu/narrow-loop*`, at an eighth of the widths, unrolled past the most iterations): the trip count
is the original's, and the index arithmetic of `iv` that `narrow-indices` computes becomes 32-bit, such as the
grid-stride loop's addresses. LLVM's IndVarSimplify widens such a counter back to `i64` when an address
zero-extends it, since the AGX data layout makes `i64` a native integer.

| Kernel | `narrow-loops=0` | `narrow-loops` |
| --- | --- | --- |
| `grid_stride` (grid-stride loop) | 646.3 cycles, 36 instructions | 542.3 cycles, 40 instructions |
| `softmax_rows` (strided row loops) | 3460.8 cycles, 179 instructions | 3349.7 cycles, 164 instructions |
| `gemm_smem_tile` (K loop of tiles) | 5832.8 cycles, 95 instructions | 5679.8 cycles, 95 instructions |
| `matmul` (K loop) | 5617.5 cycles, 37 instructions | 5609.3 cycles, 32 instructions |

With `linearize-indices` (on), an access of a memref of two or more dimensions of static shape, non-negative
static strides and offset, whose last element is within 2^32 of its base, is one of the 32-bit index
`offset + sum(index_k * stride_k)` into the memref reinterpreted as one dimension: each index is below its size,
so the sum is at most the last element's offset (alive2: `tests/alive-tv/gpu/linearize-memref-index`). The memref
lowering's own linearization is 64-bit, which AGX computes in pairs of registers.

| Kernel | `linearize-indices=0` | `linearize-indices` |
| --- | --- | --- |
| `bias_relu_2d_f16` (2-D, f16) | 176.3 cycles, 23 instructions | 127.6 cycles, 16 instructions |
| `stencil5_2d` (2-D neighbours) | 263.4 cycles, 36 instructions | 236.8 cycles, 27 instructions |
| `transpose_smem` (2-D, threadgroup tile) | 541.1 cycles, 69 instructions | 505.7 cycles, 46 instructions |
| `gemm_smem_tile` (2-D tiles) | 5679.8 cycles, 95 instructions | 5495.7 cycles, 91 instructions |
| `softmax_rows` (2-D rows) | 3349.7 cycles, 164 instructions | 3338.3 cycles, 163 instructions |
| `layernorm_rows` (2-D rows, all-reduce) | 2439.4 cycles, 643 instructions | 2626.6 cycles, 630 instructions |

`layernorm_rows` issues its device loads right after the 32-bit address additions, which the simulator holds back
(a load after ALU work waits for the pending loads to pass the L1); the 64-bit ones happened to come earlier.

`mlir/utils/agx-kernels/bench/bench.py` costs the kernels of that directory on AGX with the AGX Performance
Simulator (`agx_cost`, one SIMD-group's cycles), against another `mlir-opt` with `--base`. Against this
conversion before these changes (the last two rows also gain from the AGX backend's 64-bit widening as a
zero high half and its single fence around a threadgroup barrier, `-agx-merge-barrier-fences`):

| Kernel | Before | Now |
| --- | --- | --- |
| `warp_reduce` (butterfly `gpu.shuffle xor`) | 310.5 cycles, 120 instructions | 147.8 cycles, 21 instructions |
| `warp_scan` (`gpu.shuffle up`) | 328.1 cycles, 134 instructions | 154.2 cycles, 21 instructions |
| `block_reduce` (subgroup sums, threadgroup memory) | 532.1 cycles, 45 instructions | 396.8 cycles, 30 instructions |
| `tile2d` (2-D indices, `known_block_size`) | 157.4 cycles, 26 instructions | 119.6 cycles, 12 instructions |

`gpu.shuffle` by a constant offset within the SIMD-group is AIR's own `shuffle_xor` (within any power-of-two width),
`shuffle_up` or `shuffle_down` (over the whole SIMD-group), whose source lanes are those and whose valid flag is
known (xor) or one comparison of the lane. Within narrower segments (up, down), at a constant lane (idx), or by an
offset that is the same in every thread (a constant, a kernel argument, a threadgroup-level id or size, a loop
counter with such bounds, or arithmetic of those) they are the same AIR shuffles with that uniform offset (up and down
by a uniform offset of either sign: the shift by it or the opposite shift by its negation), and the thread's own value
where the source lane is outside its segment; the other shuffles compute the source lane and use the index shuffle.
`gpu.rotate` is a shuffle down by the offset and up by width - offset (AIR has no rotate), selected by the lane's
position in its segment; `gpu.subgroup_broadcast` is AIR's shuffle of the lane or its broadcast of the first active
lane.
`gpu.subgroup_reduce` over clusters of 4 lanes is AIR's quad-group reduction (`air.quad_sum` ...), over 32 the
SIMD-group's. `maximumf` / `minimumf` of `f32` (a NaN if any lane's is, -0.0 below +0.0) are not AIR's float
maximum and minimum, which are `maxnum` / `minnum`: they are the signed `i32` maximum (minimum) of the keys
`b ^ ((b >>s 31) & 0x7fffffff)` of the floats' bits, decoded the same way. A NaN's key is replaced by the largest
float's, the quiet NaN's (by the smallest's, `0xffffffff`'s, whose decoded NaN then selects the quiet NaN), so that
no second reduction (an or of "is NaN") is needed (alive2: `tests/alive-tv/gpu/subgroup-maximum-nan-key`,
`subgroup-minimum-nan-key`); a value with two such reductions keeps its own key for both and selects a NaN where the
group's or of "is NaN" is set (`subgroup-maximum-key`, `subgroup-minimum-key`).

A uniform `gpu.all_reduce` of an `f16`, `f32` or `i32` by an operation AIR's SIMD-group reductions have, in a
kernel whose `known_block_size` is a multiple of 32 threads (whole SIMD-groups), is the SIMD-group reduction, each
SIMD-group's partial in threadgroup memory, one barrier, and every SIMD-group's reduction of the partials: lane l
reads partial l, and the lanes past the SIMD-group count read partial 0 (minimum, maximum, and, or) or the identity
(`-0.0` or 0 for an addition, 1 for a multiplication, 0 for xor), which leave the result as it is (alive2:
`tests/alive-tv/gpu/simdgroup-all-reduce-fill`). Where the reduction may run again (in a loop) a second barrier
keeps the next partials from overwriting ones not yet read (`simdgroup-all-reduce`, on). The generic lowering,
for any threadgroup, handles partial SIMD-groups with shuffles and broadcasts through threadgroup memory after a
second barrier: `layernorm_rows`, two reductions of 256 threads, goes from 2626.6 cycles and 630 instructions to
1568.4 cycles and 123 instructions.

alive2 (its AIR and AGX semantics, `tests/alive-tv/gpu/shuffles-native`) proves the native shuffles equal to the
lane arithmetic they replace.

## Target and pipeline

`#air.target<O, triple, chip, features>` on a `gpu.module` (`air-attach-target` attaches it):

- without a chip, `gpu-module-to-binary` produces a `.metallib` for Apple's Metal driver;
- with an AGX chip (an LLVM AGX processor such as `agx-g14g`), an AGX object: the module's AIR goes through the AIR
  target's portable passes (`air-portable`), then the AGX backend, whose `agx-lower-air` pass turns AIR into AGX IR
  at the start of its pipeline (`llvm/docs/AGXUsage.rst`, "AIR input"). `features` are AGX target features, such as
  `+agx-strict` or `+agx-checked`.

The conversion is `convert-gpu-to-air`, in each `gpu.module`. A memref without a memory space is device memory,
except a `memref.alloca`, which is the thread's (AIR has no generic address space). AIR has no double precision:
run `arith-emulate-f64` first, which computes f64 on i64 with a software library.

```mlir
mlir-opt kernel.mlir --pass-pipeline='builtin.module(air-attach-target{chip=agx-g14g},
    gpu.module(arith-emulate-f64,convert-gpu-to-air),gpu-module-to-binary)'
```

`mlir/utils/agx-kernels/agx-build-fatbin.py` compiles a library kernel source for several AGX chips and packs the
objects into one fat AGX object (`ZAGXFAT1`), which a runtime's `cuModuleLoadData` picks the device's entry from.

## Where the `agx` dialect went

The `agx` and `agxir` dialects (`convert-gpu-to-agxir`, `convert-agx-to-agxir`, `#agxir.target`) are gone. Their
chip-specific choices are the AGX backend's now, so that one AIR kernel serves every chip:

| `agx` operation | Now |
| --- | --- |
| `agx.mma`, `agx.matrix_load` / `store`, `agx.mma_tile` | `air` SIMD-group matrices (a tile is a loop of them) |
| `agx.shuffle`, `agx.reduce`, `agx.ballot` | `gpu.shuffle`, `gpu.subgroup_reduce`, `gpu.ballot` |
| `agx.threadgroup_barrier`, `agx.memory_barrier` | `gpu.barrier` (with `memfence` for the memory it orders) |
| `agx.block_barrier` | `gpu.barrier memfence []`; the backend is to use the 4-SIMD-group barrier when the threadgroup is that small |
| `agx.fma_lane_shift` | `gpu.shuffle up` in segments, then `math.fma`; the backend is to match the G14 lane-shift FMA |
| `agx.uniform_region` | inline code; the backend is to hoist uniform work into the preamble |
| `agx.core_id`, `agx.simd_slot` | none (machine specific); PTX's `%smid` keeps `llvm.agx.core.id` |
| `agx.set_fpmode`, `agx.get_fpmode` | none in MLIR; the PTX path keeps the `llvm.agx` intrinsics |
| `#agx.launch_hints` | to be kernel attributes the AGX backend reads and AIR ignores |

The capability table that decided the native lowerings (`AGXChipInfo`) moves into the AGX backend's subtarget.
