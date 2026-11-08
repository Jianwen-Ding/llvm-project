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

`mlir/utils/agx-kernels/bench/bench.py` costs the kernels of that directory on AGX with the AGX Performance
Simulator (`agx_cost`, one SIMD-group's cycles), against another `mlir-opt` with `--base`. Against this
conversion before these changes (the last two rows also gain from the AGX backend's 64-bit widening as a
zero high half and its single fence around a threadgroup barrier, `-agx-merge-barrier-fences`):

| Kernel | Before | Now |
| --- | --- | --- |
| `warp_reduce` (butterfly `gpu.shuffle xor`) | 310.5 cycles, 120 instructions | 147.8 cycles, 21 instructions |
| `warp_scan` (`gpu.shuffle up`) | 328.1 cycles, 134 instructions | 154.2 cycles, 21 instructions |
| `block_reduce` (subgroup sums, threadgroup memory) | 532.1 cycles, 45 instructions | 396.8 cycles, 30 instructions |
| `tile2d` (2-D indices, `known_block_size`) | 157.4 cycles, 26 instructions | 145.9 cycles, 24 instructions |

`gpu.shuffle` by a constant offset within the SIMD-group is AIR's own `shuffle_xor` (within any power-of-two width),
`shuffle_up` or `shuffle_down` (over the whole SIMD-group), whose source lanes are those and whose valid flag is
known (xor) or one comparison of the lane; the other shuffles compute the source lane and use the index shuffle.
`gpu.subgroup_reduce` over clusters of 4 lanes is AIR's quad-group reduction (`air.quad_sum` ...), over 32 the
SIMD-group's.

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
objects into one fat AGX object (`ZAGXFAT1`), which ZLUDA's `cuModuleLoadData` picks the device's entry from.

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
| `agx.set_fpmode`, `agx.get_fpmode` | none in MLIR; ZLUDA's PTX path keeps the `llvm.agx` intrinsics |
| `#agx.launch_hints` | to be kernel attributes the AGX backend reads and AIR ignores |

The capability table that decided the native lowerings (`AGXChipInfo`) moves into the AGX backend's subtarget.
