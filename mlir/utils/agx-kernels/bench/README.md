# MLIR kernels for AGX

`bench.py` lowers each kernel here as `agx-build-fatbin.py` does (convert-gpu-to-air with bare pointers,
mlir-translate, the AIR portable passes, opt -O3 and llc -O3 for AGX) and prints the agx_cost cycles of one SIMD-group,
the instruction count and the GPRs:

```
python3 mlir/utils/agx-kernels/bench/bench.py --bin build/bin \
    --cost AGX-Performance-Simulator/rust/target/release/agx_cost [--base OLD_MLIR_OPT] [--asm DIR] [KERNEL.mlir ...]
```

Each file holds a `gpu.module @k` with one kernel, static memref shapes (the bare-pointer convention),
`known_block_size` and, where loops or bounds depend on arguments, a `// agx-cost:` line with the launch agx_cost
traces (`--uniform I=V` sets 32-bit uniform I: a pointer takes two, an `index` two, an `f32` one, in argument order;
the `threadgroups_per_grid` uniform is named in llc's kernel header comment).

| Kernel | What it models |
| --- | --- |
| `block_reduce` | threadgroup sum: SIMD-group sums, partials in threadgroup memory, a second SIMD-group sum |
| `matmul` | naive matrix product, one output per thread, a loop over K with two loads and an fma |
| `tile2d` | 16 x 16 tile of a row-major array: index math from the block size |
| `warp_reduce` | SIMD-group sum by butterfly `gpu.shuffle xor` |
| `warp_scan` | inclusive SIMD-group scan by `gpu.shuffle up` |
| `gemm_smem_tile` | 2-D GEMM with 16 x 16 tiles of A and B in workgroup memrefs, two `gpu.barrier`s per K step, K an argument |
| `gemm_simd_matrix` | GEMM on the SIMD-group matrix unit: `air.simdgroup_matrix_load` / `multiply_accumulate` / `store`, an 8 x 8 C tile per SIMD-group |
| `transpose_smem` | 1024 x 1024 transpose through a padded 32 x 33 workgroup tile (CUDA's transposeCoalesced) |
| `softmax_rows` | row softmax, a SIMD-group per row: strided loops with `gpu.subgroup_reduce maxnumf` / `add` and `math.exp` |
| `layernorm_rows` | LayerNorm of 1024-float rows by a 256-thread threadgroup: `gpu.all_reduce` (the dialect's lowering), `math.rsqrt` |
| `silu_f16x4` | f16 bias + SiLU on `vector<4xf16>` loads and stores, computed in f32 |
| `saxpy_vec4` | grid-stride `y = a x + y` on `vector<4xf32>` loads and stores |
| `stencil5_2d` | 5-point Jacobi step on a 2-D memref with a halo |
| `histogram_atomic` | 256-bin byte histogram: `memref.atomic_rmw` into workgroup bins, then into device memory |
| `bsearch_while` | per-thread binary search as an `scf.while`: a divergent, data-dependent loop |
| `rowsum` | row sums, one thread per row: an `scf.for` over a static row |
| `rowsum_f16` | the same over f16, accumulated in f32 |
| `rowcopy_vec4` | a row copy by `vector<4xf32>`: an `scf.for` of step 4 over a static row |
| `two_index` | a row dot product reading one array at stride 2: two accesses that step differently |
| `stride_loop` | a grid-stride loop to a 32-bit bound |
| `conv1d` | a 3-tap filter of x[i], x[i + 1], x[i + 2] |

The kernels from `gemm_smem_tile` down were checked on the simulator's functional runner (`agx_run`) against NumPy.
