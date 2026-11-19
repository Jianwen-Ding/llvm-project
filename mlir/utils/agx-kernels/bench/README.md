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
the `threadgroups_per_grid` uniform is named in llc's kernel header comment; `--map VA=@FILE` maps a file, relative to
the kernel's, as memory at VA).

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
| `hist` | a 256-bin histogram of low bytes by device atomics |
| `dot_vec4` | row dot products by `vector<4xf32>` fma and a `vector.reduction` of the lanes |
| `shfl_sum` | SIMD-group sums by an xor shuffle tree, one store per group |
| `hash` | a 32-bit hash mix: multiplies, rotates of products, an xor shift |
| `h2` | `vector<2xf16>` accesses split into halves (2-byte alignment) and f16 arithmetic |
| `conv3x3` | a 3 x 3 convolution, its loops fully unrolled (known grid, no loops left) |
| `cluster_reduce` | `gpu.subgroup_reduce` over clusters of 8 and 4 lanes |
| `reduce_f16_i64` | `gpu.subgroup_reduce` of f16 (whole SIMD-group and clusters of 8) and of i64 |
| `relu6_vec4` | ReLU6 of `vector<4xf32>` by `arith.maximumf` / `arith.minimumf` |
| `box3_u8f` | a 3-tap box filter of bytes to floats (an exact float sum, then x 1/3) |
| `fir_taps` | an FIR filter with a runtime tap count: x[i + k] and w[k] in a loop over k |
| `spmv_csr` | CSR sparse y = A x, a thread per row: an `scf.for` between i32 row offsets from memory (`--map VA=@FILE`) and a gather of x |
| `causal_rowmax` | a causally masked row (j <= i) by a SIMD-group: running maximum, then the sum of exponentials, the mask on the 64-bit index |
| `window_rowmax` | causal_rowmax with a sliding window of 128 (j in [row - 128, row]): the maximum and the sum of e^(x - max) with the two-sided mask |
| `conv3x3_pad` | a zero-padded 3 x 3 convolution of a 1024 x 1024 image: nine bounds-guarded loads in flight across their ifs |
| `argmax_rows` | a thread's argmax over a row of 256 floats: the narrowed load index and the i32 index are one counter |
| `xent_rows` | cross-entropy per row of 1000 logits by a SIMD-group: maximumf, the sum of exponentials, lane 0's log and label |
| `colsum` | a thread per column of a 512 x 1024 matrix: a strided loop of a constant trip, unrolled by 8 |
| `rowmax_vec4` | a thread's maximumf over a row of 256 floats in vector<4xf32> steps: NaN and zero signs deferred to the loop's exit |
| `softmax_f16` | row softmax of f16 logits in f32 by a SIMD-group: extended loads, maximumf, the sum of exponentials and the f16 normalized write |
| `online_softmax` | one-pass softmax normalizer (log-sum-exp) of 1024-float rows by a SIMD-group: a running maximumf whose old and new values rescale the running sum, merged across lanes |
| `online_softmax_vec4` | the same by a thread per row in vector<4xf32> steps: four running maximumf / sums combined by vector reductions |
| `online_softmax_maxnum` | online_softmax with `arith.maxnumf` (C's fmaxf) for the running maximum and its subgroup reduction |
| `attn_decode_f16` | one query per head attending over 512 cached f16 keys and values (128 dims, four per lane): vector<4xf16> loads, the score by gpu.subgroup_reduce add, an online maximumf softmax rescaling the weighted sum of values |
| `rope_f32` | rotary position embedding (1024 tokens x 64 pairs): a thread per pair, the angle position * inv_freq through precise math.cos and math.sin |
| `rmsnorm_f16` | RMSNorm of f16 rows by a SIMD-group: vector<4xf16> loads, the sum of squares in f32, rsqrt, the f16 weighted write |

The kernels from `gemm_smem_tile` down were checked on the simulator's functional runner (`agx_run`) against NumPy.
