# ptx-tv results: llvm/test/CodeGen/NVPTX

Three passes over every test file (first plain nvptx64 `llc` RUN line of each): up to 6, then 8, then 10 further functions per file; every result re-checked with the final lifter. `alive-tv --tgt-is-asm --disable-undef-input --smt-to=5000`, 20 s per function; timeouts additionally re-run with larger budgets and the weaker `--disable-poison-input` (tagged); 28 functions re-run at 30 s per query (`smt_to`); files using cp.async/cp.async.bulk/st.bulk re-run with copy semantics, timeouts among them with the opaque copy model (`copy_model`); st.async, discard and spdecompress files re-run with their ISA semantics; functions with inline asm re-run with asm_sem; arbitrary-fp conversion tests re-run with arbfp (all functions); Alive2 errors re-run after the i1/FP rewrites, alignstack removal and bounded loop unrolling (up to 160); files with masked loads/stores re-run with their per-lane expansion; previously unlifted functions re-run after the naming, vararg, stacksave and call-site fixes; files with FP min/max re-run after modelling PTX min/max by their ISA semantics; all remaining timeouts re-run at 10 s per query, then with the weaker check, then with element-wise vector accesses, then per lane for vector results (also combined, and combined with the weaker check); flag-approximation inconclusives re-run against the source without those flags (also timeouts with such flags); remaining timeouts re-run with another solver seed. Alive2 adacbd5, Z3 4.13.4, LLVM from this branch's base; all built with clang.

Test files: 619; files with no usable nvptx64 llc RUN line or other file-level failure: 79 (no-llc-run 78, llc-failed 1).
Functions checked: 4199.

| status | functions | share |
|---|---:|---:|
| `correct` | 3421 | 81.5% |
| `timeout` | 505 | 12.0% |
| `mismatch-known` | 171 | 4.1% |
| `inconclusive` | 59 | 1.4% |
| `mismatch-param-oob` | 33 | 0.8% |
| `mismatch` | 7 | 0.2% |
| `lift-unsupported` | 2 | 0.0% |
| `src-inline-asm` | 1 | 0.0% |

55 of the correct results were obtained only with a weaker check (`--disable-poison-input`: inputs assumed not to be poison) after the full check timed out.

Of the correct results, 402 rely on probe-derived opaque instructions (probe.py): they check the surrounding code and operand plumbing, not the instruction selection of the intrinsic itself.

8 of these used the opaque copy model (`PTXTV_OPAQUE_COPIES=1`) after the check with cp.async/st.bulk copy semantics timed out.

81 of the correct results were obtained with vector accesses split into element accesses on both sides (`scalarized`, PTXTV_SCALARIZE=1) after a timeout.

60 of the correct results were obtained by checking a vector result one lane at a time (`lanes`, PTXTV_LANES=1) after a timeout.

10 of the correct results were proved against the source without the fast-math flags Alive2 approximates (`stripped_fmf`; a stricter source).

1 of the correct results involve loops and were checked with bounded unrolling (`unroll` field).

Conclusive (proved correct or counterexample): 3632 (86.5%).

## Counterexamples with out-of-bounds `.param` accesses

All are finding 1 (sub-byte vectors) except `param-vectorize-device.ll` `callee_St4x*`, the lower-severity over-read into alignment padding.

* `arbitrary-fp-to-float.ll` `fp4_to_f32_vec`: ld fp4_to_f32_vec_param_0+2 (2 bytes) > 2
* `cache-hint-transforms.ll` `legalize_scalarize_load_v3i1`: st func_retval0+2 (1 bytes) > 1; st func_retval0+1 (1 bytes) > 1
* `cache-hint-transforms.ll` `legalize_scalarize_store_v3i1`: ld legalize_scalarize_store_v3i1_param_1+1 (1 bytes) > 1; ld legalize_scalarize_store_v3i1_param_1+2 (1 bytes) > 1
* `f16x2-instructions.ll` `test_fcmp_ogt`: st func_retval0+1 (1 bytes) > 1
* `f16x2-instructions.ll` `test_fcmp_uge`: st func_retval0+1 (1 bytes) > 1
* `f16x2-instructions.ll` `test_fcmp_ult`: st func_retval0+1 (1 bytes) > 1
* `f32x2-instructions.ll` `test_fcmp_ole`: st func_retval0+1 (1 bytes) > 1
* `f32x2-instructions.ll` `test_fcmp_olt`: st func_retval0+1 (1 bytes) > 1
* `f32x2-instructions.ll` `test_fcmp_ueq`: st func_retval0+1 (1 bytes) > 1
* `f32x2-instructions.ll` `test_fcmp_uge`: st func_retval0+1 (1 bytes) > 1
* `f32x2-instructions.ll` `test_fcmp_une`: st func_retval0+1 (1 bytes) > 1
* `f32x2-instructions.ll` `test_fcmp_uno`: st func_retval0+1 (1 bytes) > 1
* `float-to-arbitrary-fp.ll` `to_f4e2m1fn_v4f32`: st func_retval0+2 (2 bytes) > 2
* `i32x2-instructions.ll` `test_trunc_2xi32_to_2xi1`: st func_retval0+1 (1 bytes) > 1
* `insertelt-dynamic.ll` `dynamic_i1`: st func_retval0+4 (1 bytes) > 1; st func_retval0+7 (1 bytes) > 1
* `insertelt-dynamic.ll` `dynamic_i2`: st func_retval0+8 (2 bytes) > 2; st func_retval0+14 (2 bytes) > 2
* `insertelt-dynamic.ll` `dynamic_i3`: st func_retval0+12 (4 bytes) > 4; st func_retval0+8 (4 bytes) > 4
* `insertelt-dynamic.ll` `dynamic_i4`: st func_retval0+12 (4 bytes) > 4; st func_retval0+8 (4 bytes) > 4
* `insertelt-dynamic.ll` `dynamic_i5`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `dynamic_i6`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `dynamic_i7`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `multiple_dynamic_i3`: st func_retval0+2 (2 bytes) > 2
* `masked-store-variable-mask.ll` `global_variable_mask`: ld global_variable_mask_param_2+3 (1 bytes) > 1; ld global_variable_mask_param_2+2 (1 bytes) > 1
* `param-load-store.ll` `test_v3i1`: ld test_v3i1_param_0+1 (1 bytes) > 1; ld test_v3i1_param_0+2 (1 bytes) > 1
* `param-vectorize-device.ll` `callee_St4x3`: ld callee_St4x3_param_0+0 (16 bytes) > 12
* `param-vectorize-device.ll` `callee_St4x7`: ld callee_St4x7_param_0+16 (16 bytes) > 28
* `vector-returns.ll` `bit15`: st func_retval0+14 (1 bytes) > 2; st func_retval0+12 (2 bytes) > 2
* `vector-returns.ll` `bit16`: st func_retval0+14 (2 bytes) > 2; st func_retval0+12 (2 bytes) > 2
* `vector-returns.ll` `bit2`: st func_retval0+1 (1 bytes) > 1
* `vector-returns.ll` `bit3`: st func_retval0+2 (1 bytes) > 1; st func_retval0+1 (1 bytes) > 1
* `vector-returns.ll` `bit4`: st func_retval0+3 (1 bytes) > 1; st func_retval0+2 (1 bytes) > 1
* `vector-returns.ll` `bit8`: st func_retval0+7 (1 bytes) > 1; st func_retval0+6 (1 bytes) > 1
* `vector-returns.ll` `bit9`: st func_retval0+8 (1 bytes) > 2; st func_retval0+6 (2 bytes) > 2

## Unclassified counterexamples

Triage (see `README.md`):

* `bf16-instructions.ll` `test_uitofp_i32`, `test_uitofp_i64`: finding 3.
* `f16-abs.ll` `test_fabs`: finding 2.
* `math-intrins.ll` `abs_float_ftz`: finding 2b.
* `math-intrins.ll` `minimum_float_ftz`: finding 4.
* `atomicrmw-elementwise-sm60.ll` `fmin_acq_rel_v2bf16_global_cta`: NaN bits
  only (bf16 `minnum` done in f32; allowed for `minnum`); the NaN-bits note.
* `vaargs.ll` `foo`: not a defect; Alive2's assembly mode resolves the
  varargs pointer, stored to local memory as an integer and reloaded, to a
  different block than the source pointer (the PTX does the source's
  arithmetic exactly).

Finding 5 (`lower-args-gridconstant.ll` kernels storing to an undeclared
`func_retval0`) is reported as `lift-unsupported`, since the PTX is invalid.
The `nan-payload` entries of `arbitrary-fp-to-float.ll` are finding 6.

## Counterexamples of known modelling classes (not defects)

| class | functions |
|---|---:|
| `partword-atomic-word-access` | 87 |
| `nan-payload` | 29 |
| `approximation-permitted-by-flags` | 24 |
| `masked-load-widened` | 10 |
| `indirect-call-asm-mode` | 6 |
| `approx-substitution` | 4 |
| `aligned-overread-widened-load` | 3 |
| `byval-copy-to-local` | 3 |
| `cas-based-atomic-load` | 3 |
| `dynamic-alloca-size-overflow` | 2 |

## Not lifted (top reasons)

| reason | functions |
|---|---:|
| `kernel` | 2 |

