# ptx-tv results: llvm/test/CodeGen/NVPTX

Three passes over every test file (first plain nvptx64 `llc` RUN line of each): up to 6, then 8, then 10 further functions per file; every result re-checked with the final lifter. `alive-tv --tgt-is-asm --disable-undef-input --smt-to=5000`, 20 s per function; timeouts additionally re-run with larger budgets and the weaker `--disable-poison-input` (tagged); 28 functions re-run at 30 s per query (`smt_to`). Alive2 adacbd5, Z3 4.13.4, LLVM from this branch's base; all built with clang.

Test files: 619; files with no usable nvptx64 llc RUN line or other file-level failure: 79 (no-llc-run 78, llc-failed 1).
Functions checked: 4167.

| status | functions | share |
|---|---:|---:|
| `correct` | 3123 | 74.9% |
| `timeout` | 640 | 15.4% |
| `mismatch-known` | 124 | 3.0% |
| `inconclusive` | 100 | 2.4% |
| `src-inline-asm` | 74 | 1.8% |
| `alive-unsupported` | 37 | 0.9% |
| `mismatch-param-oob` | 30 | 0.7% |
| `alive-error` | 17 | 0.4% |
| `mismatch` | 11 | 0.3% |
| `lift-unsupported` | 11 | 0.3% |

20 of the correct results were obtained only with a weaker check (`--disable-poison-input`: inputs assumed not to be poison) after the full check timed out.

Of the correct results, 668 rely on probe-derived opaque instructions (probe.py): they check the surrounding code and operand plumbing, not the instruction selection of the intrinsic itself.

Conclusive (proved correct or counterexample): 3288 (78.9%).

## Counterexamples with out-of-bounds `.param` accesses

All are finding 1 (sub-byte vectors) except `param-vectorize-device.ll` `callee_St4x*`, the lower-severity over-read into alignment padding.

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
* `i32x2-instructions.ll` `test_trunc_2xi32_to_2xi1`: st func_retval0+1 (1 bytes) > 1
* `insertelt-dynamic.ll` `dynamic_i1`: st func_retval0+4 (1 bytes) > 1; st func_retval0+7 (1 bytes) > 1
* `insertelt-dynamic.ll` `dynamic_i2`: st func_retval0+8 (2 bytes) > 2; st func_retval0+14 (2 bytes) > 2
* `insertelt-dynamic.ll` `dynamic_i3`: st func_retval0+12 (4 bytes) > 4; st func_retval0+8 (4 bytes) > 4
* `insertelt-dynamic.ll` `dynamic_i4`: st func_retval0+12 (4 bytes) > 4; st func_retval0+8 (4 bytes) > 4
* `insertelt-dynamic.ll` `dynamic_i5`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `dynamic_i6`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `dynamic_i7`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `multiple_dynamic_i3`: st func_retval0+2 (2 bytes) > 2
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

Triage (see `README.md`): `bf16-instructions.ll` `test_uitofp_i32`,
`test_uitofp_i64`: finding 3; `f16-abs.ll` `test_fabs`: finding 2;
`math-intrins.ll` `abs_float_ftz`: finding 2b; `math-intrins.ll`
`minimum_float_ftz`: finding 4; `math-intrins.ll`
`minimumnum_float`/`maximumnum_float`/`maximumnum_double` and `fmax3.ll`
`test_fminimumnum3`/`test_fmaximumnum3`: unresolved (PTX min/max signed
zeros); `bf16-instructions.ll` `test_maxnum_v2`: NaN bits only (allowed for
`maxnum`). Finding 5 (`lower-args-gridconstant.ll` kernels storing to an
undeclared `func_retval0`) is reported as `lift-unsupported`, since the PTX
is invalid.

* `bf16-instructions.ll` `test_maxnum_v2`
* `bf16-instructions.ll` `test_uitofp_i32`
* `bf16-instructions.ll` `test_uitofp_i64`
* `f16-abs.ll` `test_fabs`
* `fmax3.ll` `test_fmaximumnum3`
* `fmax3.ll` `test_fminimumnum3`
* `math-intrins.ll` `abs_float_ftz`
* `math-intrins.ll` `maximumnum_double`
* `math-intrins.ll` `maximumnum_float`
* `math-intrins.ll` `minimum_float_ftz`
* `math-intrins.ll` `minimumnum_float`

## Counterexamples of known modelling classes (not defects)

| class | functions |
|---|---:|
| `partword-atomic-word-access` | 80 |
| `approximation-permitted-by-flags` | 21 |
| `indirect-call-asm-mode` | 4 |
| `approx-substitution` | 4 |
| `nan-payload` | 4 |
| `aligned-overread-widened-load` | 3 |
| `byval-copy-to-local` | 3 |
| `cas-based-atomic-load` | 3 |
| `dynamic-alloca-size-overflow` | 2 |

## Not lifted (top reasons)

| reason | functions |
|---|---:|
| `unknown` | 3 |
| `kernel` | 2 |
| `indirect` | 2 |
| `vararg` | 2 |
| `stacksave.u64` | 1 |
| `stackrestore.u64` | 1 |

## Alive2 could not handle the source

* 36: ERROR: Unsupported type: metadata
* 8: ERROR: The target program doesn't reach a return instruction.
* 6: ERROR: SMT Error: invalid extract application
* 3: ERROR: The source program doesn't reach a return instruction.
* 1: ERROR: Unsupported attribute: alignstack(32)

