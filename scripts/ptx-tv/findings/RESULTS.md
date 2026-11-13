# ptx-tv results: llvm/test/CodeGen/NVPTX

Two passes over every test file (first plain nvptx64 `llc` RUN line of each): up to 6 functions per file, then up to 8 further functions per file; non-correct results re-checked with the final lifter. `alive-tv --tgt-is-asm --disable-undef-input --smt-to=5000`, 20 s per function; functions that timed out were re-run with `--smt-to=10000` and the weaker `--disable-poison-input` (tagged). Alive2 adacbd5, Z3 4.13.4, LLVM from this branch's base; all built with clang.

Test files: 620; files with no usable nvptx64 llc RUN line or other file-level failure: 158 (no-llc-run 156, llc-failed 2).
Functions checked: 3323.

| status | functions | share |
|---|---:|---:|
| `correct` | 2457 | 73.9% |
| `timeout` | 518 | 15.6% |
| `inconclusive` | 92 | 2.8% |
| `mismatch-known` | 84 | 2.5% |
| `src-inline-asm` | 60 | 1.8% |
| `lift-unsupported` | 36 | 1.1% |
| `alive-unsupported` | 27 | 0.8% |
| `alive-error` | 20 | 0.6% |
| `mismatch-param-oob` | 16 | 0.5% |
| `mismatch` | 10 | 0.3% |
| `extract-failed` | 3 | 0.1% |

20 of the correct results were obtained only with a weaker check (`--disable-poison-input`: inputs assumed not to be poison) after the full check timed out.

Of the correct results, 595 rely on probe-derived opaque instructions (probe.py): they check the surrounding code and operand plumbing, not the instruction selection of the intrinsic itself.

Conclusive (proved correct or counterexample): 2567 (77.2%).

## Counterexamples with out-of-bounds `.param` accesses

All are finding 1 (sub-byte vectors) except `param-vectorize-device.ll` `callee_St4x*`, the lower-severity over-read into alignment padding.

* `cache-hint-transforms.ll` `legalize_scalarize_load_v3i1`: st func_retval0+2 (1 bytes) > 1; st func_retval0+1 (1 bytes) > 1
* `cache-hint-transforms.ll` `legalize_scalarize_store_v3i1`: ld legalize_scalarize_store_v3i1_param_1+1 (1 bytes) > 1; ld legalize_scalarize_store_v3i1_param_1+2 (1 bytes) > 1
* `f16x2-instructions.ll` `test_fcmp_ogt`: st func_retval0+1 (1 bytes) > 1
* `f16x2-instructions.ll` `test_fcmp_uge`: st func_retval0+1 (1 bytes) > 1
* `f32x2-instructions.ll` `test_fcmp_uge`: st func_retval0+1 (1 bytes) > 1
* `insertelt-dynamic.ll` `dynamic_i1`: st func_retval0+4 (1 bytes) > 1; st func_retval0+7 (1 bytes) > 1
* `insertelt-dynamic.ll` `dynamic_i3`: st func_retval0+12 (4 bytes) > 4; st func_retval0+8 (4 bytes) > 4
* `insertelt-dynamic.ll` `dynamic_i4`: st func_retval0+12 (4 bytes) > 4; st func_retval0+8 (4 bytes) > 4
* `insertelt-dynamic.ll` `dynamic_i5`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `dynamic_i7`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `multiple_dynamic_i3`: st func_retval0+2 (2 bytes) > 2
* `param-load-store.ll` `test_v3i1`: ld test_v3i1_param_0+1 (1 bytes) > 1; ld test_v3i1_param_0+2 (1 bytes) > 1
* `param-vectorize-device.ll` `callee_St4x7`: ld callee_St4x7_param_0+16 (16 bytes) > 28
* `vector-returns.ll` `bit16`: st func_retval0+14 (2 bytes) > 2; st func_retval0+12 (2 bytes) > 2
* `vector-returns.ll` `bit2`: st func_retval0+1 (1 bytes) > 1
* `vector-returns.ll` `bit9`: st func_retval0+8 (1 bytes) > 2; st func_retval0+6 (2 bytes) > 2

## Unclassified counterexamples

Triage (see `README.md`): `bf16-instructions.ll` `test_uitofp_i64`:
finding 3; `f16-abs.ll` `test_fabs`: finding 2; `math-intrins.ll`
`abs_float_ftz`: finding 2b; `math-intrins.ll` `minimum_float_ftz`: finding
4; `math-intrins.ll` `minimumnum_float`/`maximumnum_float` and `fmax3.ll`
`test_fminimumnum3`/`test_fmaximumnum3`: unresolved (PTX min/max signed
zeros); `bf16-instructions.ll` `test_maxnum_v2` and
`atomicrmw-elementwise-sm60.ll` `fmin_acq_rel_v2bf16_global_cta`: NaN bits
only (allowed for `minnum`/`maxnum`).

* `atomicrmw-elementwise-sm60.ll` `fmin_acq_rel_v2bf16_global_cta`
* `bf16-instructions.ll` `test_maxnum_v2`
* `bf16-instructions.ll` `test_uitofp_i64`
* `f16-abs.ll` `test_fabs`
* `fmax3.ll` `test_fmaximumnum3`
* `fmax3.ll` `test_fminimumnum3`
* `math-intrins.ll` `abs_float_ftz`
* `math-intrins.ll` `maximumnum_float`
* `math-intrins.ll` `minimum_float_ftz`
* `math-intrins.ll` `minimumnum_float`

## Counterexamples of known modelling classes (not defects)

| class | functions |
|---|---:|
| `partword-atomic-word-access` | 49 |
| `approximation-permitted-by-flags` | 16 |
| `indirect-call-asm-mode` | 5 |
| `approx-substitution` | 4 |
| `byval-copy-to-local` | 3 |
| `dynamic-alloca-size-overflow` | 2 |
| `aligned-overread-widened-load` | 2 |
| `nan-payload` | 2 |
| `cas-based-atomic-load` | 1 |

## Not lifted (top reasons)

| reason | functions |
|---|---:|
| `call` | 8 |
| `unknown` | 6 |
| `vararg` | 5 |
| `indirect` | 3 |
| `modifier .async` | 3 |
| `directive` | 2 |
| `non` | 2 |
| `multimem.st` | 2 |
| `stacksave.u64` | 1 |
| `stackrestore.u64` | 1 |
| `suld.b` | 1 |
| `tex.1d` | 1 |
| `modifier .mmio` | 1 |

## Alive2 could not handle the source

* 26: ERROR: Unsupported type: metadata
* 8: ERROR: The target program doesn't reach a return instruction.
* 5: ERROR: SMT Error: invalid extract application
* 4: ERROR: SMT Error: max. memory exceeded
* 3: ERROR: The source program doesn't reach a return instruction.
* 1: ERROR: Unsupported attribute: alignstack(32)

