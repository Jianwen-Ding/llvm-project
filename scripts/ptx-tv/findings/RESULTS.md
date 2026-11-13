# ptx-tv results: llvm/test/CodeGen/NVPTX

Final pass over every test file: first plain nvptx64 `llc` RUN line of each file, up to 6 functions per file (deterministic sample), `alive-tv --tgt-is-asm --disable-undef-input --smt-to=5000`, 20 s wall clock per function. LLVM built from this branch's base commit; Alive2 adacbd5 with Z3 4.13.4, all built with clang.

Test files: 620; files with no usable nvptx64 llc RUN line or other file-level failure: 79 (no-llc-run 78, llc-failed 1).
Functions checked: 2111.

| status | functions | share |
|---|---:|---:|
| `correct` | 949 | 45.0% |
| `lift-unsupported` | 648 | 30.7% |
| `timeout` | 351 | 16.6% |
| `inconclusive` | 56 | 2.7% |
| `mismatch-known` | 40 | 1.9% |
| `src-inline-asm` | 31 | 1.5% |
| `alive-unsupported` | 13 | 0.6% |
| `alive-error` | 11 | 0.5% |
| `mismatch-param-oob` | 6 | 0.3% |
| `extract-failed` | 3 | 0.1% |
| `mismatch` | 3 | 0.1% |

Conclusive (proved correct or counterexample): 998 (47.3%).

## Counterexamples with out-of-bounds `.param` accesses

All are finding 1 (sub-byte vectors) except `param-vectorize-device.ll`
`callee_St4x7`, the lower-severity over-read into alignment padding.

* `f32x2-instructions.ll` `test_fcmp_uge`: st func_retval0+1 (1 bytes) > 1
* `insertelt-dynamic.ll` `dynamic_i7`: st func_retval0+8 (8 bytes) > 8
* `insertelt-dynamic.ll` `multiple_dynamic_i3`: st func_retval0+2 (2 bytes) > 2
* `param-vectorize-device.ll` `callee_St4x7`: ld callee_St4x7_param_0+16 (16 bytes) > 28
* `vector-returns.ll` `bit16`: st func_retval0+14 (2 bytes) > 2; st func_retval0+12 (2 bytes) > 2
* `vector-returns.ll` `bit9`: st func_retval0+8 (1 bytes) > 2; st func_retval0+6 (2 bytes) > 2

## Unclassified counterexamples

Triage: `bf16-instructions.ll` `test_uitofp_i64` is finding 3 (double
rounding); `f16-abs.ll` `test_fabs` is finding 2 (fabs via f32);
`math-intrins.ll` `maximumnum_float` is unresolved: `llvm.maximumnum` (which
orders -0 < +0) is lowered to `max.f32`, which the lifter models like
`maxnum` (either zero may be returned), so whether this is a defect depends
on how PTX `max` orders signed zeros.

* `bf16-instructions.ll` `test_uitofp_i64`
* `f16-abs.ll` `test_fabs`
* `math-intrins.ll` `maximumnum_float`

## Counterexamples of known modelling classes (not defects)

| class | functions |
|---|---:|
| `partword-atomic-word-access` | 19 |
| `approximation-permitted-by-flags` | 7 |
| `approx-substitution` | 4 |
| `indirect-call-asm-mode` | 3 |
| `dynamic-alloca-size-overflow` | 2 |
| `aligned-overread-widened-load` | 2 |
| `byval-copy-to-local` | 2 |
| `cas-based-atomic-load` | 1 |

## Not lifted (top reasons)

| reason | functions |
|---|---:|
| `cp.async` | 133 |
| `tcgen05.mma` | 92 |
| `spdecompress.b8` | 34 |
| `modifier .pzo` | 32 |
| `modifier .scaled::n2::ue8m0` | 23 |
| `spdecompress.b16` | 20 |
| `cp.reduce` | 13 |
| `applypriority.async` | 12 |
| `tcgen05.commit` | 12 |
| `tcgen05.ld` | 12 |
| `indirect` | 11 |
| `modifier .scaled::n1::ue8m0` | 11 |
| `mbarrier.arrive` | 11 |
| `modifier .proxy` | 9 |
| `modifier .async` | 9 |
| `tensormap.replace` | 9 |
| `modifier .tensormap` | 8 |
| `tcgen05.alloc` | 7 |
| `call` | 6 |
| `modifier .e2m3x2` | 6 |
| `modifier .rs` | 6 |
| `modifier .e4m3x2` | 6 |
| `modifier .oob` | 6 |
| `mbarrier.try_wait` | 6 |
| `redux.sync` | 6 |

## Alive2 could not handle the source

* 12: ERROR: Unsupported type: metadata
* 5: ERROR: The target program doesn't reach a return instruction.
* 3: ERROR: The source program doesn't reach a return instruction.
* 3: ERROR: SMT Error: invalid extract application
* 1: ERROR: Unsupported attribute: alignstack(32)

