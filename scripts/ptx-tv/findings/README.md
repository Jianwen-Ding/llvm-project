# ptx-tv findings

Defects in LLVM's NVPTX backend found by ptx-tv. This directory only
documents them; no backend code has been changed. The "fix direction"
notes are suggestions for whoever fixes them.

Defects found by ptx-tv (lifting llc's PTX back to
LLVM IR and checking refinement with Alive2), each with a reproducer in this
directory. Reproducers are ordinary lit-style tests: the RUN line's llc
output passes the CHECK lines, i.e. they pin down the current (incorrect)
code generation.

Nothing was run on a GPU or through ptxas: whether a defect produces wrong
values on hardware depends on undocumented details (parameter layout in
ptxas, NaN handling of conversions). The findings are violations of LLVM IR
semantics (LangRef) or of the PTX ISA's documented contract.

## 1. Sub-byte vectors read and written outside their `.param` (out of bounds)

Reproducer: `subbyte-vector-params.ll`

A vector whose elements are narrower than a byte (`<N x i1>`, `<N x i2>`,
`<N x i4>`, ...) is declared as a `.param` of its LLVM store size (bits
packed), but the code then accesses it one or more bytes per element:

```
.visible .func (.param .align 1 .b8 func_retval0[1]) ret_v4i1(...)
	st.param.b8 	[func_retval0+3], %rs4;      // byte 3 of a 1-byte return value

.param .align 2 .b8 pass_v8i2_param_0[2]
	ld.param.b16 	%rs8, [pass_v8i2_param_0+14]; // bytes 14-15 of a 2-byte param
```

The caller writes, and the callee reads, bytes past the declared parameter,
so the two sides only agree by accident of ptxas's parameter layout. The
existing tests encode this output (for example `param-load-store.ll`
`test_v4i1`, `f16x2-instructions.ll` `test_fcmp_*` returning `<2 x i1>`,
`vector-returns.ll` `bit2`/`bit17`, `insertelt-dynamic.ll` `dynamic_i1..i7`,
`masked-store-variable-mask.ll`).

`param_oob_scan.py` finds every such access statically; over
`llvm/test/CodeGen/NVPTX` (first nvptx64 RUN line of each test) it reports
61 functions; 56 of them are this defect (see "Static scan" below).

Fix direction: make declaration and accesses agree, either by declaring the
parameter with one (or more) bytes per element, or by packing the elements
into the declared bytes.

## 2. `llvm.fabs` on f16/bf16 does not preserve NaN bits on older targets

Reproducer: `fabs-nan-bits.ll`

LangRef: "The returned value is completely identical to the input except
for the sign bit; in particular, if the input is a NaN, then the
quiet/signaling bit and payload are perfectly preserved."

Without native `abs.f16` (PTX < 6.5) the backend computes `|x|` as
`cvt.f32.f16; abs.f32; cvt.rn.f16.f32`; without native `abs.bf16` (before
sm_80/PTX 7.0) it does the operation in f32 and rounds back to bf16 with a
sequence that explicitly sets the quiet bit for NaN results
(`or.b32 %r, %r, 4194304`). Signaling NaNs therefore come back quieted.
`fneg` and `copysign` on the same targets are lowered bitwise and are
correct, and so is `fabs` on `<2 x half>` (`and.b32 ..., 0x7fff7fff`); a
scalar `and.b16 ..., 0x7fff` would be correct on every target.

Severity: low (only signaling-NaN inputs are affected).

### 2b. `fabs`/`fneg` flush subnormals under `preservesign`

Reproducer: `fabs-fneg-ftz.ll`

The same LangRef rule ("`fneg`, `llvm.fabs`, and `llvm.copysign` ... act
directly on the underlying bit representation and never change anything
except possibly for the sign bit") is violated for f32 when the function has
`denormal_fpenv(float: preservesign)`: the backend selects `abs.ftz.f32` and
`neg.ftz.f32`, which flush a subnormal input, so `fabs(0x00000001)` is `+0.0`
and `fneg(0x00000001)` is `-0.0`. `copysign` is lowered to `copysign.f32`
(no `.ftz`) and is correct. (Test: `math-intrins.ll` `abs_float_ftz`.)
LangRef's `denormal_fpenv` text speaks of "floating-point operations"
flushing inputs, so one could argue about the intent, but the NaN section
explicitly excludes these three operations from floating-point math.

## 3. Integer to bf16 conversion rounds twice before sm_90

Reproducer: `int-to-bf16-double-rounding.ll`

Without `cvt.rn.bf16.{s,u}{32,64}` (sm_90, PTX 7.8), `sitofp`/`uitofp` of
i32/i64 to `bfloat` is lowered as integer -> f32 (round to nearest) -> bf16
(round to nearest). Rounding twice is not the same as rounding once:

| input (uitofp i64) | correct bf16 | llc (sm_80) |
|---|---|---|
| `0x0000000020200007` (538968071) | `0x4e01` | `0x4e00` |
| `0xC37FFF8C80000000` | `0x5f43` | `0x5f44` |

The first rounding (to 24 bits) lands exactly on a bf16 halfway point, and
the second rounding then goes to even. All four variants (s32, u32, s64, u64)
fail on sm_70 and sm_80; all are correct on sm_90. `fptrunc double` to
`bfloat` on the same targets is done with round-to-odd and is correct, so the
same technique (or a sticky bit) fixes the integer paths.

Severity: medium (silently wrong results, one ulp, for integers with more
than 24 significant bits).

## 4. `llvm.minimum`/`llvm.maximum` return an unflushed subnormal under `preservesign`

Reproducer: `minimum-maximum-ftz.ll` (found in `math-intrins.ll`
`minimum_float_ftz`)

With `denormal_fpenv(float: preservesign)` inputs must be treated as zero.
The lowering of `llvm.minimum.f32`/`llvm.maximum.f32` orders signed zeros by
checking `setp.eq.ftz.f32 p, a, b` and, if equal, selecting one of the
*original* operands:

```
setp.eq.b32      %p1, %r1, -2147483648;   // a == -0.0 ?
selp.f32         %r3, %r1, %r2, %p1;      // a or b, unflushed
setp.nan.ftz.f32 %p2, %r1, %r2;
min.ftz.f32      %r4, %r1, %r2;
selp.f32         %r5, 0f7FC00000, %r4, %p2;
setp.eq.ftz.f32  %p3, %r1, %r2;           // a subnormal equals 0 here
selp.f32         %r6, %r3, %r5, %p3;      // ... and is returned as is
```

`minimum(+0.0, 0x00000001)` returns `0x00000001` (1e-45) and
`maximum(+0.0, 0x00000008)` returns `0x00000008`; with flushed inputs the
result must be a zero. The default (`ieee`) lowering is correct.

Severity: low (subnormal inputs, flush-to-zero functions only).

Fix direction: select from flushed operands (or from the `min.ftz` result
with its sign fixed) in the equal case.

## 5. Kernels with a non-void return type store to an undeclared `func_retval0`

Reproducer: `kernel-nonvoid-return.ll` (found in `lower-args-gridconstant.ll`
`grid_const_ptrtoint`, `grid_const_partial_escapemem`; `entry_retval_scan.py`
finds no others in the suite)

A `ptx_kernel` function whose LLVM return type is not `void` becomes a
`.entry` with no return parameter, but its `ret` is still lowered to
`st.param.b32 [func_retval0], ...`. `func_retval0` is not declared in a
`.entry`, so the PTX names an undefined symbol (invalid PTX). The tests'
CHECK lines encode this output.

Severity: low (kernels returning values are unusual; the value is not
observable). Fix direction: drop the store for kernels (or reject non-void
kernels in the verifier).

## 6. `llvm.convert.from.arbitrary.fp` quiets signaling NaNs

Reproducer: `arbitrary-fp-snan.ll` (found in `arbitrary-fp-to-float.ll`
`from_f8e5m2_nan` and the `*_dynamic`/vector variants)

LangRef for this intrinsic: "the NaN representation is preserved (quiet
NaNs remain quiet, signaling NaNs remain signaling)". `Float8E5M2` `0x7D`
(`S.11111.01`) is a signaling NaN, but the result is the quiet f32 NaN
`0x7FC00000`, both when the call is constant folded and in the expanded
code for a dynamic input, which maps every NaN input to that constant (the
NaN's sign is dropped as well; LangRef does not say whether the sign must be
kept). The same holds for `half`/`bfloat` results. The lowering is
target-independent; it is observed here through NVPTX.

Severity: low (signaling-NaN inputs only); either the lowering or the
LangRef text should change.

## 7. NVPTXUsage documents wrong signatures for `tcgen05.st.16x32bx2` and `tcgen05.ld.red`

Reproducer: `tcgen05-st-doc-order.ll` (found while deriving the
`tcgen05.st` model from the documentation)

`llvm/docs/NVPTXUsage.md` gives

```
declare void @llvm.nvvm.tcgen05.st.16x32bx2.<num>(ptr addrspace(6) %tmem_addr,
                                                  <n x i32> %args, i64 %offset, i1 %unpack)
```

but `IntrinsicsNVVM.td` (`NVVM_TCGEN05_ST`) defines the immediate offset
before the data, `(ptr addrspace(6), i64, <n x i32>, i1)`, and the IR
verifier rejects the documented form ("intrinsic argument 1 type expected
i64"). The code generator and the tests use the defined order; only the
documentation is wrong. (The `tcgen05.ld.16x32bx2` entry, whose offset
precedes the flag, is consistent.)

The `tcgen05.ld.red` entry is wrong too (reproducer
`tcgen05-ld-red-doc-type.ll`): it documents a `<n x i32>` result for both
the `.i32` and the `.f32` forms, but `NVVM_TCGEN05_LD_RED` returns
`{<n x T>, T}` (the loaded values and the reduction result, `T` = `float`
for `.f32`), which is what the backend and its tests use.

A smaller slip in the same document: the `tcgen05.mma.ws` syntax block lists
`@llvm.nvvm.tcgen05.mma.ws.shared.zero_col_mask` twice, the second time with
a tensor-memory A operand (`ptr addrspace(6) %atensor`), which is the
`ws.tensor.zero_col_mask` form listed on the next line.

Severity: documentation only. Fix direction: swap `%args` and `%offset` in
the documented `tcgen05.st.16x32bx2` signature and give `tcgen05.ld.red`
its two-element struct result.

## Lower-severity observations

* `param-vectorize-device.ll` `callee_St4x3` / `callee_St4x7`: a 12-byte
  `byval` struct declared `.param .align 16 .b8 [12]` is read with
  `ld.param.v4.b32`, 4 bytes past the declared size (inside the alignment
  padding; the extra lane is discarded). The test's CHECK lines expect this.
* `bf16-instructions.ll` `test_maxnum_v2` (no native bf16 max), and the same
  in `atomicrmw-elementwise-sm60.ll` `fmin_acq_rel_v2bf16_global_cta` (found
  with the larger solver budget): `minnum`/`maxnum` of `<2 x bfloat>` is
  done in f32 and rounded back, so signaling-NaN inputs come back with a
  different NaN bit pattern. LangRef allows NaN results of `minnum`/`maxnum`
  to vary, so this is only a note.
* `rsqrt-opt.ll`: `1.0 / llvm.nvvm.sqrt.approx.f(x)` becomes
  `rsqrt.approx.f32` without fast-math flags (`-nvptx-rsqrt-approx-opt`,
  default on). Intentional, but it replaces an exactly rounded division.

## Static scan of `.param` accesses

`python3 param_oob_scan.py llvm/test/CodeGen/NVPTX/*.ll` reports 61 functions:

* 56: sub-byte vectors (finding 1): `<N x i1>` (N from 2 up to 17), `<8 x i2>` to
  `<8 x i7>`, `<4 x i4>` (FP4), in arguments, return values and call sites,
  across `vector-returns.ll`, `f16x2-instructions.ll`, `f32x2-instructions.ll`,
  `insertelt-dynamic.ll`, `param-load-store.ll`, `i32x2-instructions.ll`,
  `cache-hint-transforms.ll`, `masked-store-variable-mask.ll`,
  `arbitrary-fp-to-float.ll`, `float-to-arbitrary-fp.ll`;
* 2: `param-vectorize-device.ll` over-read into alignment padding;
* 3: not backend defects; the source itself reads past its `byval` object
  (the source is UB): `forward-ld-param.ll` `test_ld_param_const`,
  `lower-byval-args.ll` `memcpy_from_param{,_noalign}`.

## Resolved: `minimumnum`/`maximumnum` lowered to `min`/`max`

`llvm.minimumnum`/`llvm.maximumnum` are lowered to plain `min.f32`/`max.f32`
(`math-intrins.ll` `minimumnum_float`, `maximumnum_float`,
`maximumnum_double`; `fmax3.ll` `test_fminimumnum3`, `test_fmaximumnum3`;
`reduction-intrinsics.ll` `reduce_fmaximumnum_float`). This was open while
the lifter modelled PTX `min`/`max` like `minnum`/`maxnum`. The PTX ISA's
semantics for `min`/`max` settle it: a NaN operand yields the other
operand, two NaNs give NaN, and `-0.0 < +0.0`, which is exactly
`minimumnum`/`maximumnum`. With that model all of these verify: not a
defect.

## Test-suite run

`RESULTS.md` has the numbers (generated from `results.jsonl` by
`gen_results.py`). Three passes were combined (up to 6, 8 and 10 further
functions per test file, so files with up to 24 functions are fully
covered); every result was re-checked with the final lifter (including a
regression re-run of all previously correct functions), and timeouts were
re-run with larger budgets and, separately, the weaker
`--disable-poison-input` check (tagged). In short: 4200 functions from 619
test files; 3422 proved correct (195 through probe-derived opaque instructions,
50 only with the weaker check); 33 counterexamples with out-of-bounds
`.param` accesses (finding 1, plus the padding over-reads); findings
2/2b/3/4/5/6/7; 1 NaN-bits note; 1 Alive2 artifact (`vaargs.ll` `foo`); 171 in
known modelling classes; 2 not lifted; 564 solver timeouts or inconclusive.

Not lifted (2): the two finding-5 kernels (invalid PTX). Variadic
functions are compared after LLVM's
ExpandVariadics (which the backend runs; the pass itself is trusted). One
function still has inline asm the model does not handle
(`nvvm-reflect-arch-O0.ll` `baz`: `valid;` is not a PTX instruction).
Variadic functions are compared after LLVM's
ExpandVariadics (which the backend runs; the pass itself is trusted).

The remaining timeouts: about three quarters involve floating point; the
rest are 64-bit division and vector atomics expanded into CAS loops. A 24x
larger per-query budget resolved 7 of 81 integer ones tried; a 2x budget
with the weaker check resolved 20 of about 360. A final sweep of all 666
remaining timeouts at 10 s per query proved 9 more and classified 5; the same sweep with the
weaker check proved 15 more and classified 10. Lifting vector accesses as
element accesses on both sides (`scalarized`) then proved 46 more, and
checking vector results one lane at a time (`lanes`) 32 more; both together
at 10 s per query 17 more; all of these with the weaker check 18 more. Of the results Alive2 left
inconclusive because of approximated fast-math flags, 9 were proved
against the source without those flags (`stripped_fmf`), and 1 timeouts
with such flags likewise. Another solver seed
(`--smt-random-seed=1`) proved 5 more. Some proofs are
sensitive to machine load: a regression re-run under load turned 28
previously correct functions into timeouts; 19 of them were proved again
with a 30 s per-query budget (recorded as `smt_to`), 9 remain timeouts.

## Observations not counted as defects

* `intrinsic-old.ll` `test_gridid`: `llvm.nvvm.read.ptx.sreg.gridid` is a
  32-bit intrinsic and is lowered to `mov.u32 %r, %gridid`; the PTX ISA
  documents `%gridid` as 64-bit (since PTX 3.0). Not checkable without the
  vendor toolchain; noted only.
