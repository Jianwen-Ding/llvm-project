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

## Lower-severity observations

* `param-vectorize-device.ll` `callee_St4x3` / `callee_St4x7`: a 12-byte
  `byval` struct declared `.param .align 16 .b8 [12]` is read with
  `ld.param.v4.b32`, 4 bytes past the declared size (inside the alignment
  padding; the extra lane is discarded). The test's CHECK lines expect this.
* `bf16-instructions.ll` `test_maxnum_v2` (no native bf16 max): `maxnum` of
  `<2 x bfloat>` is done in f32 and rounded back, so signaling-NaN inputs
  come back with a different NaN bit pattern. LangRef allows NaN results of
  `maxnum` to vary, so this is only a note.
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

## Unresolved

* `llvm.minimumnum`/`llvm.maximumnum` lowered to plain `min.f32`/`max.f32`
  (`math-intrins.ll` `minimumnum_float`, `maximumnum_float`; `fmax3.ll`
  `test_fminimumnum3`, `test_fmaximumnum3`, the three-input forms). The
  counterexamples are `min(+0.0, -0.0)` (must be -0.0) and signaling-NaN
  inputs (must return the other operand). Correct only if PTX `min`/`max`
  order -0.0 below +0.0 and treat signaling NaNs as missing; the lifter does
  not assume either (it models them like `maxnum`), so this stays open.

## Test-suite run

`RESULTS.md` has the numbers (generated from `results.jsonl` by
`gen_results.py`). Two passes were combined: up to 6 functions per test file,
then up to 8 more per file not checked before, all re-checked with the final
lifter. In short: 3323 functions from 620 test files; 1727 proved correct;
15 counterexamples of finding 1 (plus 1 padding over-read); 4 of findings
2/2b/3/4; 4 unresolved (min/max signed zeros) and 1 NaN-bits note; 81 in
known modelling classes; 736 not lifted (mostly asynchronous/tensor-core
instructions); 648 solver timeouts or inconclusive.
