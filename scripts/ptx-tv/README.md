# ptx-tv: translation validation for the NVPTX backend

ptx-tv checks LLVM's NVPTX backend by lifting the PTX that `llc` emits back
to LLVM IR and asking [Alive2](https://github.com/AliveToolkit/alive2)
(`alive-tv --tgt-is-asm`) whether the lifted code refines the original
function.

Everything here is built from public material only: the PTX ISA as publicly
documented and the open-source LLVM NVPTX backend. No NVIDIA tools (ptxas,
CUDA, drivers) or proprietary code are used, and nothing runs on a GPU.

## Building the tools (clang only)

All C/C++ is compiled with clang:

```bash
cmake -G Ninja -S llvm -B ../build-llvm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld \
  -DLLVM_TARGETS_TO_BUILD="NVPTX;X86" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INCLUDE_TESTS=OFF
ninja -C ../build-llvm llc opt llvm-extract FileCheck
```

Z3 (4.13.4) and re2c (4.0.2) are built from source with
`-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++`; Alive2 is configured
with `-DBUILD_TV=ON -DLLVM_DIR=../build-llvm/lib/cmake/llvm` and the same
compilers. `tv.py` looks for the binaries via `PTXTV_LLVM_BIN` (default
`/home/user/build-llvm/bin`) and `PTXTV_ALIVE`.

## Running

```bash
python3 tv.py --root llvm/test/CodeGen/NVPTX --workdir /tmp/w -j4 \
  --smt-to 5000 --max-funcs 6 --out results.jsonl llvm/test/CodeGen/NVPTX/*.ll
python3 summarize.py results.jsonl
python3 param_oob_scan.py llvm/test/CodeGen/NVPTX/*.ll   # static check, no Alive2
./show.sh FILE.ll FUNCTION                               # one function, verbose
```

For each test file, `tv.py` takes the first plain `llc` RUN line that targets
nvptx64, compiles the file, and checks every (or a sample of) defined
function. Each result is one JSON line with a `status`:

| status | meaning |
|---|---|
| `correct` | Alive2 proved the lifted PTX refines the source |
| `mismatch` | Alive2 found a counterexample, needs review |
| `mismatch-param-oob` | counterexample, and the PTX accesses a `.param` outside its declared size (decided statically) |
| `mismatch-known` | counterexample of a recognised modelling class (`known` field) |
| `timeout` / `inconclusive` | solver limit hit / Alive2 approximated |
| `lift-unsupported` | the PTX uses something the lifter does not model |
| `alive-unsupported` / `alive-error` | Alive2 cannot handle the source |
| `src-inline-asm` | the source has inline asm (opaque on one side only) |

## The model

The lifter (`ptx_lift.py`) gives each PTX instruction its documented
single-thread semantics:

* Registers are integer allocas (`.pred` is `i1`); FP values are bitcast.
* Memory is accessed through `inttoptr` of the computed address; Alive2's
  assembly mode resolves such accesses by address. Every access is required
  to be naturally aligned, as PTX requires.
* A flow-insensitive analysis recovers which object (param slot, local
  depot, global, pointer argument, dynamic alloca) a 64-bit register points
  into, so pointers that escape to calls or are returned keep provenance.
* Parameters, return values and call-site parameters are byte buffers of
  exactly the declared `.param` size. LLVM values are stored into them using
  the NVPTX data layout (sub-byte values zero-extended, narrow scalar ints
  extended per `signext`/`zeroext`, otherwise with arbitrary upper bits).
  Accesses past the declared size are UB.
* All PTX state spaces share one address space and `cvta` is the identity.
  The source is flattened the same way (Alive2 supports only address space
  0); `addrspacecast` between two different specific spaces is undefined.
* Atomics and fences have single-thread semantics; the source's atomics are
  lowered with `lower-atomic` and its fences removed accordingly.
* Rounding modes `.rn/.rz/.rm/.rp`, `.ftz`, `.sat`, `.relu` and mixed
  precision are modelled exactly (constrained intrinsics for directed
  rounding). Instructions without a rounding modifier are modelled as
  round-to-nearest; PTX lets ptxas contract such mul/add pairs, which is not
  modelled.
* Approximate instructions (`ex2.approx`, `rsqrt.approx`, `div.full`, ...)
  are uninterpreted pure functions; cross-thread operations (`shfl`, `vote`,
  `bar`, `redux`, `match`, `elect`, special registers) are uninterpreted calls
  named after the NVVM intrinsic they implement.
* NVVM intrinsics that name a single PTX instruction get that instruction's
  semantics in the source too (`nvvm_sem.py`): the intrinsic call is replaced
  by an inlined function whose body is the lifted instruction. This checks
  instruction selection and the surrounding code, not the instruction model
  itself.
* Narrow floating-point conversions (FP8 `e4m3`/`e5m2`, FP6 `e2m3`/`e3m2`/
  `s2f6`, FP4 `e2m1`, `ue8m0`, `ue5m3`, `tf32`, and the `.pzo`, `.rs` and
  `.scaled::*` variants) are uninterpreted functions keyed by destination
  type, source type and modifier set (`narrowcvt.py`); the PTX `cvt` and the
  NVVM intrinsic map to the same key, operands in instruction order. This
  checks instruction selection, modifiers, operand order and packing, not
  the narrow formats' numerics. The key ignores `.satfinite` where PTX
  requires it (FP8/FP6/FP4 destinations) and rounding on exact
  up-conversions.
* `mbarrier.*` operations are uninterpreted calls whose names are normalised
  from the PTX opcode and the NVVM intrinsic (`mbar.py`); defaults that one
  side spells out (scope `.cta`, the default semantics of each operation,
  layout `v0`) are dropped.
* Fences (including `fence.proxy.*`) and prefetches are no-ops; the source's
  fence/prefetch intrinsics are removed accordingly. `fma.rn.oob` is
  uninterpreted.
* Probe-derived opaque instructions (`probe.py`) for intrinsic families the
  lifter does not model (tcgen05, cp.async.bulk/TMA, cp.reduce, sparse
  (de)compression, applypriority, st.bulk, texture/surface, ...): each such
  call is compiled alone (immediate arguments kept, others as parameters)
  with the same llc flags; the single instruction it becomes is traced
  operand by operand back to argument bytes and result bytes. Both the
  source call and the matching instruction in the real code become one
  opaque call named after opcode and operand shape (scalars as i64,
  addresses as pointers). This checks that the backend emits that
  instruction with the right values and uses its results correctly in
  context; it does not check the intrinsic's instruction selection (that is
  taken from the backend). Results that rely on it carry `"probe": true`.
  Texture/surface handles are modelled as the reference's address (what PTX
  uses as the handle).
* Calls through a prototype to a known function (`mov.b64 %rd, f` then
  `call ..., %rd, ..., prototype`) are lifted as direct calls with the
  source's call-site type; other indirect calls use the source signature
  that matches the PTX argument count and return.
* `PTXTV_ALIVE_EXTRA` passes extra (weakening) flags to Alive2, e.g.
  `--disable-poison-input` for functions whose full check times out; results
  record the flags in `alive_extra`.
* Any modifier the lifter does not know makes the instruction unsupported,
  so nothing is silently ignored.

Assumptions taken from the open-source backend rather than an independent
source (a bug there would not be caught): the lane order of packed
`setp.f16x2 p|q` (p = low half), the `prmt` mode table, and that an `e2m1x2`
value and a single `ue8m0` scale (`.scale.n1`) occupy the low 8 bits of the
intrinsics' `i16` operands/results.

## Known false-alarm classes (labelled automatically)

* `partword-atomic-word-access`: sub-word atomics expanded into a CAS loop on
  the containing aligned 32-bit word read bytes outside the object.
* `byval-copy-to-local`: a kernel `byval` argument whose address escapes is
  copied to local memory; Alive2 treats the `byval` address as an input.
* `approximation-permitted-by-flags`: `-nvptx-prec-divf32=0/1`,
  `-nvptx-prec-sqrtf32=0`, fast-math flags or attributes allow approximation.
* `rsqrt-approx-opt-default-on`: `1/sqrt.approx(x)` to `rsqrt.approx(x)`
  (`-nvptx-rsqrt-approx-opt`, on by default).
* `nan-payload`: both results are NaN and differ only in sign or payload
  (not applied when the source uses `fabs`/`fneg`/`copysign`).
* `dynamic-alloca-size-overflow`: needs allocation sizes near 2^63.
* `approx-substitution`: one approximate instruction replaced by another
  (e.g. `div.full(1, x)` to `rcp.approx`); both are uninterpreted, so the
  PTX ISA error bounds have to be compared by hand.
* `cas-based-atomic-load`: an atomic 128-bit load implemented as
  `atom.cas.b128 [a], 0, 0`, which needs writable memory.
* `indirect-call-asm-mode`: indirect calls through a pointer loaded from
  memory; Alive2's assembly mode fails or times out even on an unchanged
  call, so indirect calls are effectively unverified.

Aligned over-reads (for example `ld.b64` of a 6-byte `<3 x half>`) also show
up as mismatches; they are hardware-safe and are triaged by hand.

See `findings/` for the defects found.
