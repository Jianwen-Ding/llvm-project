#!/usr/bin/env python3
"""ptx-tv driver: translation-validate LLVM's NVPTX backend with Alive2.

For every function of an LLVM IR test file:
  1. compile with llc (nvptx64) using the flags from the test's RUN line,
  2. lift the PTX of that function back to LLVM IR (ptx_lift.py),
  3. check with `alive-tv --tgt-is-asm` that the lifted code refines the
     original function.

Usage: tv.py [options] FILE.ll [FILE.ll ...]
Writes one JSON object per function to --out (default: stdout).
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptx_lift  # noqa: E402
import nvvm_sem  # noqa: E402
import asm_sem  # noqa: E402
import arbfp  # noqa: E402
import masked  # noqa: E402
import probe  # noqa: E402
probe.LLC = os.environ.get("PTXTV_LLVM_BIN", "/home/user/build-llvm/bin") + "/llc"
from llvmsig import parse_sig, TypeParseError  # noqa: E402

BIN = os.environ.get("PTXTV_LLVM_BIN", "/home/user/build-llvm/bin")
ALIVE = os.environ.get("PTXTV_ALIVE", "/home/user/alive2/build/alive-tv")
WALL_TIMEOUT = int(os.environ.get("PTXTV_WALL_TIMEOUT", "0"))
LLC_FLAGS = {}   # pid -> llc flags of the file being processed
# extra (weakening) Alive2 flags, e.g. --disable-poison-input; recorded per result
ALIVE_EXTRA = os.environ.get("PTXTV_ALIVE_EXTRA", "").split()
UNROLL = int(os.environ.get("PTXTV_UNROLL", "8"))
SKIP = {}        # file -> functions to skip (--skip-from)
PROBE_LLC = {}   # pid -> llc args used for intrinsic probes
probe_mod = None


def run(cmd, inp=None, timeout=120):
    p = subprocess.run(cmd, input=inp, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


# --------------------------------------------------------------------------
# RUN line handling
# --------------------------------------------------------------------------

def run_lines(text):
    lines, cur = [], ""
    for l in text.split("\n"):
        m = re.match(r"\s*;\s*RUN:\s?(.*)$", l)
        if not m:
            if cur:
                lines.append(cur)
                cur = ""
            continue
        s = m.group(1)
        if s.rstrip().endswith("\\"):
            cur += s.rstrip()[:-1] + " "
        else:
            lines.append(cur + s)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def llc_args(text):
    """Pick the first plain `llc` RUN line targeting nvptx64; return its args."""
    tm = re.search(r'^target triple\s*=\s*"([^"]*)"', text, re.M)
    file_triple = tm.group(1) if tm else ""
    for rl in run_lines(text):
        if "%if" in rl or "sed " in rl or "not llc" in rl or \
                "not --crash" in rl or "llvm-mc" in rl:
            continue
        for seg in rl.split("|"):
            seg = seg.strip()
            if not seg.startswith("llc ") and seg != "llc":
                continue
            toks = shlex.split(seg)[1:]
            arch = [t for t in toks if re.match(r"-+(march|mtriple)=", t)]
            if arch:
                if not any("nvptx64" in t for t in arch):
                    continue
            elif not file_triple.startswith("nvptx64"):
                continue
            out = []
            skip = False
            for t in toks:
                if skip:
                    skip = False
                    continue
                if t in ("<", "%s"):
                    continue
                if t in ("-o", ">", "2>&1", "2>"):
                    skip = t != "2>&1"
                    continue
                if t.startswith("-o") and len(t) > 2:
                    continue
                if re.match(r"-+(debug|print|stop-|start-|view|time)", t):
                    continue
                if "%" in t:
                    out = None
                    break
                if re.match(r"-+filetype", t) and t.split("=")[-1] != "asm":
                    out = None
                    break
                out.append(t)
            if out is not None:
                if not arch:
                    out.append("-mtriple=" + file_triple)
                return out
    return None


# --------------------------------------------------------------------------
# Source module handling
# --------------------------------------------------------------------------

def flatten_address_spaces(text):
    """Map every address space to 0 (Alive2 only supports address space 0).

    This matches the lifter's model, in which the PTX state spaces share one
    address space and cvta is the identity.  addrspacecast becomes a no-op
    bitcast; overloaded intrinsic names are re-mangled for p0.
    """
    # A cast between two different specific (non-generic) address spaces has
    # no defined result on NVPTX (the backend emits an undefined register).
    def cast(m):
        a, b = m.group(2), m.group(4)
        if a != "0" and b != "0" and a != b:
            return "%s = freeze ptr addrspace(%s) poison" % (m.group(1), b)
        return m.group(0)
    text = re.sub(r"(%[\w.$-]+) = addrspacecast ptr addrspace\((\d+)\) (%[\w.$-]+) "
                  r"to ptr addrspace\((\d+)\)", cast, text)
    text = re.sub(r",\s*addrspace\(\d+\)", "", text)   # alloca ..., addrspace(N)
    text = re.sub(r"\s*addrspace\(\d+\)", "", text)
    text = re.sub(r"\baddrspacecast\b", "bitcast", text)
    # bitcast ptr -> ptr instructions are no-ops; keep them (they verify).
    text = re.sub(r"(@llvm\.[\w.]*)",
                  lambda m: re.sub(r"\.p\d+(?=\.|$)", ".p0", m.group(1)), text)
    out, seen = [], set()
    for l in text.split("\n"):
        if l.startswith("declare"):
            m = re.search(r"@([\w.$\"-]+)\(", l)
            if m:
                if m.group(1) in seen:
                    continue
                seen.add(m.group(1))
        out.append(l)
    return "\n".join(out)


def prepare_source(path, workdir):
    """Strip debug info and rename llvm.nvvm.* intrinsics to plain functions."""
    # Atomics are given single-thread semantics (as in the lifted code),
    # which Alive2 can reason about: lower them to plain loads and stores.
    # llc parses the module with the target's datalayout (overriding the
    # file's), which fixes the alignment of accesses written without one:
    # do the same here.
    raw = re.sub(r"^target datalayout\s*=.*$", "", open(path).read(), flags=re.M)
    nodl = os.path.join(workdir, "nodl.ll")
    with open(nodl, "w") as f:
        f.write(raw)
    # Variadic function definitions: NVPTX lowers them with LLVM's
    # ExpandVariadics (the variable arguments become an explicit buffer
    # pointer); apply the same pass so the callee can be compared.  The pass
    # itself is trusted.
    passes = "lower-atomic"
    extra = []
    if re.search(r"^define [^\n]*\.\.\.\)", raw, re.M):
        passes = "expand-variadics,lower-atomic"
        extra = ["-expand-variadics-override=lowering"]
    rc, out, err = run([BIN + "/opt", "-S", "-mtriple=nvptx64-nvidia-cuda",
                        "-strip-debug", "-passes=" + passes] + extra +
                       [nodl, "-o", "-"])
    if rc:
        raise RuntimeError("opt failed: " + err[-300:])
    # NVVM intrinsics that name one PTX instruction get that instruction's
    # semantics (inlined); the rest have no LLVM-level semantics in Alive2
    # and become uninterpreted external functions (immarg is only valid on
    # intrinsics).
    out, asm_expanded = asm_sem.expand(out, ptx_lift)
    out, arb_expanded = arbfp.expand(out)
    out, arb_to = arbfp.expand_to(out)
    arb_expanded = arb_expanded or arb_to
    asm_expanded = asm_expanded or arb_expanded
    out, expanded = nvvm_sem.expand(out, ptx_lift, PROBE_LLC.get(os.getpid()))
    out, masked_expanded = masked.expand(out)   # also those created above
    expanded = expanded or asm_expanded or masked_expanded
    if expanded:
        # inline only our helpers: the test's own alwaysinline functions may
        # be called by name in the PTX
        out = "\n".join(
            re.sub(r"\balwaysinline\b\s*", "", l)
            if (l.startswith("attributes #") or
                (l.startswith("define ") and "@ptxtv_" not in l)) else l
            for l in out.split("\n"))
        tmp = os.path.join(workdir, "src.sem.ll")
        with open(tmp, "w") as f:
            f.write(out)
        rc, out, err = run([BIN + "/opt", "-S",
                            "-passes=always-inline,globaldce,sroa,mem2reg", tmp, "-o", "-"])
        if rc:
            raise RuntimeError("intrinsic expansion failed: " + err[-300:])
    lines = []
    for l in out.split("\n"):
        if "@llvm.nvvm." in l:
            l = l.replace("@llvm.nvvm.", "@nvvm.")
            if l.startswith("declare"):
                l = l.replace(" immarg", "")
        lines.append(l)
    text = flatten_address_spaces("\n".join(lines))
    # noalias on parameters is unsupported by Alive2; dropping an assumption
    # makes the source more defined (cannot hide a defect)
    text = re.sub(r"(define [^\n]*)", lambda m: re.sub(r"\bnoalias\b\s*", "", m.group(1)), text)
    # Unnamed globals (@0, @1, ...) are printed as __unnamed_<N+1> in PTX;
    # give them those names so functions can be extracted and matched.
    text = re.sub(r"@(\d+)\b(?!\s*=\s*!)", lambda m: "@__unnamed_%d" % (int(m.group(1)) + 1), text)
    # A kernel's return value is not observable (.entry has no return
    # parameter): return poison from non-void kernels, as the lifted code does.
    def kernel_ret(m):
        body = m.group(0)
        return re.sub(r"^(\s*)ret (?!void)(.+?) (%[\w.$-]+|-?\d+|true|false|[\w.]+)$",
                      lambda r: "%sret %s poison" % (r.group(1), r.group(2)), body, flags=re.M)
    text = re.sub(r"^define [^\n]*ptx_kernel (?!void)[^\n]*\{.*?^\}", kernel_ret, text, flags=re.M | re.S)
    # volatile has no meaning under single-thread semantics (Alive2 rejects it)
    text = re.sub(r"\b(load|store) volatile\b", r"\1", text)
    # Metadata that carries hints only (or assumptions, whose removal makes
    # the source more defined) and that Alive2 cannot parse.
    text = re.sub(r",\s*!(mem\.cache_hint|nontemporal|invariant\.load|tbaa|tbaa\.struct|"
                  r"alias\.scope|noalias|llvm\.access\.group|llvm\.loop|prof|annotation|"
                  r"srcloc|nvvm\.[\w.]+)\s+![\w.]+", "", text)
    text = rewrite_i1_fp(text)
    # stack save/restore: opaque calls (the lifter emits the same)
    text = re.sub(r"@llvm\.stacksave(\.p\d+)?\(", "@ptx.stacksave(", text)
    text = re.sub(r"@llvm\.stackrestore(\.p\d+)?\(", "@ptx.stackrestore(", text)
    # alignstack(N) only constrains stack alignment (Alive2 rejects it)
    text = re.sub(r"\s*\balignstack\(\d+\)", "", text)
    src = os.path.join(workdir, "src.ll")
    with open(src, "w") as f:
        f.write(text)
    rc, _, err = run([BIN + "/opt", "-S", "-passes=verify", src, "-o", "/dev/null"])
    if rc:
        raise RuntimeError("renamed source does not verify: " + err[-300:])
    return src, text


FP_ONE = {"half": ("0xH3C00", "0xHBC00"), "bfloat": ("0xR3F80", "0xRBF80"),
          "float": ("1.0", "-1.0"), "double": ("1.0", "-1.0")}


def rewrite_i1_fp(text):
    """Scalar conversions between i1 and FP, which Alive2's SMT encoding
    rejects ("invalid extract application"), as equivalent selects.  Done
    after the last optimisation, so the poison arms stay."""
    def itofp(m):
        ind, r, op, x, t = m.groups()
        one, mone = FP_ONE[t]
        return "%s%s = select i1 %s, %s %s, %s 0.0" % (ind, r, x, t, one if op == "uitofp" else mone, t)
    text = re.sub(r"^(\s*)(%[\w.$-]+) = (uitofp|sitofp) i1 (%[\w.$-]+|true|false) to "
                  r"(half|bfloat|float|double)\s*$", itofp, text, flags=re.M)

    def fptoi(m):
        ind, r, op, t, x = m.groups()
        lo, hi = (0, 1) if op == "fptoui" else (-1, 0)
        w = "%s.w" % r
        return ("%s%s = %s %s %s to i32\n"
                "%s%s.lo = icmp sge i32 %s, %d\n%s%s.hi = icmp sle i32 %s, %d\n"
                "%s%s.ok = and i1 %s.lo, %s.hi\n%s%s.t = trunc i32 %s to i1\n"
                "%s%s = select i1 %s.ok, i1 %s.t, i1 poison") % (
                    ind, w, op, t, x, ind, r, w, lo, ind, r, w, hi, ind, r, r, r,
                    ind, r, w, ind, r, r, r)
    text = re.sub(r"^(\s*)(%[\w.$-]+) = (fptoui|fptosi) (half|bfloat|float|double) (\S+) to i1\s*$",
                  fptoi, text, flags=re.M)
    return text


def module_symbols(text):
    defines, decls, globals_, typedefs = {}, {}, {}, {}
    for l in text.split("\n"):
        tm = re.match(r'(%("[^"]*"|[-a-zA-Z$._0-9]+))\s*=\s*type\s+(.*)$', l)
        if tm:
            typedefs[tm.group(1)] = tm.group(3).strip()
            continue
        if l.startswith("define "):
            s = parse_sig(l)
            defines[s.name] = (l, s)
            decls[s.name] = s
        elif l.startswith("declare "):
            try:
                s = parse_sig(l)
                decls[s.name] = s
            except TypeParseError:
                pass
        else:
            m = re.match(r'@("[^"]*"|[-a-zA-Z$._0-9]+)\s*=\s*(.*)$', l)
            if m:
                rest = m.group(2)
                am = re.search(r"\baddrspace\((\d+)\)", rest.split("global")[0].split("constant")[0])
                globals_[m.group(1)] = "ptr addrspace(%s)" % am.group(1) if am and am.group(1) != "0" else "ptr"
    # function aliases: @b = ... alias <ty>, ptr @a  -> callable like @a
    for m in re.finditer(r'^@("[^"]*"|[-a-zA-Z$._0-9]+)\s*=[^\n]*\balias\b[^\n]*@("[^"]*"|[-a-zA-Z$._0-9]+)\s*$',
                         text, re.M):
        if m.group(2) in decls and m.group(1) not in decls:
            import copy
            sg = copy.copy(decls[m.group(2)])
            sg.name = m.group(1)
            decls[m.group(1)] = sg
    return defines, decls, globals_, typedefs


def ptx_name(n):
    if n.startswith('"'):
        n = n[1:-1]
    return "".join(c if (c.isalnum() or c in "_$") else "_$_" for c in n)


RENAMES = {}


def nvptx_renames(text):
    """LLVM name -> PTX name, emulating NVPTXAssignValidGlobalNames: local
    globals (and external .shared ones), then local functions, in module
    order, get NVPTX::getValidPTXIdentifier names; a collision makes the
    name unique with the module's running counter (no separator), as
    ValueSymbolTable::makeUniqueName does for global values."""
    NAME = r'@("[^"]*"|[-a-zA-Z$._0-9]+)'
    unq = lambda n: n[1:-1] if n.startswith('"') else n
    allnames, gvars, funcs = set(), [], []
    for l in text.split("\n"):
        m = re.match(NAME + r"\s*=\s*(.*)$", l)
        if m:
            allnames.add(unq(m.group(1)))
            rest = m.group(2)
            if re.match(r"(\w+\s+)*(alias|ifunc)\b", rest):
                continue
            head = re.split(r"\b(global|constant)\b", rest)[0]
            local = re.search(r"\b(private|internal)\b", head)
            ext_shared = not re.search(r"\b(private|internal|linkonce\w*|weak\w*|common|"
                                       r"available_externally|extern_weak)\b", head) and \
                "addrspace(3)" in head
            if local or ext_shared:
                gvars.append(m.group(1))
            continue
        m = re.match(r"(define|declare)\b[^@]*" + NAME + r"\(", l)
        if m:
            allnames.add(unq(m.group(2)))
            if m.group(1) == "define" and re.match(r"define\s+(private|internal)\b", l):
                funcs.append(m.group(2))
    counter, out = 0, {}
    for n in gvars + funcs:
        raw = unq(n)
        valid = ptx_name(n)
        if valid == raw:
            continue
        allnames.discard(raw)
        name = valid
        while name in allnames:
            counter += 1
            name = valid + str(counter)
        allnames.add(name)
        out[n] = name
    return out


def build_name_map(defines, decls, globals_):
    m = {}
    for n in list(decls) + list(globals_):
        raw = n[1:-1] if n.startswith('"') else n
        m[raw] = n
        m[RENAMES.get(n, ptx_name(n))] = n
    return m


def replace_define(text, name, new_fn):
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("define ") and re.search(r"@%s\(" % re.escape(name), l):
            j = i
            while lines[j].rstrip() != "}":
                j += 1
            out.append(new_fn)
            i = j + 1
            continue
        out.append(l)
        i += 1
    return "\n".join(out)


# --------------------------------------------------------------------------
# Alive2
# --------------------------------------------------------------------------

def classify(out, err, rc):
    o = out + err
    if "doesn't type check" in o:
        return "alive-error"
    if "Transformation seems to be correct" in o:
        return "correct"
    if "Transformation doesn't verify" in o:
        return "mismatch"
    if "Timeout" in o or "timeout" in o:
        return "timeout"
    if "Couldn't prove the correctness" in o or "approximated" in o:
        return "inconclusive"
    if "ERROR: Unsupported" in o or "Unsupported" in o:
        return "alive-unsupported"
    if "out of memory" in o or "Out of memory" in o:
        return "oom"
    if "ERROR" in o:
        return "alive-error"
    if rc != 0:
        return "alive-crash"
    return "alive-other"


def indirect_call_sigs(module_text, fname):
    """LLVM signatures of the indirect calls in function `fname`."""
    m = re.search(r"^define [^\n]*@%s\(.*?^}" % re.escape(fname), module_text, re.M | re.S)
    if not m:
        return []
    out = []
    for cm in re.finditer(r"\bcall\s+(.*?)\s+(%[\w.$-]+|@[\w.$\"-]+)\((.*)\)", m.group(0)):
        if cm.group(2).startswith("@llvm.") or cm.group(2).startswith("@nvvm."):
            continue
        head = cm.group(1)
        fnty_vararg = False
        if "(" in head:      # explicit function type: call RET (T1, ...) @f(...)
            fnty_vararg = "..." in head[head.index("("):]
            head = head[:head.index("(")]
        ret = re.sub(r"\b(fast|nnan|ninf|nsz|arcp|contract|afn|reassoc|tail|musttail|notail|noundef|zeroext|signext)\b", "",
                     head).strip().split()
        if not ret:
            continue
        try:
            sg = parse_sig("declare %s @f(%s)" % (ret[-1], cm.group(3)))
        except (TypeParseError, ValueError):
            continue
        for a in sg.args:   # call-site operands, not parameter attributes
            a.attrs = "byval(%s)" % a.byval_type() if a.byval_type() else ""
            a.name = None
        sg.callee = cm.group(2)[1:] if cm.group(2).startswith("@") else None
        sg.vararg = sg.vararg or fnty_vararg
        out.append(sg)
    return out


def function_ptx(ptx, name):
    m = re.search(r"\.(?:func|entry)[^\n]*?\b%s\(" % re.escape(name), ptx)
    if not m:
        return ""
    e = ptx.find("-- End function", m.end())
    return ptx[m.start():e if e > 0 else len(ptx)]


def param_oob(ptx, name):
    """Static check: param-space accesses beyond the declared .param size.

    Returns a list of "sym+off (N bytes) > size" strings.  This is decided
    from the PTX alone and does not depend on the lifter or Alive2.
    """
    funcs = ptx_lift.parse_module(ptx)
    f = funcs.get(name)
    if f is None:
        return []
    sizes = {p.name: p.size for p in f.params + f.ret}
    out = []
    try:
        items = ptx_lift.parse_body(f.body)
    except ptx_lift.Unsupported:
        return []
    scopes = [dict(sizes)]
    for it in items:
        if it[0] == "open":
            scopes.append({})
        elif it[0] == "close":
            scopes.pop()
        elif it[0] == "stmt" and it[1].startswith(".param"):
            pp = ptx_lift.PParam(it[1])
            scopes[-1][pp.name] = pp.size
        elif it[0] == "stmt":
            m = re.match(r"(?:@!?%\w+\s+)?(ld|st)\.param[\w:]*((?:\.[\w:]+)*)\s+(.*)", it[1])
            if not m:
                continue
            mods = m.group(2).split(".")
            vec = 1
            for md in mods:
                if md in ("v2", "v4", "v8"):
                    vec = int(md[1:])
            ts = [md for md in mods if md in ptx_lift.TBITS]
            if not ts:
                continue
            nbytes = max(ptx_lift.TBITS[ts[0]] // 8, 1) * vec
            am = re.search(r"\[\s*([\w$]+)\s*(?:\+\s*(\d+))?\s*\]", m.group(3))
            if not am:
                continue
            sym, off = am.group(1), int(am.group(2) or 0)
            size = None
            for sc in reversed(scopes):
                if sym in sc:
                    size = sc[sym]
                    break
            if size is not None and off + nbytes > size:
                out.append("%s %s+%d (%d bytes) > %d" % (m.group(1), sym, off, nbytes, size))
    return out


APPROX_FLAGS = re.compile(r"-nvptx-prec-(divf32|sqrtf32)=[01]\b|-enable-unsafe-fp-math|"
                          r"-fp-contract=fast|-nvptx-fma-level=2|--?enable-no-nans-fp-math|"
                          r"-denormal-fp-math")


def known_false_alarm(ptx, name, alive_out, src_text="", llc_flags=""):
    """Recognise mismatches caused by modelling limits, not backend bugs."""
    body = function_ptx(ptx, name)
    # Approximation / contraction explicitly permitted by llc flags or by
    # fast-math flags/attributes in the source.
    if APPROX_FLAGS.search(llc_flags) or re.search(
            r"\b(afn|arcp|contract|reassoc|fast)\b|\"unsafe-fp-math\"=\"true\"|"
            r"\"no-nans-fp-math\"=\"true\"|\"approx-func-fp-math\"=\"true\"", src_text):
        if re.search(r"\.(approx|full)\b|\bfma\.|\bmad\.|\bex2|\blg2|\brsqrt|\bftz\b", body):
            return "approximation-permitted-by-flags"
    # Sub-word atomics are expanded (AtomicExpand) into a CAS loop on the
    # aligned 32-bit word containing the value, which may extend past the
    # end of the object.  Hardware cannot fault inside an aligned word, but
    # Alive2's object bounds make the extra bytes out of bounds.
    # A kernel byval argument whose address escapes is copied by the backend
    # (NVPTXLowerArgs) from read-only param space to a local copy that is
    # then used consistently.  Alive2 models the byval address as an input,
    # so the change of object identity shows up as a mismatch.
    if "byval(" in src_text and "__local_depot" in body and "st.local" in body \
            and re.search(r"ld\.param[\w.:]*\s+%\w+, \[" + re.escape(name) + r"_param_\d+", body):
        return "byval-copy-to-local"
    # Dynamic alloca: counterexamples need sizes near 2^63, where the
    # backend's round-up of the size wraps; Alive2 lets the source alloca
    # of that size succeed.
    if re.search(r"\balloca\.u(32|64)\b", body) and re.search(r"alloca [^,\n]+, i\d+ %", src_text):
        return "dynamic-alloca-size-overflow"
    # Indirect call through a function pointer loaded from memory: Alive2's
    # assembly mode cannot relate such calls (even an unchanged call times
    # out or fails), so these are not verified.
    if ".callprototype" in body and re.search(r"call[\w.]*\s+(\([^)]*\),\s*)?%rd\d+", body):
        return "indirect-call-asm-mode"
    # One approximate operation replaced by another (e.g. div.full(1, x) ->
    # rcp.approx): both sides are uninterpreted approximations, so they can
    # never be related; the PTX ISA error bounds have to be compared by hand.
    if re.search(r"@ptx\.\w+\.(approx|full)", src_text) and re.search(r"\.(approx|full)\b", body):
        return "approx-substitution"
    # Atomic (volatile) 128-bit load implemented as cas(0, 0): needs the
    # memory to be writable, which the source load does not require.
    if re.search(r"atom\.[\w.:]*cas\.b128", body) and re.search(r"\bload (atomic )?i128\b", src_text) \
            and "cmpxchg" not in src_text:
        return "cas-based-atomic-load"
    # Loads of 3-element vectors widened to the next aligned power of two
    # (e.g. ld.b64 for a 6-byte <3 x half>): reads past the object but stay
    # inside the aligned granule, which cannot fault.
    if re.search(r"load <3 x ", src_text) and re.search(r"\bld\.[\w.:]*b(32|64)\b", body):
        return "aligned-overread-widened-load"
    # A masked load lowered to a full-width aligned load whose unused bytes
    # are marked with `.pragma "used_bytes_mask"`: reads disabled lanes (UB
    # in LLVM terms if they are not dereferenceable), but stays within an
    # aligned granule, so it cannot fault on the hardware.  Intentional.
    # (the pragma is only emitted for PTX >= 8.3; the lane loads of an
    # expanded masked load are named ptxtv_mlx*)
    if ('used_bytes_mask' in body or re.search(r"%ptxtv_mlx\d", src_text)) and \
            "Source is more defined than target" in alive_out:
        return "masked-load-widened"
    # Default-on backend combine (-nvptx-rsqrt-approx-opt): 1/sqrt.approx(x)
    # -> rsqrt.approx(x) without fast-math flags.  Intentional design.
    if "rsqrt.approx" in body and "sqrt.approx" in src_text:
        return "rsqrt-approx-opt-default-on"
    # Both results are NaNs that differ only in sign/payload/quiet bit.
    vals = re.findall(r"(Source|Target) value: [^\n]*?\((\w+)\)\s*$", alive_out, re.M)
    if vals and all(v[1] in ("QNaN", "SNaN") for v in vals) and len(vals) >= 2:
        # fneg/fabs/copysign must preserve NaN bits exactly (LangRef), so a
        # NaN-only difference is still a real mismatch when they are involved
        if not re.search(r"@llvm\.(fabs|copysign)|\bfneg\b", src_text):
            return "nan-payload"
    # vectors: every lane equal or NaN on both sides
    sv = re.search(r"^Source value: <(.*)>\s*$", alive_out, re.M)
    tv_ = re.search(r"^Target value: <(.*)>\s*$", alive_out, re.M)
    if sv and tv_ and not re.search(r"@llvm\.(fabs|copysign)|\bfneg\b", src_text):
        sl = [x.strip() for x in sv.group(1).split(",")]
        tl = [x.strip() for x in tv_.group(1).split(",")]
        isnan = lambda x: x.endswith("(QNaN)") or x.endswith("(SNaN)")
        if len(sl) == len(tl) and any(a != b for a, b in zip(sl, tl)) and \
                all(a == b or (isnan(a) and isnan(b)) for a, b in zip(sl, tl)):
            return "nan-payload"
    if "partword" in body or (
            re.search(r"and\.b64\s+%rd\d+, %rd\d+, -4;", body) and
            re.search(r"atom\.[\w.:]*\.b32", body)):
        return "partword-atomic-word-access"
    return None


def check_function(fname, define_line, sig, src_path, src_text, ptx,
                   name_map, decls, globals_, workdir, smt_to, keep, typedefs):
    res = {"function": fname}
    fdir = os.path.join(workdir, "f_" + re.sub(r"[^\w]", "_", fname)[:80])
    os.makedirs(fdir, exist_ok=True)
    rc, ext, err = run([BIN + "/llvm-extract", "-S", "--keep-const-init", "--func=" + fname.strip('"'),
                        src_path, "-o", "-"])
    if rc:
        res.update(status="extract-failed", detail=err[-300:])
        return res
    # Initializers of constant globals holding i1 values cannot be read
    # bytewise in Alive2's assembly mode (sub-byte memory); drop them there.
    if re.search(r"^@[^\n]*\bconstant\b[^\n]*\bi1\b", ext, re.M):
        rc, ext, err = run([BIN + "/llvm-extract", "-S", "--func=" + fname.strip('"'),
                            src_path, "-o", "-"])
        if rc:
            res.update(status="extract-failed", detail=err[-300:])
            return res
    ptxn = RENAMES.get(fname, ptx_name(fname))
    try:
        L = ptx_lift.lift_function(ptx, sig, name_map, decls, globals_, ptxn,
                                   typedefs, indirect_call_sigs(ext, fname))
    except ptx_lift.Unsupported as e:
        res.update(status="lift-unsupported", detail=str(e)[:200])
        return res
    except Exception as e:  # lifter bug
        res.update(status="lift-crash", detail=(repr(e) + traceback.format_exc()[-400:]))
        return res
    if re.search(r"\bcall [^\n]*\basm\b", ext):
        res.update(status="src-inline-asm")
        return res
    ext_defines = [l for l in ext.split("\n") if l.startswith("define ")]
    dl = [l for l in ext_defines if re.search(r"@%s\(" % re.escape(fname), l)]
    if not dl:
        res.update(status="extract-failed", detail="define not found")
        return res
    new_fn = ptx_lift.render(L, dl[0])
    tgt = replace_define(ext, fname, new_fn)
    have = set(re.findall(r"^declare [^@]*@([\w.$\"-]+)\(", tgt, re.M))
    extra = [d for n, d in L.extra_decls.items() if n not in have]
    tgt += "\n" + "\n".join(extra) + "\n"
    tgt, tgt_masked = masked.expand(tgt)
    src_f = os.path.join(fdir, "src.ll")
    tgt_raw = os.path.join(fdir, "tgt.raw.ll")
    tgt_f = os.path.join(fdir, "tgt.ll")
    with open(src_f, "w") as f:
        f.write(ext)
    with open(tgt_raw, "w") as f:
        f.write(tgt)
    rc, _, err = run([BIN + "/opt", "-S", "-passes=" + (
        "always-inline,globaldce,mem2reg" if tgt_masked else "mem2reg"), tgt_raw, "-o", tgt_f])
    if rc:
        res.update(status="lift-invalid-ir", detail=err[-400:])
        return res
    try:
        rc, out, err = run([ALIVE, src_f, tgt_f, "--tgt-is-asm",
                            "--disable-undef-input", "--smt-to=%d" % smt_to] + ALIVE_EXTRA + [
                            "--func=" + fname.strip('"')],
                           timeout=WALL_TIMEOUT or max(30, smt_to // 1000 * 4))
    except subprocess.TimeoutExpired:
        res.update(status="timeout", detail="wall clock")
        return res
    # Loops: Alive2 only reasons about loop-free code unless asked to unroll.
    # Retry with bounded unrolling; such results only cover executions with
    # at most UNROLL iterations per loop and are tagged.
    if "doesn't reach a return instruction" in out and UNROLL:
        try:
            rc, out, err = run([ALIVE, src_f, tgt_f, "--tgt-is-asm",
                                "--disable-undef-input", "--smt-to=%d" % smt_to,
                                "--src-unroll=%d" % UNROLL, "--tgt-unroll=%d" % UNROLL] +
                               ALIVE_EXTRA + ["--func=" + fname.strip('"')],
                               timeout=WALL_TIMEOUT or max(30, smt_to // 1000 * 4))
        except subprocess.TimeoutExpired:
            res.update(status="timeout", detail="wall clock", unroll=UNROLL)
            return res
        res["unroll"] = UNROLL
    st = classify(out, err, rc)
    if ALIVE_EXTRA:
        res["alive_extra"] = " ".join(ALIVE_EXTRA)
    if nvvm_sem.OPAQUE_COPIES:
        res["copy_model"] = "opaque"
    if L.used_probe:
        res["probe"] = True   # relies on a probe-derived opaque instruction
    if st == "mismatch":
        oob = param_oob(ptx, ptxn)
        known = known_false_alarm(ptx, ptxn, out + err, ext, LLC_FLAGS.get(os.getpid(), ""))
        if oob:
            # real defect class: param accesses outside the declared size
            res["param_oob"] = oob[:6]
            st = "mismatch-param-oob"
        elif known:
            st = "mismatch-known"
            res["known"] = known
    res["status"] = st
    if st != "correct":
        res["detail"] = (out + err)[-1500:]
    if st in ("mismatch",) or keep:
        res["dir"] = fdir
    return res


def process_file(path, args):
    results = []
    base = {"file": os.path.relpath(path, args.root) if args.root else path}
    try:
        text = open(path).read()
    except Exception as e:
        return [dict(base, status="read-failed")]
    largs = llc_args(text)
    if largs is None:
        return [dict(base, status="no-llc-run")]
    workdir = tempfile.mkdtemp(prefix="ptxtv_", dir=args.workdir)
    LLC_FLAGS[os.getpid()] = " ".join(largs)
    PROBE_LLC[os.getpid()] = largs
    try:
        rc, ptx, err = run([BIN + "/llc"] + largs + [path, "-o", "-"], timeout=300)
    except subprocess.TimeoutExpired:
        return [dict(base, status="llc-timeout")]
    if rc:
        return [dict(base, status="llc-failed", detail=err[-300:])]
    with open(os.path.join(workdir, "out.ptx"), "w") as f:
        f.write(ptx)
    try:
        src_path, src_text = prepare_source(path, workdir)
        defines, decls, globals_, typedefs = module_symbols(src_text)
    except Exception as e:
        return [dict(base, status="source-prep-failed", detail=str(e)[:300])]
    RENAMES.clear()
    RENAMES.update(nvptx_renames(src_text))
    name_map = build_name_map(defines, decls, globals_)
    names = list(defines)
    skip_here = SKIP.get(base["file"], set())
    if skip_here:
        names = [n for n in names if n not in skip_here]
    if args.max_funcs and len(names) > args.max_funcs:
        import random
        rnd = random.Random(hash(path) & 0xffff)
        names = sorted(rnd.sample(names, args.max_funcs), key=names.index)
    skip = SKIP.get(base["file"], set())
    for n in names:
        if args.func and n not in args.func:
            continue
        if n in skip:
            continue
        line, sig = defines[n]
        try:
            r = check_function(n, line, sig, src_path, src_text, ptx, name_map,
                               decls, globals_, workdir, args.smt_to, args.keep,
                               typedefs)
        except subprocess.TimeoutExpired:
            r = {"function": n, "status": "timeout", "detail": "tool timeout"}
        except Exception as e:
            r = {"function": n, "status": "driver-crash", "detail": repr(e)[:300]}
        r.update(base)
        r["llc"] = " ".join(largs)
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="-")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument("--smt-to", type=int, default=20000)
    ap.add_argument("--max-funcs", type=int, default=0)
    ap.add_argument("--func", action="append")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--skip-from", default=None,
                    help="skip functions already present in this results file")
    ap.add_argument("--resume", action="store_true",
                    help="skip files already present in --out")
    args = ap.parse_args()
    if args.resume and args.out != "-" and os.path.exists(args.out):
        done = set()
        for l in open(args.out):
            try:
                done.add(json.loads(l)["file"])
            except Exception:
                pass
        args.files = [f for f in args.files
                      if (os.path.relpath(f, args.root) if args.root else f) not in done]
    if args.skip_from:
        for l in open(args.skip_from):
            r = json.loads(l)
            if "function" in r:
                SKIP.setdefault(r["file"], set()).add(r["function"])
    out = sys.stdout if args.out == "-" else open(args.out, "a")
    if args.j <= 1:
        for f in args.files:
            for r in process_file(f, args):
                out.write(json.dumps(r) + "\n")
                out.flush()
        return
    with ProcessPoolExecutor(args.j) as ex:
        futs = {ex.submit(process_file, f, args): f for f in args.files}
        for fu in as_completed(futs):
            try:
                rs = fu.result()
            except Exception as e:
                rs = [{"file": futs[fu], "status": "driver-crash", "detail": repr(e)[:300]}]
            for r in rs:
                out.write(json.dumps(r) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
