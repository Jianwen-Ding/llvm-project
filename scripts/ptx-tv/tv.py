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
from llvmsig import parse_sig, TypeParseError  # noqa: E402

BIN = os.environ.get("PTXTV_LLVM_BIN", "/home/user/build-llvm/bin")
ALIVE = os.environ.get("PTXTV_ALIVE", "/home/user/alive2/build/alive-tv")
WALL_TIMEOUT = int(os.environ.get("PTXTV_WALL_TIMEOUT", "0"))
LLC_FLAGS = {}   # pid -> llc flags of the file being processed


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
    rc, out, err = run([BIN + "/opt", "-S", "-mtriple=nvptx64-nvidia-cuda",
                        "-strip-debug", "-passes=lower-atomic",
                        path, "-o", "-"])
    if rc:
        raise RuntimeError("opt failed: " + err[-300:])
    # NVVM intrinsics that name one PTX instruction get that instruction's
    # semantics (inlined); the rest have no LLVM-level semantics in Alive2
    # and become uninterpreted external functions (immarg is only valid on
    # intrinsics).
    out, expanded = nvvm_sem.expand(out, ptx_lift)
    if expanded:
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
    # volatile has no meaning under single-thread semantics (Alive2 rejects it)
    text = re.sub(r"\b(load|store) volatile\b", r"\1", text)
    # Metadata that carries hints only (or assumptions, whose removal makes
    # the source more defined) and that Alive2 cannot parse.
    text = re.sub(r",\s*!(mem\.cache_hint|nontemporal|invariant\.load|tbaa|tbaa\.struct|"
                  r"alias\.scope|noalias|llvm\.access\.group|llvm\.loop|prof|annotation|"
                  r"srcloc|nvvm\.[\w.]+)\s+![\w.]+", "", text)
    src = os.path.join(workdir, "src.ll")
    with open(src, "w") as f:
        f.write(text)
    rc, _, err = run([BIN + "/opt", "-S", "-passes=verify", src, "-o", "/dev/null"])
    if rc:
        raise RuntimeError("renamed source does not verify: " + err[-300:])
    return src, text


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
    return defines, decls, globals_, typedefs


def ptx_name(n):
    if n.startswith('"'):
        n = n[1:-1]
    return "".join(c if (c.isalnum() or c in "_$") else "_$_" for c in n)


def build_name_map(defines, decls, globals_):
    m = {}
    for n in list(decls) + list(globals_):
        raw = n[1:-1] if n.startswith('"') else n
        m[raw] = n
        m[ptx_name(n)] = n
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
    # Default-on backend combine (-nvptx-rsqrt-approx-opt): 1/sqrt.approx(x)
    # -> rsqrt.approx(x) without fast-math flags.  Intentional design.
    if "rsqrt.approx" in body and "sqrt.approx" in src_text:
        return "rsqrt-approx-opt-default-on"
    # Both results are NaNs that differ only in sign/payload/quiet bit.
    vals = re.findall(r"(Source|Target) value: [^\n]*?\((\w+)\)\s*$", alive_out, re.M)
    if vals and all(v[1] in ("QNaN", "SNaN") for v in vals) and len(vals) >= 2:
        return "nan-payload"
    if "partword" in body or (
            re.search(r"and\.b64\s+%rd\d+, %rd\d+, -4;", body) and
            re.search(r"atom\.[\w.:]*cas\.b32", body)):
        return "partword-atomic-word-access"
    return None


def check_function(fname, define_line, sig, src_path, src_text, ptx,
                   name_map, decls, globals_, workdir, smt_to, keep, typedefs):
    res = {"function": fname}
    fdir = os.path.join(workdir, "f_" + re.sub(r"[^\w]", "_", fname)[:80])
    os.makedirs(fdir, exist_ok=True)
    rc, ext, err = run([BIN + "/llvm-extract", "-S", "--func=" + fname.strip('"'),
                        src_path, "-o", "-"])
    if rc:
        res.update(status="extract-failed", detail=err[-300:])
        return res
    ptxn = ptx_name(fname)
    try:
        L = ptx_lift.lift_function(ptx, sig, name_map, decls, globals_, ptxn,
                                   typedefs)
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
    src_f = os.path.join(fdir, "src.ll")
    tgt_raw = os.path.join(fdir, "tgt.raw.ll")
    tgt_f = os.path.join(fdir, "tgt.ll")
    with open(src_f, "w") as f:
        f.write(ext)
    with open(tgt_raw, "w") as f:
        f.write(tgt)
    rc, _, err = run([BIN + "/opt", "-S", "-passes=mem2reg", tgt_raw, "-o", tgt_f])
    if rc:
        res.update(status="lift-invalid-ir", detail=err[-400:])
        return res
    try:
        rc, out, err = run([ALIVE, src_f, tgt_f, "--tgt-is-asm",
                            "--disable-undef-input", "--smt-to=%d" % smt_to,
                            "--func=" + fname.strip('"')],
                           timeout=WALL_TIMEOUT or max(30, smt_to // 1000 * 4))
    except subprocess.TimeoutExpired:
        res.update(status="timeout", detail="wall clock")
        return res
    st = classify(out, err, rc)
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
    name_map = build_name_map(defines, decls, globals_)
    names = list(defines)
    if args.max_funcs and len(names) > args.max_funcs:
        import random
        rnd = random.Random(hash(path) & 0xffff)
        names = sorted(rnd.sample(names, args.max_funcs), key=names.index)
    for n in names:
        if args.func and n not in args.func:
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
    args = ap.parse_args()
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
