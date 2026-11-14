"""Expand llvm.masked.load / llvm.masked.store into per-lane conditional
loads and stores (LangRef semantics: disabled lanes are not accessed and
take the pass-through value).  Alive2 does not know the current form of
these intrinsics (alignment as a parameter attribute) and treats them as
unknown calls.  Applied to the source and to the lifted code alike; the
helpers are always-inline and are inlined by the following opt run.
"""
import re

from llvmsig import parse_type, split_top, TypeParseError

CALL_RE = re.compile(r"^(\s*(?:(%[\w.$-]+)\s*=\s*)?(?:tail\s+)?call\s+)(void|<\d+ x [^>]+>)\s+"
                     r"@llvm\.masked\.(load|store)\.[\w.]+\((.*)\)(.*)$")
ESIZE = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "i128": 16, "half": 2, "bfloat": 2,
         "float": 4, "double": 8, "ptr": 8}


def _args(text):
    out = []
    for a in split_top(text):
        a = a.strip()
        t, e = parse_type(a)
        rest = a[e:].strip()
        al = re.search(r"\balign (\d+)", rest)
        rest = re.sub(r"^((noundef|nonnull|align \d+|captures\([^)]*\)|nocapture|"
                      r"dereferenceable\(\d+\))\s+)*", "", rest)
        out.append((t, rest, int(al.group(1)) if al else None))
    return out


def lane_align(A, off):
    a = A
    while off % a:
        a //= 2
    return a


def helper(kind, fname, vty, pty, A):
    m = re.fullmatch(r"<(\d+) x (.+)>", vty)
    n, et = int(m.group(1)), m.group(2).strip()
    es = ESIZE[et]
    mty = "<%d x i1>" % n
    L = []
    if kind == "load":
        L.append("define internal %s @%s(%s %%p, %s %%m, %s %%pt) alwaysinline {" % (vty, fname, pty, mty, vty))
    else:
        L.append("define internal void @%s(%s %%v, %s %%p, %s %%m) alwaysinline {" % (fname, vty, pty, mty))
    L.append("entry:\n  br label %l0")
    acc = "poison"
    for k in range(n):
        L.append("l%d:" % k)
        L.append("  %%c%d = extractelement %s %%m, i32 %d" % (k, mty, k))
        if kind == "load":
            L.append("  %%t%d = extractelement %s %%pt, i32 %d" % (k, vty, k))
        L.append("  br i1 %%c%d, label %%a%d, label %%j%d" % (k, k, k))
        L.append("a%d:" % k)
        L.append("  %%q%d = getelementptr inbounds %s, %s %%p, i32 %d" % (k, et, pty, k))
        al = lane_align(A, k * es)
        if kind == "load":
            L.append("  %%ptxtv_mlx%d = load %s, %s %%q%d, align %d" % (k, et, pty, k, al))
        else:
            L.append("  %%s%d = extractelement %s %%v, i32 %d" % (k, vty, k))
            L.append("  store %s %%s%d, %s %%q%d, align %d" % (et, k, pty, k, al))
        L.append("  br label %%j%d" % k)
        L.append("j%d:" % k)
        if kind == "load":
            L.append("  %%e%d = phi %s [ %%ptxtv_mlx%d, %%a%d ], [ %%t%d, %%l%d ]" % (k, et, k, k, k, k))
            L.append("  %%r%d = insertelement %s %s, %s %%e%d, i32 %d" % (k, vty, acc, et, k, k))
            acc = "%%r%d" % k
        L.append("  br label %%l%d" % (k + 1))
    L.append("l%d:" % n)
    L.append("  ret %s %s" % ((vty, acc) if kind == "load" else ("void", "")))
    L.append("}")
    return "\n".join(L).replace("ret void \n", "ret void\n")


def expand(text):
    out, defs = [], {}
    for line in text.split("\n"):
        m = CALL_RE.match(line) if "@llvm.masked." in line else None
        if not m:
            out.append(line)
            continue
        head, _, rt, kind, argtext, tail = m.groups()
        try:
            args = _args(argtext)
        except TypeParseError:
            out.append(line)
            continue
        if kind == "load" and len(args) == 3:
            (pty, p, A), (mty, mv, _), (vty, pt, _) = args
            vals = [(pty, p), (mty, mv), (vty, pt)]
        elif kind == "store" and len(args) == 3:
            (vty, v, _), (pty, p, A), (mty, mv, _) = args
            vals = [(vty, v), (pty, p), (mty, mv)]
        else:
            out.append(line)
            continue
        em = re.fullmatch(r"<(\d+) x (.+)>", vty)
        if not em or em.group(2).strip() not in ESIZE or not pty.startswith("ptr"):
            out.append(line)
            continue
        A = A or 1
        key = (kind, vty, pty, A)
        if key not in defs:
            fname = "ptxtv_masked_%d" % len(defs)
            defs[key] = (fname, helper(kind, fname, vty, pty, A))
        tail = re.sub(r",\s*!\S+\s+!\d+", "", tail)
        out.append("%s%s @%s(%s)%s" % (head, rt, defs[key][0],
                                        ", ".join("%s %s" % a for a in vals), tail))
    if not defs:
        return text, False
    out += [d for _, d in defs.values()]
    return "\n".join(out), True
