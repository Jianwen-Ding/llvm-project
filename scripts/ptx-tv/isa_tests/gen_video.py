#!/usr/bin/env python3
"""Generate video-instruction self-tests from the PTX ISA pseudo-code
(9.7.20: partSelectSignExtend, optSaturate returning .s33, optSecOp on .s33
values with c signed per dtype, optMerge; extractAndSignExt_2/_4 over the
halfwords/bytes of the pair b:a).  Writes video_scalar.{,ref.}ll and
video_simdsel.{,ref.}ll next to this script."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


class F:
    def __init__(self):
        self.b, self.n = [], 0

    def v(self, s):
        self.n += 1
        nm = "%%v%d" % self.n
        self.b.append("  %s = %s" % (nm, s))
        return nm


def clamp(f, x, lo, hi, ty="i128"):
    c1 = f.v("icmp slt %s %s, %d" % (ty, x, lo))
    x = f.v("select i1 %s, %s %d, %s %s" % (c1, ty, lo, ty, x))
    c2 = f.v("icmp sgt %s %s, %d" % (ty, x, hi))
    return f.v("select i1 %s, %s %d, %s %s" % (c2, ty, hi, ty, x))


def asm_fn(name, opc, nops):
    if nops == 3:
        return ('define i32 @%s(i32 %%a, i32 %%b, i32 %%c) {\n  %%d = call i32 asm "%s $0, $1, $2;", '
                '"=r,r,r"(i32 %%a, i32 %%b)\n  ret i32 %%d\n}\n' % (name, opc))
    return ('define i32 @%s(i32 %%a, i32 %%b, i32 %%c) {\n  %%d = call i32 asm "%s $0, $1, $2, $3;", '
            '"=r,r,r,r"(i32 %%a, i32 %%b, i32 %%c)\n  ret i32 %%d\n}\n' % (name, opc))


def scalar():
    cases = []
    for op in ("vadd", "vsub", "vabsdiff", "vmin", "vmax"):
        for (dt, at, bt) in (("s32", "s32", "s32"), ("u32", "u32", "u32"), ("s32", "u32", "s32"), ("u32", "s32", "s32")):
            for sat in ("", "sat"):
                for op2 in ("", "add", "min", "max"):
                    cases.append((op, dt, at, bt, sat, op2, None))
    for cmp in ("eq", "ne", "lt", "le", "gt", "ge"):
        for (at, bt) in (("s32", "s32"), ("u32", "u32"), ("s32", "u32")):
            for op2 in ("", "add", "min", "max"):
                cases.append(("vset", None, at, bt, "", op2, cmp))
    asm, ref = [], []
    for (op, dt, at, bt, sat, op2, cmp) in cases:
        name = "_".join(x for x in (op, dt, at, bt, cmp, sat, op2) if x)
        opc = ".".join(x for x in (op, dt, at, bt, cmp, sat, op2) if x)
        asm.append(asm_fn(name, opc, 4 if op2 else 3))
        f = F()
        A = f.v("%s i32 %%a to i128" % ("sext" if at == "s32" else "zext"))
        B = f.v("%s i32 %%b to i128" % ("sext" if bt == "s32" else "zext"))
        if op == "vadd":
            t = f.v("add i128 %s, %s" % (A, B))
        elif op == "vsub":
            t = f.v("sub i128 %s, %s" % (A, B))
        elif op == "vabsdiff":
            d = f.v("sub i128 %s, %s" % (A, B))
            t = f.v("select i1 %s, i128 %s, i128 %s" % (f.v("icmp slt i128 %s, 0" % d), f.v("sub i128 0, %s" % d), d))
        elif op in ("vmin", "vmax"):
            t = f.v("select i1 %s, i128 %s, i128 %s" % (f.v("icmp %s i128 %s, %s" % ("slt" if op == "vmin" else "sgt", A, B)), A, B))
        else:
            pr = {"eq": "eq", "ne": "ne", "lt": "slt", "le": "sle", "gt": "sgt", "ge": "sge"}[cmp]
            t = f.v("zext i1 %s to i128" % f.v("icmp %s i128 %s, %s" % (pr, A, B)))
        if sat:
            t = clamp(f, t, *((-(1 << 31), (1 << 31) - 1) if dt == "s32" else (0, (1 << 32) - 1)))
        # optSaturate returns .s33
        t = f.v("ashr i128 %s, 95" % f.v("shl i128 %s, 95" % t))
        if op2:
            cs = "sext" if dt == "s32" else "zext"      # vset (no dtype): unsigned
            C = f.v("%s i32 %%c to i128" % cs)
            if op2 == "add":
                t = f.v("add i128 %s, %s" % (t, C))
            else:
                t = f.v("select i1 %s, i128 %s, i128 %s" % (f.v("icmp %s i128 %s, %s" % ("slt" if op2 == "min" else "sgt", t, C)), t, C))
        r = f.v("trunc i128 %s to i32" % t)
        ref.append("define i32 @%s(i32 %%a, i32 %%b, i32 %%c) {\n%s\n  ret i32 %s\n}\n" % (name, "\n".join(f.b), r))
    return asm, ref


def simdsel():
    cases = []
    for n, sels in ((2, [("h10", "h32"), ("h01", "h23"), ("h33", "h00"), ("h21", "h12"), ("h02", "h31")]),
                    (4, [("b3210", "b7654"), ("b0123", "b4567"), ("b7531", "b6420"), ("b1111", "b0707"), ("b4321", "b5432")])):
        for op in ("vadd", "vsub", "vmin", "vabsdiff", "vset"):
            for (sa, sb) in sels:
                for (at, bt) in (("s32", "s32"), ("u32", "s32")):
                    for mode in ("", "add"):
                        cases.append((op, n, sa, sb, at, bt, mode))
    asm, ref = [], []
    for k, (op, n, sa, sb, at, bt, mode) in enumerate(cases):
        name = "%s%d_sel%d" % (op, n, k)
        if op == "vset":
            opc = "vset%d.%s.%s.lt" % (n, at, bt)
        else:
            opc = "%s%d.s32.%s.%s" % (op, n, at, bt)
        opc += ".add" if mode else ""
        asm.append('define i32 @%s(i32 %%a, i32 %%b, i32 %%c) {\n  %%d = call i32 asm "%s $0, $1.%s, $2.%s, $3;", '
                   '"=r,r,r,r"(i32 %%a, i32 %%b, i32 %%c)\n  ret i32 %%d\n}\n' % (name, opc, sa, sb))
        w = 32 // n
        f = F()
        pair = f.v("or i64 %s, %s" % (f.v("zext i32 %a to i64"), f.v("shl i64 %s, 32" % f.v("zext i32 %b to i64"))))

        def elem(idx, t):
            sh = f.v("shl i64 %s, %d" % (pair, 64 - w * (idx + 1)))
            return f.v("%s i64 %s, %d" % ("ashr" if t == "s32" else "lshr", sh, 64 - w))
        res = []
        for i in range(n):
            A = elem(int(sa[len(sa) - 1 - i]), at)
            B = elem(int(sb[len(sb) - 1 - i]), bt)
            if op == "vadd":
                t = f.v("add i64 %s, %s" % (A, B))
            elif op == "vsub":
                t = f.v("sub i64 %s, %s" % (A, B))
            elif op == "vmin":
                t = f.v("select i1 %s, i64 %s, i64 %s" % (f.v("icmp slt i64 %s, %s" % (A, B)), A, B))
            elif op == "vabsdiff":
                d = f.v("sub i64 %s, %s" % (A, B))
                t = f.v("select i1 %s, i64 %s, i64 %s" % (f.v("icmp slt i64 %s, 0" % d), f.v("sub i64 0, %s" % d), d))
            else:
                t = f.v("zext i1 %s to i64" % f.v("icmp slt i64 %s, %s" % (A, B)))
            res.append(f.v("trunc i64 %s to i32" % t))
        if mode:
            acc = "%c"
            for r in res:
                acc = f.v("add i32 %s, %s" % (acc, r))
        else:
            acc = "0"
            for i, r in enumerate(res):
                acc = f.v("or i32 %s, %s" % (acc, f.v("shl i32 %s, %d" % (f.v("and i32 %s, %d" % (r, (1 << w) - 1)), w * i))))
        ref.append("define i32 @%s(i32 %%a, i32 %%b, i32 %%c) {\n%s\n  ret i32 %s\n}\n" % (name, "\n".join(f.b), acc))
    return asm, ref


HDR = "; Generated by gen_video.py from the PTX ISA pseudo-code (9.7.20).\n\n"
for nm, fn in (("video_scalar", scalar), ("video_simdsel", simdsel)):
    a, r = fn()
    open(os.path.join(HERE, nm + ".ll"), "w").write(HDR + "\n".join(a))
    open(os.path.join(HERE, nm + ".ref.ll"), "w").write(HDR + "\n".join(r))
    print(nm, len(a))
