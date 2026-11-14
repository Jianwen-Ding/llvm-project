#!/usr/bin/env python3
"""ptx-tv: lift PTX emitted by LLVM's NVPTX backend back to LLVM IR.

The lifted function is meant to be checked against the original LLVM IR
function with Alive2 (`alive-tv src.ll tgt.ll --tgt-is-asm`).

Modelling choices (all derived from the public PTX ISA description and the
open-source LLVM NVPTX backend; no vendor tools or code are used):

  * Every PTX register is an integer alloca of its declared width (.pred is
    i1).  Floating point values are bitcast in and out of integers.
  * Memory accesses go through `inttoptr` of the computed address; Alive2's
    assembly mode (--tgt-is-asm) resolves them by physical address.  Every
    access carries the natural alignment PTX requires (size of the whole
    access), so a misaligned access is UB in the lifted code.
  * Kernel/function parameters and call-site parameters are allocas of the
    size declared in PTX.  The incoming LLVM argument is stored into that
    alloca using LLVM's in-memory layout (scalar integers narrower than the
    declared slot are extended: sext for signext, zext for zeroext/i1,
    otherwise the upper bits are arbitrary).  Accesses past the declared
    size are therefore UB.
  * The address spaces of PTX share one physical address space in the
    model; cvta/cvta.to are identities.
  * fences/membar are no-ops (single-threaded semantics, as in Alive2).
  * Instructions without a rounding modifier (add.f32, mul.f32, ...) are
    modelled as round-to-nearest-even.  PTX allows ptxas to contract such
    mul/add pairs into fma; that freedom is not modelled.
  * Approximate instructions (ex2.approx, rsqrt.approx, ...) and NVVM
    intrinsics with no LLVM semantics are modelled as uninterpreted calls
    `@nvvm.<intrinsic>` (the driver renames `@llvm.nvvm.X` to `@nvvm.X` in
    the source), so only the mapping intrinsic -> instruction is checked.
"""
import os
import re
import struct
import sys

from llvmsig import parse_sig, int_bits, split_top
import layout
import narrowcvt
import mbar
import probe


class Unsupported(Exception):
    pass


# --------------------------------------------------------------------------
# PTX module parsing
# --------------------------------------------------------------------------

def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    # debug directives are not terminated by ';'
    return re.sub(r"^\s*\.(loc|file|section)\b[^\n]*$", "", text, flags=re.M)


class PParam:
    def __init__(self, decl):
        self.decl = decl
        toks = decl.replace(".param", "", 1).split()
        self.align = None
        self.ptr = False
        name = toks[-1]
        m = re.fullmatch(r"([\w$.]+)\[(\d+)\]", name)
        types = [t for t in toks[:-1] if t.startswith(".") and t[1:] in TBITS]
        ebits = TBITS[types[0][1:]] if types else 8
        if ".align" in toks:
            self.align = int(toks[toks.index(".align") + 1])
        if ".ptr" in toks:
            self.ptr = True
        if m:
            self.name = m.group(1)
            self.size = int(m.group(2)) * max(ebits // 8, 1)
        else:
            self.name = name
            self.size = max(ebits // 8, 1)
        if self.align is None or self.ptr:
            self.align = max(self.size if not m else 1, 1) if not self.ptr else 8
        self.scalar_bits = None if m else ebits
        # texture/sampler/surface reference parameters hold an opaque handle
        self.handle = any(t in toks for t in (".texref", ".samplerref", ".surfref"))
        if self.handle:
            self.size, self.align, self.scalar_bits = 8, 8, 64


class PFunc:
    pass


def parse_module(text):
    text = strip_comments(text)
    funcs = {}
    for m in re.finditer(
            r"(?:\.(?:visible|weak|extern)\s+)*\.(func|entry)\s*(\([^)]*\))?\s*"
            r"([\w$.]+)\s*(\([^)]*\))?([^{;]*)([{;])", text):
        if m.group(6) == ";":
            continue  # declaration only
        f = PFunc()
        f.kind = m.group(1)
        f.name = m.group(3)
        f.ret = [PParam(p.strip()) for p in m.group(2)[1:-1].split(",")
                 if p.strip()] if m.group(2) else []
        f.params = [PParam(p.strip()) for p in m.group(4)[1:-1].split(",")
                    if p.strip()] if m.group(4) else []
        # body
        i = m.end()
        depth = 1
        j = i
        while depth:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        f.body = text[i:j - 1]
        funcs[f.name] = f
    return funcs


def parse_body(body):
    """Return a list of ('label', name) | ('open',) | ('close',) | ('stmt', text).

    Scope braces are recognised at the start of a statement; braces inside a
    statement are vector operands.
    """
    out = []
    buf = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if not "".join(buf).strip() and c in "{}":
            out.append(("open",) if c == "{" else ("close",))
            buf = []
            i += 1
            continue
        if c == ";":
            stmt = " ".join("".join(buf).split())
            if stmt:
                out.append(("stmt", stmt))
            buf = []
            i += 1
            continue
        if c == ":" and (i + 1 >= n or body[i + 1] != ":") and \
                (i == 0 or body[i - 1] != ":"):
            cand = "".join(buf).strip()
            if re.fullmatch(r"[$\w.]+", cand):
                out.append(("label", cand))
                buf = []
                i += 1
                continue
        buf.append(c)
        i += 1
    if "".join(buf).strip():
        raise Unsupported("trailing text: " + "".join(buf).strip()[:40])
    return out


def decode(stmt):
    """Split a statement into (pred, opcode, ops)."""
    pred = None
    m = re.match(r"@(!?)(%\w+)\s+", stmt)
    if m:
        pred = (m.group(1) == "!", m.group(2))
        stmt = stmt[m.end():]
    m = re.match(r"([a-z_0-9]+(?:\.[\w:]+)*)\s*", stmt)
    if not m:
        raise Unsupported("stmt " + stmt[:60])
    rest = stmt[m.end():]
    ops = split_top(rest) if rest.strip() else []
    return pred, m.group(1), ops


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

TBITS = {"pred": 1, "b8": 8, "u8": 8, "s8": 8, "b16": 16, "u16": 16,
         "s16": 16, "b32": 32, "u32": 32, "s32": 32, "b64": 64, "u64": 64,
         "s64": 64, "b128": 128, "f16": 16, "bf16": 16, "f16x2": 32,
         "bf16x2": 32, "f32": 32, "f64": 64, "f32x2": 64, "s16x2": 32,
         "u16x2": 32, "tf32": 32}
FPTY = {"f16": "half", "bf16": "bfloat", "f32": "float", "f64": "double",
        "f16x2": "<2 x half>", "bf16x2": "<2 x bfloat>",
        "f32x2": "<2 x float>"}
FP_SCALAR = {"half": "f16", "bfloat": "bf16", "float": "f32",
             "double": "f64"}
MODES = {"rn": "round.tonearest", "rz": "round.towardzero",
         "rm": "round.downward", "rp": "round.upward"}


def is_signed(t):
    return t.startswith("s")


def is_float(t):
    return t in FPTY


def elem_fty(t):
    f = FPTY[t]
    return f.split(" x ")[1][:-1] if f.startswith("<") else f


def intname(f):
    """LLVM intrinsic type suffix for an FP type."""
    if f.startswith("<"):
        n, e = f[1:-1].split(" x ")
        return "v%s%s" % (n, FP_SCALAR[e])
    return FP_SCALAR[f]


SMALLEST_NORMAL = {"float": "0x3810000000000000",
                   "double": "0x0010000000000000"}


def fp_const(fty, value):
    """LLVM hex literal for a float constant of type fty."""
    if fty in ("half", "bfloat"):
        raw = struct.unpack("<H", struct.pack("<e", value))[0] \
            if fty == "half" else (struct.unpack(
                "<I", struct.pack("<f", value))[0] >> 16)
        return ("0xH%04X" if fty == "half" else "0xR%04X") % raw
    raw = struct.unpack("<Q", struct.pack("<d", value))[0]
    return "0x%016X" % raw


def splat(fty, scalar):
    if fty.startswith("<"):
        n, e = fty[1:-1].split(" x ")
        return "<" + ", ".join("%s %s" % (e, scalar) for _ in range(int(n))) + ">"
    return scalar


def fconst(fty, value):
    e = fty.split(" x ")[1][:-1] if fty.startswith("<") else fty
    return splat(fty, fp_const(e, value))


# --------------------------------------------------------------------------
# Lifter
# --------------------------------------------------------------------------

SREG_RE = re.compile(r"%(tid|ntid|ctaid|nctaid|clusterid|nclusterid|"
                     r"cluster_ctaid|cluster_nctaid)\.([xyzw])$")
SREG_PLAIN = {"laneid", "warpid", "nwarpid", "smid", "nsmid", "gridid",
              "lanemask_eq", "lanemask_le", "lanemask_lt", "lanemask_ge",
              "lanemask_gt", "clock", "clock64", "globaltimer",
              "globaltimer_lo", "globaltimer_hi", "warpsize",
              "cluster_ctarank", "cluster_nctarank", "is_explicit_cluster",
              "dynamic_smem_size", "total_smem_size", "aggr_smem_size",
              "current_graph_exec", "reserved_smem_offset_begin",
              "reserved_smem_offset_end", "reserved_smem_offset_cap",
              "reserved_smem_offset_0", "reserved_smem_offset_1",
              "pm0", "pm1", "pm2", "pm3", "pm4",
              "pm5", "pm6", "pm7"}
SREG_PURE = {"tid", "ntid", "ctaid", "nctaid", "laneid", "warpsize",
             "nclusterid", "clusterid", "cluster_ctaid", "cluster_nctaid",
             "cluster_ctarank", "cluster_nctarank", "lanemask_eq",
             "lanemask_le", "lanemask_lt", "lanemask_ge", "lanemask_gt",
             "is_explicit_cluster", "nsmid", "nwarpid"}


# instructions whose first operand is not a destination register
NO_DEF_OPS = {"st", "red", "call", "bra", "brx", "ret", "exit", "bar",
              "barrier", "membar", "fence", "prefetch", "prefetchu", "trap",
              "nanosleep", "stackrestore", "cp", "applypriority", "pmevent"}


class Lifter:
    def __init__(self, pfunc, sig, name_map, decls, globals_, typedefs=None,
                 indirect_sigs=None):
        self.typedefs = typedefs or {}
        self.indirect_sigs = indirect_sigs or []
        self.used_probe = False
        self.handle_params = {}      # .texref/.samplerref param -> LLVM arg
        self.pf = pfunc
        self.sig = sig
        self.name_map = name_map     # PTX symbol -> LLVM global/function name
        self.decls = decls           # LLVM fn name -> Sig (callees)
        self.globals = globals_      # LLVM global name -> type text (ptr ty)
        self.n = 0
        self.regs = {}               # ptx reg -> bits
        self.regslot = {}            # ptx reg -> alloca
        self.reg_shadow = []         # per scope: shadowed registers
        self.slots_used = set()
        self.base_of = {}            # 64-bit reg -> base object key
        self.exact = set()           # regs equal to their base address
        self.const_off = {}          # regs equal to base + constant
        self.slot_base = {}          # param slot -> base key of stored ptr
        self.jump_tables = {}        # label -> [target labels]
        self.last_label = None
        self.allocas = []
        self.blocks = []             # [label, lines]
        self.cur = None
        self.scopes = [{}]           # symbol -> (alloca name, PParam)
        self.extra_decls = {}        # name -> declare line
        self.depot = {}
        self.cc = None
        self.callees = []

    # ---- emission helpers ------------------------------------------------
    def tmp(self):
        self.n += 1
        return "%%t%d" % self.n

    def emit(self, s):
        self.cur[1].append("  " + s)

    def op(self, s):
        t = self.tmp()
        self.emit("%s = %s" % (t, s))
        return t

    def new_block(self, label):
        self.cur = [label, []]
        self.blocks.append(self.cur)

    def terminated(self):
        if not self.cur[1]:
            return False
        last = self.cur[1][-1].strip()
        return last.startswith(("br ", "ret", "unreachable", "switch "))

    def declare(self, name, line):
        self.extra_decls[name] = line

    def call_intr(self, ret, name, args, decl_args=None):
        if decl_args is None:
            from llvmsig import parse_type
            decl_args = [parse_type(a)[0] for a in args]
        self.declare(name, "declare %s @%s(%s)" % (ret, name,
                                                     ", ".join(decl_args)))
        if ret == "void":
            self.emit("call void @%s(%s)" % (name, ", ".join(args)))
            return None
        return self.op("call %s @%s(%s)" % (ret, name, ", ".join(args)))

    # ---- registers -------------------------------------------------------
    def reg_slot(self, r):
        return self.regslot[r]

    def declare_regs(self, stmt):
        m = re.fullmatch(r"\.reg\s+\.(\w+)\s+(.+)", stmt)
        if not m:
            raise Unsupported("reg decl " + stmt)
        bits = TBITS.get(m.group(1))
        if bits is None:
            raise Unsupported("reg type " + m.group(1))
        for item in m.group(2).split(","):
            item = item.strip()
            mm = re.fullmatch(r"(%[\w$]+)<(\d+)>", item)
            names = ["%s%d" % (mm.group(1), k) for k in range(int(mm.group(2)))] \
                if mm else [item]
            for nm in names:
                slot = "%%reg.%s" % re.sub(r"[^\w.]", "_", nm.lstrip("%"))
                if slot in self.slots_used:
                    slot += ".%d" % self.n
                    self.n += 1
                self.slots_used.add(slot)
                if self.reg_shadow:
                    self.reg_shadow[-1].append(
                        (nm, (self.regs[nm], self.regslot[nm]) if nm in self.regs else None))
                self.regs[nm] = bits
                self.regslot[nm] = slot
                self.allocas.append("%s = alloca i%d" % (slot, bits))

    def read_reg(self, r, bits):
        if r not in self.regs:
            raise Unsupported("unknown register " + r)
        w = self.regs[r]
        v = self.op("load i%d, ptr %s" % (w, self.reg_slot(r)))
        return self.resize(v, w, bits, False)

    def resize(self, v, w, bits, signed):
        if w == bits:
            return v
        if w > bits:
            return self.op("trunc i%d %s to i%d" % (w, v, bits))
        return self.op("%s i%d %s to i%d" % ("sext" if signed else "zext", w, v, bits))

    def write(self, dst, v, bits, signed=False):
        if dst == "_":
            return
        if dst not in self.regs:
            raise Unsupported("bad destination " + dst)
        w = self.regs[dst]
        v = self.resize(v, bits, w, signed)
        self.emit("store i%d %s, ptr %s" % (w, v, self.reg_slot(dst)))

    # ---- operands --------------------------------------------------------
    def imm(self, s):
        s = s.strip()
        m = re.fullmatch(r"0[fF]([0-9a-fA-F]{8})", s)
        if m:
            return int(m.group(1), 16)
        m = re.fullmatch(r"0[dD]([0-9a-fA-F]{16})", s)
        if m:
            return int(m.group(1), 16)
        m = re.fullmatch(r"(-?)0[xX]([0-9a-fA-F]+)U?", s)
        if m:
            v = int(m.group(2), 16)
            return -v if m.group(1) else v
        m = re.fullmatch(r"(-?\d+)U?", s)
        if m:
            return int(m.group(1))
        return None

    def const(self, v, bits):
        v &= (1 << bits) - 1
        if bits == 1:
            return "true" if v else "false"
        if v >= 1 << (bits - 1):
            v -= 1 << bits
        return str(v)

    def check_entry_retval(self, sym):
        if sym == "func_retval0" and self.pf.kind == "entry" and not self.pf.ret:
            raise Unsupported("kernel stores to undeclared func_retval0 (finding 5)")

    def symbol_addr(self, sym):
        """i64 address of a PTX symbol (param, global, function, depot)."""
        for sc in reversed(self.scopes):
            if sym in sc:
                return self.op("ptrtoint ptr %s to i64" % sc[sym][0])
        if sym in self.depot:
            return self.op("ptrtoint ptr %s to i64" % self.depot[sym])
        if sym in self.name_map:
            g = self.name_map[sym]
            ty = self.globals.get(g, "ptr")
            return self.op("ptrtoint %s @%s to i64" % (ty, g))
        raise Unsupported("unknown symbol " + sym)

    def get(self, s, bits):
        s = s.strip()
        if s in self.regs:
            return self.read_reg(s, bits)
        if s.startswith("%"):
            m = SREG_RE.fullmatch(s)
            if m or s[1:] in SREG_PLAIN:
                return self.sreg(s, bits)
            if s.startswith("%envreg"):
                return self.sreg(s, bits)
            return self.read_reg(s, bits)
        v = self.imm(s)
        if v is not None:
            return self.const(v, bits)
        if s == "WARP_SZ":          # PTX constant: the warp size, 32
            return self.const(32, bits)
        m = re.fullmatch(r"([\w$.]+)(?:\+(-?\d+))?", s)
        if m:
            a = self.symbol_addr(m.group(1))
            if m.group(2):
                a = self.op("add i64 %s, %s" % (a, m.group(2)))
            return self.resize(a, 64, bits, False)
        raise Unsupported("operand " + s)

    def sreg(self, s, bits):
        if s == "%is_explicit_cluster":
            self.declare("nvvm.is_explicit_cluster",
                         "declare i1 @nvvm.is_explicit_cluster() nounwind willreturn memory(none)")
            v = self.op("call i1 @nvvm.is_explicit_cluster()")
            return self.resize(v, 1, bits, False)
        m = SREG_RE.fullmatch(s)
        if m:
            base = m.group(1)
            nm = "%s.%s" % (m.group(1), m.group(2))
        else:
            base = nm = s[1:]
        # TableGen intrinsic naming turns _ into . (lanemask_ge ->
        # lanemask.ge) unless the intrinsic overrides its name
        if not (nm.endswith("_smem_size") or nm.startswith("reserved_smem_offset")):
            nm = nm.replace("_", ".")
        ty = 64 if nm in ("clock64", "globaltimer") else 32  # gridid: i32 in LLVM
        fname = "nvvm.read.ptx.sreg." + nm
        attrs = "memory(none)" if base in SREG_PURE else \
            "memory(inaccessiblemem: readwrite)"
        self.declare(fname, "declare i%d @%s() nounwind willreturn %s" %
                     (ty, fname, attrs))
        v = self.op("call i%d @%s()" % (ty, fname))
        return self.resize(v, ty, bits, False)

    def getf(self, s, t):
        """Operand as FP value of PTX type t."""
        fty = FPTY[t]
        v = self.get(s, TBITS[t])
        return self.op("bitcast i%d %s to %s" % (TBITS[t], v, fty))

    def writef(self, dst, v, t):
        bits = TBITS[t]
        iv = self.op("bitcast %s %s to i%d" % (FPTY[t], v, bits))
        self.write(dst, iv, bits)

    # ---- address computation ----------------------------------------------
    def address(self, s, space):
        """Return a typed pointer operand ("ptr %x") for the [addr] operand."""
        s = s.strip()
        if not (s.startswith("[") and s.endswith("]")):
            raise Unsupported("address " + s)
        s = s[1:-1].strip()
        if re.fullmatch(r"(-?\d+|0x[0-9a-fA-F]+)", s):      # absolute address
            return "ptr " + self.op("inttoptr i64 %d to ptr" % self.imm(s))
        m = re.fullmatch(r"([%\w$.]+)\s*(?:\+\s*(-?\d+|0x[0-9a-fA-F]+))?", s)
        if not m:
            raise Unsupported("address " + s)
        base, off = m.group(1), m.group(2)
        off = self.imm(off) if off else 0
        if not base.startswith("%") and base not in self.regs:
            ty, p = self.sym_ptr(base)
            if off:
                p = self.op("getelementptr i8, %s %s, i64 %d" % (ty, p, off))
            return "%s %s" % (ty, p)
        a = self.get(base, 64) if self.regs.get(base, 64) == 64 else \
            self.resize(self.get(base, 32), 32, 64, False)
        if off:
            a = self.op("add i64 %s, %d" % (a, off))
        key = self.base_of.get(base)
        if key is not None and base in self.const_off:
            b = self.base_ptr(key)
            if b is not None:
                total = self.const_off[base] + off
                if total == 0:
                    return "%s %s" % b
                return "%s %s" % (b[0], self.op("getelementptr i8, %s %s, i64 %d" % (b[0], b[1], total)))
        if key is not None:
            if base in self.exact and not off:
                b = self.base_ptr(key)
                if b is not None:
                    return "%s %s" % b
            r = self.rebase(key, a)
            if r is not None:
                return r
        return "ptr " + self.op("inttoptr i64 %s to ptr" % a)

    def sym_ptr(self, sym):
        """(pointer type, pointer value) of a PTX symbol."""
        self.check_entry_retval(sym)
        for sc in reversed(self.scopes):
            if sym in sc:
                return "ptr", sc[sym][0]
        if sym in self.depot:
            return "ptr", self.depot[sym]
        if sym in self.name_map:
            g = self.name_map[sym]
            return self.globals.get(g, "ptr"), "@" + g
        raise Unsupported("unknown symbol " + sym)

    def base_ptr(self, key):
        kind, v = key
        if kind == "arg":
            a = self.sized_args(self.sig.args)[v]
            return a.ty, a.name
        if kind == "dyn":
            return "ptr", self.op("load ptr, ptr %s" % self.dyn_slot(v))
        try:
            return self.sym_ptr(v)
        except Unsupported:
            return None

    def rebase(self, key, addr):
        """Pointer with the provenance of `key` for i64 address `addr`."""
        b = self.base_ptr(key)
        if b is None:
            return None
        ty, p = b
        bi = self.op("ptrtoint %s %s to i64" % (ty, p))
        d = self.op("sub i64 %s, %s" % (addr, bi))
        return "%s %s" % (ty, self.op("getelementptr i8, %s %s, i64 %s" % (ty, p, d)))

    def analyze_bases(self, items):
        """Flow-insensitive: which object does each 64-bit register point into?

        A register derives from a base if every definition is a move, cvta,
        or add/sub of a constant/non-pointer offset applied to that base (a
        symbol, or a pointer argument read with ld.param).
        """
        ptr_args = {}
        for k, (pp, arg) in enumerate(zip(self.pf.params, self.sized_args(self.sig.args))):
            if arg.ty.startswith("ptr") and not arg.byval_type():
                ptr_args[pp.name] = k
        widths = {}
        for it in items:
            if it[0] == "stmt" and it[1].startswith(".reg"):
                m = re.fullmatch(r"\.reg\s+\.(\w+)\s+(.+)", it[1])
                if m and TBITS.get(m.group(1)) == 64:
                    for item in m.group(2).split(","):
                        item = item.strip()
                        mm = re.fullmatch(r"(%[\w$]+)<(\d+)>", item)
                        for nm in (["%s%d" % (mm.group(1), k) for k in range(int(mm.group(2)))]
                                   if mm else [item]):
                            widths[nm] = 64
        defs = []
        for it in items:
            if it[0] != "stmt" or it[1].startswith("."):
                continue
            try:
                pred, opc, ops = decode(it[1])
            except Unsupported:
                continue
            if not ops:
                continue
            d = ops[0].strip()
            parts = opc.split(".")
            if d not in widths or parts[0] in NO_DEF_OPS:
                continue
            defs.append((parts[0], parts[1:], d, ops[1:]))
        # registers whose value is used as an address (or call target),
        # directly or through add/sub/mov/cvta/and
        addr_used = set()
        for it in items:
            if it[0] != "stmt" or it[1].startswith("."):
                continue
            for m in re.finditer(r"\[\s*(%[\w$]+)", it[1]):
                addr_used.add(m.group(1))
            mc = re.match(r"(?:@!?%\w+\s+)?call[\w.]*\s+(?:\([^)]*\)\s*,\s*)?(%[\w$]+)", it[1])
            if mc:
                addr_used.add(mc.group(1))
        changed = True
        while changed:
            changed = False
            for b, mods, d, srcs in defs:
                if d in addr_used and b in ("add", "sub", "mov", "cvta", "and"):
                    for x in srcs:
                        x = x.strip()
                        if x in widths and x not in addr_used:
                            addr_used.add(x)
                            changed = True
        base = {r: set() for r in widths}
        NONE = ("none", None)

        def ob(x):
            x = x.strip()
            if x in widths:
                return base[x]
            if self.imm(x) is not None:
                return {NONE}
            if re.fullmatch(r"[\w$.]+", x):
                return {("sym", x)}
            return {NONE}
        changed = True
        while changed:
            changed = False
            for b, mods, d, srcs in defs:
                if b in ("mov", "cvta") and len(srcs) == 1 and not srcs[0].startswith("{"):
                    new = ob(srcs[0])
                elif b == "add" and len(srcs) == 2 and "cc" not in mods:
                    A, B = ob(srcs[0]), ob(srcs[1])
                    if A == {NONE}:
                        new = B
                    elif B == {NONE}:
                        new = A
                    else:
                        new = {NONE}
                elif b == "sub" and len(srcs) == 2 and "cc" not in mods and ob(srcs[1]) == {NONE}:
                    new = ob(srcs[0])
                elif b == "and" and len(srcs) == 2 and (self.imm(srcs[1]) or 0) < 0:
                    # clearing low address bits (alignment) stays in the object
                    new = ob(srcs[0])
                elif b == "alloca":
                    new = {("dyn", d)}
                elif b in ("ld", "ldu") and "param" not in mods and len(srcs) == 1 \
                        and d in addr_used \
                        and any(m in ("b64", "u64", "s64") for m in mods) \
                        and not any(m in ("v2", "v4", "v8") for m in mods):
                    # a loaded 64-bit value may be a pointer: keep its provenance
                    new = {("dyn", d)}
                elif b == "ld" and "param" in mods and len(srcs) == 1:
                    m = re.fullmatch(r"\[\s*([\w$]+)\s*\]", srcs[0].strip())
                    new = {("arg", ptr_args[m.group(1)])} if m and m.group(1) in ptr_args else {NONE}
                else:
                    new = {NONE}
                if not new <= base[d]:
                    base[d] |= new
                    changed = True
        self.base_of = {r: next(iter(v)) for r, v in base.items()
                        if len(v) == 1 and NONE not in v}
        # registers that always hold exactly the base address (offset 0)
        exact = {r for r in self.base_of}
        changed = True
        while changed:
            changed = False
            for b, mods, d, srcs in defs:
                if d not in exact:
                    continue
                ok = b in ("ld", "ldu") or b == "alloca" or \
                    (b in ("mov", "cvta") and len(srcs) == 1 and
                     (srcs[0].strip() in exact or
                      (srcs[0].strip() not in widths and self.imm(srcs[0]) is None)))
                if not ok:
                    exact.discard(d)
                    changed = True
        self.exact = exact
        # registers holding base + a constant: only defined by an exact base
        # register (offset 0) or add/sub of an immediate to such a register
        defs_of = {}
        for b, mods, d, srcs in defs:
            defs_of.setdefault(d, []).append((b, mods, srcs))
        offs = {r: 0 for r in exact}
        changed = True
        while changed:
            changed = False
            for d, dl in defs_of.items():
                if d in offs or d not in self.base_of or len(dl) != 1:
                    continue
                b, mods, srcs = dl[0]
                if b in ("add", "sub") and len(srcs) == 2 and "cc" not in mods:
                    a, c = srcs[0].strip(), srcs[1].strip()
                    ia, ic = self.imm(a), self.imm(c)
                    if a in offs and ic is not None:
                        offs[d] = offs[a] + (ic if b == "add" else -ic)
                        changed = True
                    elif b == "add" and c in offs and ia is not None:
                        offs[d] = offs[c] + ia
                        changed = True
                elif b in ("mov", "cvta") and len(srcs) == 1 and srcs[0].strip() in offs:
                    offs[d] = offs[srcs[0].strip()]
                    changed = True
        self.const_off = offs

    # ---- function body -----------------------------------------------------
    def lift(self):
        sig = self.sig
        pf = self.pf
        self.new_block("entry")
        entry = self.cur
        # parameters
        if sig.vararg:
            raise Unsupported("vararg")
        args = self.sized_args(sig.args)
        if len(pf.params) != len(args):
            raise Unsupported("param count mismatch %d vs %d" %
                              (len(pf.params), len(args)))
        for k, (pp, arg) in enumerate(zip(pf.params, args)):
            if arg.byval_type() and re.search(
                    r"(?<![\[\w$])%s(?![\w$])(?!\s*\+?\s*\d*\s*\])" % re.escape(pp.name), pf.body):
                # the address of the .param is taken: it *is* the callee's
                # byval copy, so keep the source's object identity
                self.scopes[0][pp.name] = (arg.name, pp)
                continue
            slot = "%%param.%d" % k
            if pp.handle:
                self.handle_params[pp.name] = "%s %s" % (arg.ty, arg.name)
            self.allocas.append("%s = alloca [%d x i8], align %d" %
                                (slot, pp.size, pp.align))
            self.scopes[0][pp.name] = (slot, pp)
            self.store_value(arg.ty, arg.name, arg, slot, pp, is_ret=False)
        if pf.ret:
            if len(pf.ret) != 1:
                raise Unsupported("multiple return params")
            rp = pf.ret[0]
            self.allocas.append("%%retval = alloca [%d x i8], align %d" %
                                (rp.size, rp.align))
            self.scopes[0][rp.name] = ("%retval", rp)
            self.retparam = rp
        elif sig.ret != "void" and not self.zero_sized(sig.ret) and pf.kind != "entry":
            raise Unsupported("non-void function without retval param")
        self.emit("br label %start")
        self.new_block("start")
        items = parse_body(pf.body)
        self.analyze_bases(items)
        for item in items:
            kind = item[0]
            if kind == "label":
                self.last_label = item[1]
                lab = self.label(item[1])
                if not self.terminated():
                    self.emit("br label %" + lab)
                self.new_block(lab)
            elif kind == "open":
                self.scopes.append({})
                self.reg_shadow.append([])
            elif kind == "close":
                self.scopes.pop()
                for nm, old in reversed(self.reg_shadow.pop()):
                    if old is None:
                        del self.regs[nm]
                        del self.regslot[nm]
                    else:
                        self.regs[nm], self.regslot[nm] = old
            else:
                if self.terminated():
                    self.new_block("dead%d" % self.n)
                    self.n += 1
                self.stmt(item[1])
        if not self.terminated():
            self.emit("unreachable")
        entry[1] = ["  " + a for a in self.allocas] + entry[1]

    def zero_sized(self, ty):
        if ty.startswith("ptr") or ty == "void":
            return False
        try:
            return layout.alloc_size(layout.parse(ty, self.typedefs)) == 0
        except ValueError:
            return False

    def sized_args(self, args):
        """Arguments that get a PTX parameter (zero-sized types get none)."""
        out = []
        for a in args:
            if not a.ty.startswith("ptr"):
                try:
                    if layout.alloc_size(layout.parse(a.ty, self.typedefs)) == 0:
                        continue
                except ValueError:
                    pass
            out.append(a)
        return out

    def label(self, l):
        return "L." + l.replace("$", "_")

    def store_value(self, ty, val, arg, slot, pp, is_ret):
        """Store LLVM value `val` of type `ty` into a PTX param slot."""
        if ty.startswith("ptr") and arg is not None and arg.byval_type():
            self.emit("call void @llvm.memcpy.p0.p0.i64(ptr %s, ptr %s, i64 %d, i1 false)"
                      % (slot, val, pp.size))
            self.declare("llvm.memcpy.p0.p0.i64",
                         "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)")
            return
        if ty == "ptr" and pp.size < 8:
            # 32-bit pointers (e.g. tensor memory, p6:32:32) flattened to ptr
            iv = self.op("ptrtoint ptr %s to i64" % val)
            iv = self.op("trunc i64 %s to i%d" % (iv, pp.size * 8))
            self.emit("store i%d %s, ptr %s" % (pp.size * 8, iv, slot))
            return
        b = int_bits(ty)
        if b is not None and pp.scalar_bits and pp.scalar_bits > b:
            w = pp.scalar_bits
            signext = arg is not None and arg.has("signext")
            zeroext = arg is not None and arg.has("zeroext")
            if signext:
                v = self.op("sext %s %s to i%d" % (ty, val, w))
            elif zeroext or b == 1:
                v = self.op("zext %s %s to i%d" % (ty, val, w))
            else:
                z = self.op("zext %s %s to i%d" % (ty, val, w))
                g = self.op("freeze i%d poison" % w)
                hi = self.op("and i%d %s, %s" % (w, g, self.const(~((1 << b) - 1), w)))
                v = self.op("or i%d %s, %s" % (w, z, hi))
            self.emit("store i%d %s, ptr %s" % (w, v, slot))
            return
        self.store_mem(layout.parse(ty, self.typedefs), val, slot, 0)

    def load_value(self, ty, slot, pp):
        if ty == "ptr" and pp.size < 8:
            v = self.op("load i%d, ptr %s" % (pp.size * 8, slot))
            v = self.op("zext i%d %s to i64" % (pp.size * 8, v))
            return self.op("inttoptr i64 %s to ptr" % v)
        b = int_bits(ty)
        if b is not None and pp.scalar_bits and pp.scalar_bits > b:
            v = self.op("load i%d, ptr %s" % (pp.scalar_bits, slot))
            return self.op("trunc i%d %s to %s" % (pp.scalar_bits, v, ty))
        return self.load_mem(layout.parse(ty, self.typedefs), slot, 0)

    # LLVM values are moved through PTX parameter memory field by field,
    # following the NVPTX data layout.  Sub-byte scalars and vectors of
    # sub-byte elements are stored as zero-extended whole bytes.
    def gep(self, base, off):
        return base if off == 0 else self.op("getelementptr i8, ptr %s, i64 %d" % (base, off))

    def store_mem(self, t, val, base, off):
        if t.kind in ("struct", "arr") and layout.has_sub_byte(t):
            n = len(t.elems) if t.kind == "struct" else t.n
            offs = layout.offsets(t) if t.kind == "struct" else \
                [k * layout.alloc_size(t.elem) for k in range(n)]
            for k in range(n):
                e = t.elems[k] if t.kind == "struct" else t.elem
                x = self.op("extractvalue %s %s, %d" % (t.text, val, k))
                self.store_mem(e, x, base, off + offs[k])
            return
        p = self.gep(base, off)
        if layout.is_sub_byte(t):
            nb = layout.bits(t)
            sb = layout.store_size(t) * 8
            x = self.op("bitcast %s %s to i%d" % (t.text, val, nb)) if t.kind == "vec" else val
            if sb != nb:
                x = self.op("zext i%d %s to i%d" % (nb, x, sb))
            self.emit("store i%d %s, ptr %s, align 1" % (sb, x, p))
            return
        self.emit("store %s %s, ptr %s, align 1" % (t.text, val, p))

    def load_mem(self, t, base, off):
        if t.kind in ("struct", "arr") and layout.has_sub_byte(t):
            n = len(t.elems) if t.kind == "struct" else t.n
            offs = layout.offsets(t) if t.kind == "struct" else \
                [k * layout.alloc_size(t.elem) for k in range(n)]
            acc = "poison"
            for k in range(n):
                e = t.elems[k] if t.kind == "struct" else t.elem
                x = self.load_mem(e, base, off + offs[k])
                acc = self.op("insertvalue %s %s, %s %s, %d" % (t.text, acc, e.text, x, k))
            return acc
        p = self.gep(base, off)
        if layout.is_sub_byte(t):
            nb = layout.bits(t)
            sb = layout.store_size(t) * 8
            x = self.op("load i%d, ptr %s, align 1" % (sb, p))
            if sb != nb:
                x = self.op("trunc i%d %s to i%d" % (sb, x, nb))
            if t.kind == "vec":
                x = self.op("bitcast i%d %s to %s" % (nb, x, t.text))
            return x
        return self.op("load %s, ptr %s, align 1" % (t.text, p))

    # ---- statements --------------------------------------------------------
    def stmt(self, s):
        if s.startswith(".reg"):
            return self.declare_regs(s)
        if s.startswith(".local"):
            m = re.fullmatch(r"\.local\s+\.align\s+(\d+)\s+\.b8\s+([\w$]+)\[(\d+)\]", s)
            if not m:
                raise Unsupported("local decl " + s)
            name = "%%depot.%d" % len(self.depot)
            self.allocas.append("%s = alloca [%s x i8], align %s" %
                                (name, m.group(3), m.group(1)))
            self.depot[m.group(2)] = name
            return
        if s.startswith(".param"):
            pp = PParam(s)
            slot = "%%cs.%d" % self.n
            self.n += 1
            self.allocas.append("%s = alloca [%d x i8], align %d" %
                                (slot, pp.size, pp.align))
            self.scopes[-1][pp.name] = (slot, pp)
            return
        if s.startswith(".shared"):
            # a module-scope shared global the backend demoted into the function
            m = re.search(r"([\w$]+)(\[\d+\])?\s*$", s)
            if m and m.group(1) in self.name_map:
                return
            raise Unsupported("function-scope .shared")
        if s.startswith(".callprototype"):
            return  # the LLVM type of indirect calls comes from the source
        if s.startswith(".branchtargets"):
            self.jump_tables[self.last_label] = [x.strip() for x in s[len(".branchtargets"):].split(",")]
            return
        if s.startswith(".pragma") or s.startswith(".loc") or \
                s.startswith(".file") or s.startswith(".maxnreg"):
            return
        if s.startswith("."):
            raise Unsupported("directive " + s.split()[0])
        pred, opcode, ops = decode(s)
        parts = opcode.split(".")
        base = parts[0]
        if base == "bra":
            tgt = self.label(ops[0])
            if pred:
                c = self.read_reg(pred[1], 1)
                nxt = "fall%d" % self.n
                self.n += 1
                if pred[0]:
                    self.emit("br i1 %s, label %%%s, label %%%s" % (c, nxt, tgt))
                else:
                    self.emit("br i1 %s, label %%%s, label %%%s" % (c, tgt, nxt))
                self.new_block(nxt)
            else:
                self.emit("br label %" + tgt)
            return
        if pred:
            c = self.read_reg(pred[1], 1)
            then = "pthen%d" % self.n
            cont = "pcont%d" % self.n
            self.n += 1
            if pred[0]:
                self.emit("br i1 %s, label %%%s, label %%%s" % (c, cont, then))
            else:
                self.emit("br i1 %s, label %%%s, label %%%s" % (c, then, cont))
            self.new_block(then)
            self.instr(base, parts[1:], ops, opcode)
            if not self.terminated():
                self.emit("br label %" + cont)
            self.new_block(cont)
            return
        self.instr(base, parts[1:], ops, opcode)

    # Every modifier the lifter gives meaning to (or that is semantically
    # irrelevant under its model).  Anything else makes the instruction
    # unsupported rather than silently ignored.
    KNOWN_MODS = set(TBITS) | {
        "rn", "rz", "rm", "rp", "rni", "rzi", "rmi", "rpi", "ftz", "sat",
        "relu", "approx", "full", "NaN", "abs", "lo", "hi", "wide", "cc",
        "global", "shared", "local", "const", "param", "generic",
        "shared::cta", "shared::cluster", "param::func", "param::entry",
        "v2", "v4", "v8", "volatile", "relaxed", "acquire", "release",
        "acq_rel", "sc", "weak", "cta", "gpu", "sys", "cluster", "gl",
        "ca", "cg", "cs", "lu", "cv", "wb", "wt", "nc",
        "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hs", "equ", "neu",
        "ltu", "leu", "gtu", "geu", "num", "nan", "and", "or", "xor",
        "add", "min", "max", "inc", "dec", "exch", "cas", "noftz",
        "l", "r", "wrap", "clamp", "f4e", "b4e", "rc8", "ecl", "ecr", "rc16",
        "shiftamt", "uni", "sync", "aligned", "warp", "arrive", "up", "down",
        "bfly", "idx", "all", "any", "ballot", "to", "finite", "infinite",
        "number", "notanumber", "normal", "subnormal", "u32", "s32",
        "u64", "satfinite", "xorsign", "oob", "mmio", "red", "popc", "inc", "dec", "m8n8", "trans",
    }

    def instr(self, base, mods, ops, opcode):
        if base == "cvt" and narrowcvt.is_narrow(mods, mods):
            return self.cvt_narrow(opcode, ops)
        if base == "mbarrier":
            return self.mbarrier(opcode, ops)
        if probe.PTX_FAMILIES.match(base):
            name = probe.op_name(opcode, ops)
            if name in self.decls:
                return self.generic_op(name, ops)
        h = getattr(self, "i_" + base, None)
        if h is None:
            raise Unsupported("opcode " + opcode)
        if base in ("fence", "membar", "prefetch", "prefetchu", "cp", "discard",
                    "spdecompress", "multimem") or \
                (base == "wgmma" and not set(mods) - {"fence", "commit_group", "wait_group",
                                                     "sync", "aligned"}) or \
                (base == "st" and ("bulk" in mods or "async" in mods)) or \
                (base == "tcgen05" and mods[:1] in (["ld"], ["st"])) or \
                (base == "tcgen05" and mods[:1] == ["mma"] and "decompress::lut::b" not in mods) or \
                (base == "tcgen05" and mods[:1] in (["alloc"], ["dealloc"], ["relinquish_alloc_permit"],
                                                   ["commit"], ["cp"], ["shift"], ["wait::ld"],
                                                   ["wait::st"], ["fence::before_thread_sync"],
                                                   ["fence::after_thread_sync"])):
            return h(mods, ops, opcode)   # check their own modifiers
        for m in mods:
            if m not in self.KNOWN_MODS and not m.startswith(("L1::", "L2::")):
                raise Unsupported("modifier .%s in %s" % (m, opcode))
        h(mods, ops, opcode)

    # helpers to classify modifiers
    @staticmethod
    def types_of(mods):
        return [m for m in mods if m in TBITS]

    # ---- control -----------------------------------------------------------
    def ptr_from_slot(self, ty, slot):
        """Read a pointer stored as an integer in a param slot."""
        v = self.op("load i64, ptr %s, align 1" % slot)
        key = self.slot_base.get(slot)
        if key is not None:
            r = self.rebase(key, v)
            if r is not None and r.rsplit(" ", 1)[0] == ty:
                return r.rsplit(" ", 1)[1]
        return self.op("inttoptr i64 %s to %s" % (v, ty))

    def i_ret(self, mods, ops, opc):
        if self.sig.ret == "void":
            self.emit("ret void")
        elif self.zero_sized(self.sig.ret):
            self.emit("ret %s zeroinitializer" % self.sig.ret)
        elif self.pf.kind == "entry":
            # a kernel's return value is not observable (.entry has no retval)
            self.emit("ret %s poison" % self.sig.ret)
        elif self.sig.ret.startswith("ptr"):
            v = self.ptr_from_slot(self.sig.ret, "%retval")
            self.emit("ret %s %s" % (self.sig.ret, v))
        else:
            v = self.load_value(self.sig.ret, "%retval", self.retparam)
            self.emit("ret %s %s" % (self.sig.ret, v))

    def i_exit(self, mods, ops, opc):
        if self.pf.kind == "entry" and self.sig.ret == "void":
            self.emit("ret void")
            return
        # thread exit: never returns to the caller
        self.declare("nvvm.exit", "declare void @nvvm.exit() noreturn nounwind")
        self.emit("call void @nvvm.exit()")
        self.emit("unreachable")

    def i_trap(self, mods, ops, opc):
        self.call_intr("void", "llvm.trap", [])
        self.emit("unreachable")

    def i_brkpt(self, mods, ops, opc):
        self.call_intr("void", "nvvm.brkpt", [])

    def i_fence(self, mods, ops, opc):
        pass  # no-op under single-thread semantics (source fences likewise)

    def i_membar(self, mods, ops, opc):
        pass

    # ---- moves -------------------------------------------------------------
    def vec_ops(self, s):
        s = s.strip()
        if s.startswith("{"):
            return [x.strip() for x in s[1:-1].split(",")]
        return None

    def i_mov(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        dv, sv = self.vec_ops(ops[0]), self.vec_ops(ops[1])
        if dv is None and sv is None:
            self.write(ops[0], self.get(ops[1], bits), bits)
        elif dv is None:  # pack
            eb = bits // len(sv)
            acc = None
            for k, e in enumerate(sv):
                v = self.resize(self.get(e, eb), eb, bits, False)
                if k:
                    v = self.op("shl i%d %s, %d" % (bits, v, eb * k))
                acc = v if acc is None else self.op("or i%d %s, %s" % (bits, acc, v))
            self.write(ops[0], acc, bits)
        else:  # unpack
            eb = bits // len(dv)
            v = self.get(ops[1], bits)
            for k, d in enumerate(dv):
                if d == "_":
                    continue
                x = self.op("lshr i%d %s, %d" % (bits, v, eb * k)) if k else v
                self.write(d, self.resize(x, bits, eb, False), eb)

    def i_cvta(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        self.write(ops[0], self.get(ops[1], bits), bits)

    def i_selp(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        a, b = self.get(ops[1], bits), self.get(ops[2], bits)
        c = self.get(ops[3], 1)
        self.write(ops[0], self.op("select i1 %s, i%d %s, i%d %s" % (c, bits, a, bits, b)), bits)

    # ---- integer arithmetic ------------------------------------------------
    def int_vec(self, t):
        if t in ("s16x2", "u16x2"):
            return "<2 x i16>"
        return None

    def ibin(self, mods, ops, opc, llop):
        ts = self.types_of(mods)
        t = ts[0]
        if is_float(t):
            return self.fbin(mods, ops, opc, "f" + llop)
        bits = TBITS[t]
        vt = self.int_vec(t)
        if vt:
            a = self.op("bitcast i32 %s to %s" % (self.get(ops[1], 32), vt))
            b = self.op("bitcast i32 %s to %s" % (self.get(ops[2], 32), vt))
            r = self.op("%s %s %s, %s" % (llop, vt, a, b))
            self.write(ops[0], self.op("bitcast %s %s to i32" % (vt, r)), 32)
            return
        a, b = self.get(ops[1], bits), self.get(ops[2], bits)
        if "sat" in mods:
            if t != "s32":
                raise Unsupported(opc)
            fn = {"add": "sadd.sat", "sub": "ssub.sat"}[llop]
            r = self.call_intr("i32", "llvm.%s.i32" % fn, ["i32 " + a, "i32 " + b])
        elif "cc" in mods or "c" in mods:
            raise Unsupported(opc)
        else:
            r = self.op("%s i%d %s, %s" % (llop, bits, a, b))
        self.write(ops[0], r, bits)

    def carry_slot(self):
        if self.cc is None:
            self.cc = "%cc.cf"
            self.allocas.append("%cc.cf = alloca i1")
        return self.cc

    def i_add(self, mods, ops, opc):
        if "cc" in mods:
            return self.addsub_cc(mods, ops, False, False)
        self.ibin(mods, ops, opc, "add")

    def i_sub(self, mods, ops, opc):
        if "cc" in mods:
            return self.addsub_cc(mods, ops, True, False)
        self.ibin(mods, ops, opc, "sub")

    def i_addc(self, mods, ops, opc):
        self.addsub_cc(mods, ops, False, True)

    def i_subc(self, mods, ops, opc):
        self.addsub_cc(mods, ops, True, True)

    def addsub_cc(self, mods, ops, sub, use_cin):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        w = bits + 1
        a, b = self.get(ops[1], bits), self.get(ops[2], bits)
        ea = self.op("zext i%d %s to i%d" % (bits, a, w))
        eb = self.op("zext i%d %s to i%d" % (bits, b, w))
        if use_cin:
            cin = self.op("load i1, ptr %s" % self.carry_slot())
            ec = self.op("zext i1 %s to i%d" % (cin, w))
            eb = self.op("add i%d %s, %s" % (w, eb, ec))
        r = self.op("%s i%d %s, %s" % ("sub" if sub else "add", w, ea, eb))
        if "cc" in mods:
            co = self.op("lshr i%d %s, %d" % (w, r, bits))
            co = self.op("trunc i%d %s to i1" % (w, co))
            self.emit("store i1 %s, ptr %s" % (co, self.carry_slot()))
        self.write(ops[0], self.op("trunc i%d %s to i%d" % (w, r, bits)), bits)

    def i_mul(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        if is_float(t):
            return self.fbin(mods, ops, opc, "fmul")
        bits = TBITS[t]
        s = is_signed(t)
        ext = "sext" if s else "zext"
        a, b = self.get(ops[1], bits), self.get(ops[2], bits)
        if "lo" in mods:
            self.write(ops[0], self.op("mul i%d %s, %s" % (bits, a, b)), bits)
        elif "wide" in mods:
            w = bits * 2
            ea = self.op("%s i%d %s to i%d" % (ext, bits, a, w))
            eb = self.op("%s i%d %s to i%d" % (ext, bits, b, w))
            self.write(ops[0], self.op("mul i%d %s, %s" % (w, ea, eb)), w)
        elif "hi" in mods:
            self.write(ops[0], self.mulhi(a, b, bits, ext), bits)
        else:
            raise Unsupported(opc)

    def mulhi(self, a, b, bits, ext):
        w = bits * 2
        ea = self.op("%s i%d %s to i%d" % (ext, bits, a, w))
        eb = self.op("%s i%d %s to i%d" % (ext, bits, b, w))
        p = self.op("mul i%d %s, %s" % (w, ea, eb))
        p = self.op("lshr i%d %s, %d" % (w, p, bits))
        return self.op("trunc i%d %s to i%d" % (w, p, bits))

    def i_mad(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        if is_float(t):
            return self.ffma(mods, ops, opc)
        if "cc" in mods or "sat" in mods:
            raise Unsupported(opc)
        bits = TBITS[t]
        ext = "sext" if is_signed(t) else "zext"
        a, b = self.get(ops[1], bits), self.get(ops[2], bits)
        if "lo" in mods:
            p = self.op("mul i%d %s, %s" % (bits, a, b))
            c = self.get(ops[3], bits)
            self.write(ops[0], self.op("add i%d %s, %s" % (bits, p, c)), bits)
        elif "hi" in mods:
            p = self.mulhi(a, b, bits, ext)
            c = self.get(ops[3], bits)
            self.write(ops[0], self.op("add i%d %s, %s" % (bits, p, c)), bits)
        elif "wide" in mods:
            w = bits * 2
            ea = self.op("%s i%d %s to i%d" % (ext, bits, a, w))
            eb = self.op("%s i%d %s to i%d" % (ext, bits, b, w))
            p = self.op("mul i%d %s, %s" % (w, ea, eb))
            c = self.get(ops[3], w)
            self.write(ops[0], self.op("add i%d %s, %s" % (w, p, c)), w)
        else:
            raise Unsupported(opc)

    def mul24(self, a, b, signed):
        ext = "sext" if signed else "zext"
        ta = self.op("trunc i32 %s to i24" % a)
        tb = self.op("trunc i32 %s to i24" % b)
        ea = self.op("%s i24 %s to i64" % (ext, ta))
        eb = self.op("%s i24 %s to i64" % (ext, tb))
        return self.op("mul i64 %s, %s" % (ea, eb))

    def i_mul24(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        p = self.mul24(self.get(ops[1], 32), self.get(ops[2], 32), is_signed(t))
        if "hi" in mods:
            p = self.op("lshr i64 %s, 16" % p)
        self.write(ops[0], self.op("trunc i64 %s to i32" % p), 32)

    def i_mad24(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        if "sat" in mods:
            raise Unsupported(opc)
        p = self.mul24(self.get(ops[1], 32), self.get(ops[2], 32), is_signed(t))
        if "hi" in mods:
            p = self.op("lshr i64 %s, 16" % p)
        p = self.op("trunc i64 %s to i32" % p)
        self.write(ops[0], self.op("add i32 %s, %s" % (p, self.get(ops[3], 32))), 32)

    def divrem(self, mods, ops, opc, signed_op, unsigned_op):
        t = self.types_of(mods)[0]
        if is_float(t):
            return self.fdiv(mods, ops, opc)
        bits = TBITS[t]
        a, b = self.get(ops[1], bits), self.get(ops[2], bits)
        bad = self.op("icmp eq i%d %s, 0" % (bits, b))
        if is_signed(t):
            mn = self.op("icmp eq i%d %s, %s" % (bits, a, self.const(1 << (bits - 1), bits)))
            m1 = self.op("icmp eq i%d %s, -1" % (bits, b))
            ov = self.op("and i1 %s, %s" % (mn, m1))
            bad = self.op("or i1 %s, %s" % (bad, ov))
        safe = self.op("select i1 %s, i%d 1, i%d %s" % (bad, bits, bits, b))
        r = self.op("%s i%d %s, %s" % (signed_op if is_signed(t) else unsigned_op,
                                        bits, a, safe))
        # PTX: division by zero (and signed overflow) yields an unspecified value
        any_ = self.op("freeze i%d poison" % bits)
        self.write(ops[0], self.op("select i1 %s, i%d %s, i%d %s" %
                                   (bad, bits, any_, bits, r)), bits)

    def i_div(self, mods, ops, opc):
        self.divrem(mods, ops, opc, "sdiv", "udiv")

    def i_rem(self, mods, ops, opc):
        self.divrem(mods, ops, opc, "srem", "urem")

    def i_abs(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        if is_float(t):
            return self.funary(mods, ops, opc, "fabs")
        bits = TBITS[t]
        r = self.call_intr("i%d" % bits, "llvm.abs.i%d" % bits,
                           ["i%d %s" % (bits, self.get(ops[1], bits)), "i1 false"])
        self.write(ops[0], r, bits)

    def i_neg(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        if is_float(t):
            return self.funary(mods, ops, opc, "fneg")
        bits = TBITS[t]
        self.write(ops[0], self.op("sub i%d 0, %s" % (bits, self.get(ops[1], bits))), bits)

    def minmax(self, mods, ops, opc, is_min):
        t = self.types_of(mods)[0]
        if is_float(t):
            return self.fminmax(mods, ops, opc, is_min)
        bits = TBITS[t]
        vt = self.int_vec(t)
        s = is_signed(t)
        fn = ("s" if s else "u") + ("min" if is_min else "max")
        if vt:
            a = self.op("bitcast i32 %s to %s" % (self.get(ops[1], 32), vt))
            b = self.op("bitcast i32 %s to %s" % (self.get(ops[2], 32), vt))
            r = self.call_intr(vt, "llvm.%s.v2i16" % fn, ["%s %s" % (vt, a), "%s %s" % (vt, b)])
            if "relu" in mods:
                r = self.call_intr(vt, "llvm.smax.v2i16", ["%s %s" % (vt, r), "%s zeroinitializer" % vt])
            self.write(ops[0], self.op("bitcast %s %s to i32" % (vt, r)), 32)
            return
        if len(ops) != 3:
            raise Unsupported(opc)
        it = "i%d" % bits
        r = self.call_intr(it, "llvm.%s.%s" % (fn, it),
                           ["%s %s" % (it, self.get(ops[1], bits)),
                            "%s %s" % (it, self.get(ops[2], bits))])
        if "relu" in mods:
            r = self.call_intr(it, "llvm.smax.%s" % it, ["%s %s" % (it, r), "%s 0" % it])
        self.write(ops[0], r, bits)

    def i_min(self, mods, ops, opc):
        self.minmax(mods, ops, opc, True)

    def i_max(self, mods, ops, opc):
        self.minmax(mods, ops, opc, False)

    def logic(self, mods, ops, opc, llop):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        a, b = self.get(ops[1], bits), self.get(ops[2], bits)
        self.write(ops[0], self.op("%s i%d %s, %s" % (llop, bits, a, b)), bits)

    def i_and(self, mods, ops, opc):
        self.logic(mods, ops, opc, "and")

    def i_or(self, mods, ops, opc):
        self.logic(mods, ops, opc, "or")

    def i_xor(self, mods, ops, opc):
        self.logic(mods, ops, opc, "xor")

    def i_not(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        self.write(ops[0], self.op("xor i%d %s, -1" % (bits, self.get(ops[1], bits))), bits)

    def i_cnot(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        c = self.op("icmp eq i%d %s, 0" % (bits, self.get(ops[1], bits)))
        self.write(ops[0], self.op("zext i1 %s to i%d" % (c, bits)), bits)

    def shift_amt(self, s, bits):
        return self.get(s, 32)

    def shift(self, mods, ops, opc, kind):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        a = self.get(ops[1], bits)
        amt = self.get(ops[2], 32)
        big = self.op("icmp ugt i32 %s, %d" % (amt, bits - 1))
        cl = self.op("select i1 %s, i32 %d, i32 %s" % (big, bits - 1, amt))
        cl = self.resize(cl, 32, bits, False) if bits != 32 else cl
        if kind == "shl":
            r = self.op("shl i%d %s, %s" % (bits, a, cl))
            r = self.op("select i1 %s, i%d 0, i%d %s" % (big, bits, bits, r))
        elif kind == "lshr":
            r = self.op("lshr i%d %s, %s" % (bits, a, cl))
            r = self.op("select i1 %s, i%d 0, i%d %s" % (big, bits, bits, r))
        else:
            r = self.op("ashr i%d %s, %s" % (bits, a, cl))
        self.write(ops[0], r, bits)

    def i_shl(self, mods, ops, opc):
        self.shift(mods, ops, opc, "shl")

    def i_shr(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        self.shift(mods, ops, opc, "ashr" if is_signed(t) else "lshr")

    def i_shf(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        if bits != 32:
            raise Unsupported(opc)
        a, b, c = self.get(ops[1], 32), self.get(ops[2], 32), self.get(ops[3], 32)
        left = "l" in mods
        fn = "fshl" if left else "fshr"
        if "clamp" in mods:
            big = self.op("icmp uge i32 %s, 32" % c)
            r = self.call_intr("i32", "llvm.%s.i32" % fn, ["i32 " + b, "i32 " + a, "i32 " + c])
            r = self.op("select i1 %s, i32 %s, i32 %s" % (big, a if left else b, r))
        else:
            r = self.call_intr("i32", "llvm.%s.i32" % fn, ["i32 " + b, "i32 " + a, "i32 " + c])
        self.write(ops[0], r, 32)

    def i_popc(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        r = self.call_intr("i%d" % bits, "llvm.ctpop.i%d" % bits, ["i%d %s" % (bits, self.get(ops[1], bits))])
        self.write(ops[0], self.resize(r, bits, 32, False), 32)

    def i_clz(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        r = self.call_intr("i%d" % bits, "llvm.ctlz.i%d" % bits, ["i%d %s" % (bits, self.get(ops[1], bits)), "i1 false"])
        self.write(ops[0], self.resize(r, bits, 32, False), 32)

    def i_brev(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        r = self.call_intr("i%d" % bits, "llvm.bitreverse.i%d" % bits, ["i%d %s" % (bits, self.get(ops[1], bits))])
        self.write(ops[0], r, bits)

    def i_bfind(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        a = self.get(ops[1], bits)
        if is_signed(t):
            neg = self.op("icmp slt i%d %s, 0" % (bits, a))
            na = self.op("xor i%d %s, -1" % (bits, a))
            a = self.op("select i1 %s, i%d %s, i%d %s" % (neg, bits, na, bits, a))
        lz = self.call_intr("i%d" % bits, "llvm.ctlz.i%d" % bits, ["i%d %s" % (bits, a), "i1 false"])
        lz = self.resize(lz, bits, 32, False)
        z = self.op("icmp eq i%d %s, 0" % (bits, a))
        r = lz if "shiftamt" in mods else self.op("sub i32 %d, %s" % (bits - 1, lz))
        self.write(ops[0], self.op("select i1 %s, i32 -1, i32 %s" % (z, r)), 32)

    def i_bfe(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        w = TBITS[t]
        it = "i%d" % w
        a = self.get(ops[1], w)
        pos = self.op("and i32 %s, 255" % self.get(ops[2], 32))
        ln = self.op("and i32 %s, 255" % self.get(ops[3], 32))
        s = is_signed(t)
        posw = self.resize(pos, 32, w, False)
        pbig = self.op("icmp uge i32 %s, %d" % (pos, w))
        pcl = self.op("select i1 %s, %s %d, %s %s" % (pbig, it, w - 1, it, posw))
        if s:
            sh = self.op("ashr %s %s, %s" % (it, a, pcl))
        else:
            sh = self.op("lshr %s %s, %s" % (it, a, pcl))
            sh = self.op("select i1 %s, %s 0, %s %s" % (pbig, it, it, sh))
        lbig = self.op("icmp uge i32 %s, %d" % (ln, w))
        lz = self.op("icmp eq i32 %s, 0" % ln)
        lnw = self.resize(ln, 32, w, False)
        lsafe = self.op("select i1 %s, %s 1, %s %s" % (self.op("or i1 %s, %s" % (lbig, lz)), it, it, lnw))
        amt = self.op("sub %s %d, %s" % (it, w, lsafe))
        x = self.op("shl %s %s, %s" % (it, sh, amt))
        x = self.op("%s %s %s, %s" % ("ashr" if s else "lshr", it, x, amt))
        r = self.op("select i1 %s, %s %s, %s %s" % (lbig, it, sh, it, x))
        r = self.op("select i1 %s, %s 0, %s %s" % (lz, it, it, r))
        self.write(ops[0], r, w)

    def i_bfi(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        w = TBITS[t]
        it = "i%d" % w
        a, b = self.get(ops[1], w), self.get(ops[2], w)
        pos = self.op("and i32 %s, 255" % self.get(ops[3], 32))
        ln = self.op("and i32 %s, 255" % self.get(ops[4], 32))
        posw = self.resize(pos, 32, w, False)
        lnw = self.resize(ln, 32, w, False)
        lbig = self.op("icmp uge i32 %s, %d" % (ln, w))
        lsafe = self.op("select i1 %s, %s 0, %s %s" % (lbig, it, it, lnw))
        m = self.op("shl %s 1, %s" % (it, lsafe))
        m = self.op("sub %s %s, 1" % (it, m))
        m = self.op("select i1 %s, %s -1, %s %s" % (lbig, it, it, m))
        pbig = self.op("icmp uge i32 %s, %d" % (pos, w))
        psafe = self.op("select i1 %s, %s 0, %s %s" % (pbig, it, it, posw))
        m = self.op("shl %s %s, %s" % (it, m, psafe))
        m = self.op("select i1 %s, %s 0, %s %s" % (pbig, it, it, m))
        ash = self.op("shl %s %s, %s" % (it, a, psafe))
        x = self.op("and %s %s, %s" % (it, ash, m))
        nm = self.op("xor %s %s, -1" % (it, m))
        y = self.op("and %s %s, %s" % (it, b, nm))
        self.write(ops[0], self.op("or %s %s, %s" % (it, x, y)), w)

    def i_prmt(self, mods, ops, opc):
        a, b, c = self.get(ops[1], 32), self.get(ops[2], 32), self.get(ops[3], 32)
        ea = self.op("zext i32 %s to i64" % a)
        eb = self.op("zext i32 %s to i64" % b)
        eb = self.op("shl i64 %s, 32" % eb)
        ab = self.op("or i64 %s, %s" % (ea, eb))
        mode = [m for m in mods if m in ("f4e", "b4e", "rc8", "ecl", "ecr", "rc16")]
        res = None
        if mode:
            s = self.op("and i32 %s, 3" % c)
        for i in range(4):
            if not mode:
                nib = self.op("lshr i32 %s, %d" % (c, 4 * i))
                sel = self.op("and i32 %s, 7" % nib)
                sgn = self.op("and i32 %s, 8" % nib)
                sgn = self.op("icmp ne i32 %s, 0" % sgn)
            else:
                md = mode[0]
                if md == "f4e":
                    sel = self.op("add i32 %s, %d" % (s, i))
                elif md == "b4e":
                    sel = self.op("sub i32 %s, %d" % (s, i))
                    sel = self.op("and i32 %s, 7" % sel)
                elif md == "rc8":
                    sel = s
                elif md == "ecl":
                    sel = self.call_intr("i32", "llvm.umax.i32", ["i32 %s" % s, "i32 %d" % i])
                elif md == "ecr":
                    sel = self.call_intr("i32", "llvm.umin.i32", ["i32 %s" % s, "i32 %d" % i])
                else:  # rc16
                    s0 = self.op("and i32 %s, 1" % c)
                    sel = self.op("shl i32 %s, 1" % s0)
                    sel = self.op("or i32 %s, %d" % (sel, i & 1))
                sgn = "false"
            sh = self.op("shl i32 %s, 3" % sel)
            sh = self.op("zext i32 %s to i64" % sh)
            byte = self.op("lshr i64 %s, %s" % (ab, sh))
            byte = self.op("trunc i64 %s to i8" % byte)
            rep = self.op("ashr i8 %s, 7" % byte)
            byte = self.op("select i1 %s, i8 %s, i8 %s" % (sgn, rep, byte))
            v = self.op("zext i8 %s to i32" % byte)
            if i:
                v = self.op("shl i32 %s, %d" % (v, 8 * i))
            res = v if res is None else self.op("or i32 %s, %s" % (res, v))
        self.write(ops[0], res, 32)

    def i_lop3(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        it = "i%d" % bits
        a, b, c = (self.get(o, bits) for o in ops[1:4])
        lut = self.imm(ops[4])
        if lut is None:
            raise Unsupported("lop3 non-constant")
        na, nb, nc = (self.op("xor %s %s, -1" % (it, x)) for x in (a, b, c))
        acc = "0"
        for idx in range(8):
            if not (lut >> idx) & 1:
                continue
            x = self.op("and %s %s, %s" % (it, a if idx & 4 else na, b if idx & 2 else nb))
            x = self.op("and %s %s, %s" % (it, x, c if idx & 1 else nc))
            acc = self.op("or %s %s, %s" % (it, acc, x))
        self.write(ops[0], acc, bits)

    def i_sad(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        it = "i%d" % bits
        a, b, c = (self.get(o, bits) for o in ops[1:4])
        s = "s" if is_signed(t) else "u"
        mx = self.call_intr(it, "llvm.%smax.%s" % (s, it), ["%s %s" % (it, a), "%s %s" % (it, b)])
        mn = self.call_intr(it, "llvm.%smin.%s" % (s, it), ["%s %s" % (it, a), "%s %s" % (it, b)])
        d = self.op("sub %s %s, %s" % (it, mx, mn))
        self.write(ops[0], self.op("add %s %s, %s" % (it, d, c)), bits)

    def i_dp4a(self, mods, ops, opc):
        ts = self.types_of(mods)
        a, b, c = (self.get(o, 32) for o in ops[1:4])
        acc = c
        for i in range(4):
            xs = []
            for v, t in ((a, ts[0]), (b, ts[1])):
                x = self.op("lshr i32 %s, %d" % (v, 8 * i)) if i else v
                x = self.op("trunc i32 %s to i8" % x)
                xs.append(self.op("%s i8 %s to i32" % ("sext" if is_signed(t) else "zext", x)))
            p = self.op("mul i32 %s, %s" % tuple(xs))
            acc = self.op("add i32 %s, %s" % (acc, p))
        self.write(ops[0], acc, 32)

    def i_dp2a(self, mods, ops, opc):
        ts = self.types_of(mods)
        a, b, c = (self.get(o, 32) for o in ops[1:4])
        hi = "hi" in mods
        acc = c
        for i in range(2):
            x = self.op("lshr i32 %s, %d" % (a, 16 * i)) if i else a
            x = self.op("trunc i32 %s to i16" % x)
            x = self.op("%s i16 %s to i32" % ("sext" if is_signed(ts[0]) else "zext", x))
            k = i + (2 if hi else 0)
            y = self.op("lshr i32 %s, %d" % (b, 8 * k)) if k else b
            y = self.op("trunc i32 %s to i8" % y)
            y = self.op("%s i8 %s to i32" % ("sext" if is_signed(ts[1]) else "zext", y))
            acc = self.op("add i32 %s, %s" % (acc, self.op("mul i32 %s, %s" % (x, y))))
        self.write(ops[0], acc, 32)

    # ---- comparisons -------------------------------------------------------
    ICMP = {"eq": ("eq", "eq"), "ne": ("ne", "ne"), "lt": ("slt", "ult"),
            "le": ("sle", "ule"), "gt": ("sgt", "ugt"), "ge": ("sge", "uge"),
            "lo": ("ult", "ult"), "ls": ("ule", "ule"), "hi": ("ugt", "ugt"),
            "hs": ("uge", "uge")}
    FCMP = {"eq": "oeq", "ne": "one", "lt": "olt", "le": "ole", "gt": "ogt",
            "ge": "oge", "equ": "ueq", "neu": "une", "ltu": "ult",
            "leu": "ule", "gtu": "ugt", "geu": "uge", "num": "ord",
            "nan": "uno"}

    def compare(self, cmp, t, ops, ftz):
        if is_float(t):
            if t.endswith("x2"):
                raise Unsupported("packed setp")
            fty = FPTY[t]
            a = self.ftz_in(self.getf(ops[0], t), fty, ftz)
            b = self.ftz_in(self.getf(ops[1], t), fty, ftz)
            return self.op("fcmp %s %s %s, %s" % (self.FCMP[cmp], fty, a, b))
        bits = TBITS[t]
        a, b = self.get(ops[0], bits), self.get(ops[1], bits)
        pred = self.ICMP[cmp][0 if is_signed(t) else 1]
        return self.op("icmp %s i%d %s, %s" % (pred, bits, a, b))

    def setp_x2(self, cmp, t, mods, ops):
        # setp.<cmp>.f16x2 p|q, a, b: p = low halves, q = high halves.
        # (Lane order as used by the open-source NVPTX backend.)
        if any(m in ("and", "or", "xor") for m in mods):
            raise Unsupported("packed setp with bool op")
        vt = FPTY[t]
        a = self.ftz_in(self.getf(ops[1], t), vt, "ftz" in mods)
        b = self.ftz_in(self.getf(ops[2], t), vt, "ftz" in mods)
        c = self.op("fcmp %s %s %s, %s" % (self.FCMP[cmp], vt, a, b))
        dsts = [d.strip() for d in ops[0].split("|")]
        if len(dsts) != 2:
            raise Unsupported("packed setp needs p|q")
        self.write(dsts[0], self.op("extractelement <2 x i1> %s, i32 0" % c), 1)
        self.write(dsts[1], self.op("extractelement <2 x i1> %s, i32 1" % c), 1)

    def i_setp(self, mods, ops, opc):
        cmp = mods[0]
        t = self.types_of(mods)[0]
        if t in ("f16x2", "bf16x2"):
            return self.setp_x2(cmp, t, mods, ops)
        boolop = [m for m in mods if m in ("and", "or", "xor")]
        r = self.compare(cmp, t, ops[1:3], "ftz" in mods)
        dsts = ops[0].split("|")
        nr = self.op("xor i1 %s, true" % r) if len(dsts) > 1 else None
        if boolop:
            c = ops[3].strip()
            neg = c.startswith("!")
            cv = self.get(c.lstrip("!"), 1)
            if neg:
                cv = self.op("xor i1 %s, true" % cv)
            r = self.op("%s i1 %s, %s" % (boolop[0], r, cv))
            if nr:
                nr = self.op("%s i1 %s, %s" % (boolop[0], nr, cv))
        self.write(dsts[0].strip(), r, 1)
        if len(dsts) > 1:
            self.write(dsts[1].strip(), nr, 1)

    def i_set(self, mods, ops, opc):
        raise Unsupported(opc)

    # ---- floating point ----------------------------------------------------
    def ftz_in(self, v, fty, ftz):
        if not ftz or fty not in ("float", "<2 x float>"):
            return v
        e = "float"
        a = self.call_intr(fty, "llvm.fabs.%s" % intname(fty), ["%s %s" % (fty, v)])
        sm = self.op("fcmp olt %s %s, %s" % (fty, a, splat(fty, SMALLEST_NORMAL[e])))
        z = self.call_intr(fty, "llvm.copysign.%s" % intname(fty),
                           ["%s %s" % (fty, fconst(fty, 0.0)), "%s %s" % (fty, v)])
        return self.op("select %s %s, %s %s, %s %s" %
                       ("i1" if not fty.startswith("<") else "<2 x i1>", sm, fty, z, fty, v))

    ftz_out = ftz_in

    def rounding(self, mods):
        r = [m for m in mods if m in MODES]
        return r[0] if r else "rn"

    def sat(self, v, fty):
        lo = self.call_intr(fty, "llvm.maxnum.%s" % intname(fty), ["%s %s" % (fty, v), "%s %s" % (fty, fconst(fty, 0.0))])
        return self.call_intr(fty, "llvm.minnum.%s" % intname(fty), ["%s %s" % (fty, lo), "%s %s" % (fty, fconst(fty, 1.0))])

    def relu(self, v, fty):
        ct = "<%s x i1>" % fty[1:].split(" x ")[0] if fty.startswith("<") else "i1"
        isn = self.op("fcmp uno %s %s, %s" % (fty, v, v))
        neg = self.op("fcmp olt %s %s, %s" % (fty, v, fconst(fty, 0.0)))
        r = self.op("select %s %s, %s %s, %s %s" % (ct, neg, fty, fconst(fty, 0.0), fty, v))
        return self.op("select %s %s, %s %s, %s %s" % (ct, isn, fty, v, fty, r))

    MAX_FINITE = {"half": 65504.0, "bfloat": 3.3895313892515355e38,
                  "float": 3.4028234663852886e38}

    def satfinite(self, v, fty):
        """.satfinite: +-Inf results become +-MAX_NORM; NaN stays NaN."""
        e = fty.split(" x ")[1][:-1] if fty.startswith("<") else fty
        if e not in self.MAX_FINITE:
            raise Unsupported("satfinite for " + fty)
        ct = "<%s x i1>" % fty[1:].split(" x ")[0] if fty.startswith("<") else "i1"
        a = self.call_intr(fty, "llvm.fabs.%s" % intname(fty), ["%s %s" % (fty, v)])
        inf = self.op("fcmp oeq %s %s, %s" % (fty, a, fconst(fty, float("inf"))))
        mx = self.call_intr(fty, "llvm.copysign.%s" % intname(fty),
                            ["%s %s" % (fty, fconst(fty, self.MAX_FINITE[e])), "%s %s" % (fty, v)])
        return self.op("select %s %s, %s %s, %s %s" % (ct, inf, fty, mx, fty, v))

    def post(self, v, fty, mods):
        if "satfinite" in mods:
            v = self.satfinite(v, fty)
        v = self.ftz_out(v, fty, "ftz" in mods)
        if "sat" in mods:
            v = self.sat(v, fty)
        if "relu" in mods:
            v = self.relu(v, fty)
        return v

    def mixed_operand(self, op, t, at):
        """Operand of PTX type `at` converted exactly to type `t` (mixed precision)."""
        if at == t:
            return self.getf(op, t)
        if t.endswith("x2") != at.endswith("x2"):
            raise Unsupported("packed mixed precision")
        v = self.getf(op, at)
        return self.op("fpext %s %s to %s" % (FPTY[at], v, FPTY[t]))

    WIDTH = {"f16": 1, "bf16": 1, "f16x2": 1, "bf16x2": 1, "f32": 2, "f32x2": 2}

    def fbin_down(self, mods, ops, opc, llop, t, st):
        """add/sub/mul.rnd.<narrow>.<wide> d, a, b: one rounding to `t`.

        Computed in double, then rounded to `t`: exact for directed rounding
        (directed roundings compose) and for rn multiplication (the f32
        product is exact in double).  Packed forms are done lane by lane.
        """
        rm = self.rounding(mods)
        if rm == "rn" and llop != "fmul":
            raise Unsupported("rn down-converting " + opc)
        vec = t.endswith("x2")
        sfty = FPTY[st]
        a, b = self.getf(ops[1], st), self.getf(ops[2], st)
        fty = FPTY[t]
        efty = fty.split(" x ")[1][:-1] if vec else fty
        esty = sfty.split(" x ")[1][:-1] if vec else sfty
        lanes = []
        for k in (range(2) if vec else [None]):
            x = self.op("extractelement %s %s, i32 %d" % (sfty, a, k)) if vec else a
            y = self.op("extractelement %s %s, i32 %d" % (sfty, b, k)) if vec else b
            x = self.op("fpext %s %s to double" % (esty, x))
            y = self.op("fpext %s %s to double" % (esty, y))
            if llop == "fmul":
                r = self.op("fmul double %s, %s" % (x, y))
            else:
                name = "llvm.experimental.constrained.%s.f64" % llop
                self.declare(name, "declare double @%s(double, double, metadata, metadata)" % name)
                r = self.op('call double @%s(double %s, double %s, metadata !"%s", metadata !"fpexcept.ignore") strictfp'
                            % (name, x, y, MODES[rm]))
            if rm == "rn":
                v = self.op("fptrunc double %s to %s" % (r, efty))
            else:
                name = "llvm.experimental.constrained.fptrunc.%s.f64" % FP_SCALAR[efty]
                self.declare(name, "declare %s @%s(double, metadata, metadata)" % (efty, name))
                v = self.op('call %s @%s(double %s, metadata !"%s", metadata !"fpexcept.ignore") strictfp'
                            % (efty, name, r, MODES[rm]))
            lanes.append(v)
        if vec:
            v = self.op("insertelement %s poison, %s %s, i32 0" % (fty, efty, lanes[0]))
            v = self.op("insertelement %s %s, %s %s, i32 1" % (fty, v, efty, lanes[1]))
        else:
            v = lanes[0]
        self.writef(ops[0], self.post(v, fty, mods), t)

    def fbin(self, mods, ops, opc, llop):
        ts = self.types_of(mods)
        t = ts[0]
        at = ts[1] if len(ts) > 1 else t   # add.rn.f32.f16 d, a(f16), b(f32)
        if len(ts) > 1 and self.WIDTH.get(t, 0) < self.WIDTH.get(at, 0):
            return self.fbin_down(mods, ops, opc, llop, t, at)
        fty = FPTY[t]
        ftz = "ftz" in mods
        a = self.ftz_in(self.mixed_operand(ops[1], t, at), fty, ftz)
        b = self.ftz_in(self.getf(ops[2], t), fty, ftz)
        rm = self.rounding(mods)
        if "approx" in mods or "full" in mods:
            raise Unsupported(opc)
        if rm == "rn":
            r = self.op("%s %s %s, %s" % (llop, fty, a, b))
        else:
            name = "llvm.experimental.constrained.%s.%s" % (llop, intname(fty))
            self.declare(name, "declare %s @%s(%s, %s, metadata, metadata)" % (fty, name, fty, fty))
            r = self.op('call %s @%s(%s %s, %s %s, metadata !"%s", metadata !"fpexcept.ignore") strictfp'
                        % (fty, name, fty, a, fty, b, MODES[rm]))
        self.writef(ops[0], self.post(r, fty, mods), t)

    def ffma(self, mods, ops, opc):
        ts = self.types_of(mods)
        t = ts[0]
        at = ts[1] if len(ts) > 1 else t   # fma.rn.f32.f16 d, a(f16), b(f16), c(f32)
        fty = FPTY[t]
        ftz = "ftz" in mods
        if not any(m in MODES for m in mods):
            raise Unsupported(opc + " (no rounding mode)")
        a = self.ftz_in(self.mixed_operand(ops[1], t, at), fty, ftz)
        b = self.ftz_in(self.mixed_operand(ops[2], t, at), fty, ftz)
        c = self.ftz_in(self.getf(ops[3], t), fty, ftz)
        rm = self.rounding(mods)
        if rm == "rn":
            r = self.call_intr(fty, "llvm.fma.%s" % intname(fty),
                               ["%s %s" % (fty, x) for x in (a, b, c)])
        else:
            name = "llvm.experimental.constrained.fma.%s" % intname(fty)
            self.declare(name, "declare %s @%s(%s, %s, %s, metadata, metadata)" % (fty, name, fty, fty, fty))
            r = self.op('call %s @%s(%s %s, %s %s, %s %s, metadata !"%s", metadata !"fpexcept.ignore") strictfp'
                        % (fty, name, fty, a, fty, b, fty, c, MODES[rm]))
        self.writef(ops[0], self.post(r, fty, mods), t)

    def i_fma(self, mods, ops, opc):
        if "oob" in mods:
            # .oob: special handling of "out-of-bounds" NaN inputs; uninterpreted
            return self.uninterp(mods, ops, opc, 3)
        self.ffma(mods, ops, opc)

    def fdiv(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        if "approx" in mods or "full" in mods:
            return self.uninterp(mods, ops, opc, 2)
        self.fbin(mods, ops, opc, "fdiv")

    def uninterp(self, mods, ops, opc, nargs):
        """Approximate instruction -> uninterpreted pure function."""
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        name = "ptx." + opc
        args = [self.getf(o, t) for o in ops[1:1 + nargs]]
        self.declare(name, "declare %s @%s(%s) nounwind willreturn memory(none)" %
                     (fty, name, ", ".join([fty] * nargs)))
        r = self.op("call %s @%s(%s)" % (fty, name, ", ".join("%s %s" % (fty, a) for a in args)))
        self.writef(ops[0], r, t)

    def funary(self, mods, ops, opc, kind):
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        a = self.ftz_in(self.getf(ops[1], t), fty, "ftz" in mods)
        if kind == "fneg":
            r = self.op("fneg %s %s" % (fty, a))
        else:
            r = self.call_intr(fty, "llvm.fabs.%s" % intname(fty), ["%s %s" % (fty, a)])
        self.writef(ops[0], self.post(r, fty, mods), t)

    def fminmax(self, mods, ops, opc, is_min):
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        # PTX ISA semantics of min/max: a NaN operand yields the other operand,
        # two NaNs give NaN, and -0.0 < +0.0 -- LLVM's minimumnum/maximumnum;
        # with .NaN any NaN gives NaN -- minimum/maximum
        fn = ("minimum" if is_min else "maximum") if "NaN" in mods else \
            ("minimumnum" if is_min else "maximumnum")
        args = [self.ftz_in(self.getf(o, t), fty, "ftz" in mods) for o in ops[1:]]
        if "xorsign" in mods:
            # min/max.xorsign.abs: |a| op |b| with sign bit sign(a) ^ sign(b)
            if len(args) != 2 or "abs" not in mods:
                raise Unsupported(opc)
            bits = TBITS[t]
            it = "i%d" % bits if not fty.startswith("<") else "<2 x i%d>" % (bits // 2)
            eb = bits if not fty.startswith("<") else bits // 2
            ai = self.op("bitcast %s %s to %s" % (fty, args[0], it))
            bi = self.op("bitcast %s %s to %s" % (fty, args[1], it))
            sg = self.op("xor %s %s, %s" % (it, ai, bi))
            sm = splat(it, str(-(1 << (eb - 1)))) if it.startswith("<") else str(-(1 << (eb - 1)))
            sg = self.op("and %s %s, %s" % (it, sg, sm))
            ab = [self.call_intr(fty, "llvm.fabs.%s" % intname(fty), ["%s %s" % (fty, x)]) for x in args]
            r = self.call_intr(fty, "llvm.%s.%s" % (fn, intname(fty)), ["%s %s" % (fty, ab[0]), "%s %s" % (fty, ab[1])])
            ri = self.op("bitcast %s %s to %s" % (fty, r, it))
            nm = splat(it, str((1 << (eb - 1)) - 1)) if it.startswith("<") else str((1 << (eb - 1)) - 1)
            ri = self.op("and %s %s, %s" % (it, ri, nm))
            ri = self.op("or %s %s, %s" % (it, ri, sg))
            self.writef(ops[0], self.op("bitcast %s %s to %s" % (it, ri, fty)), t)
            return
        if "abs" in mods:  # min.abs / max.abs: operate on |inputs|
            args = [self.call_intr(fty, "llvm.fabs.%s" % intname(fty), ["%s %s" % (fty, x)])
                    for x in args]
        r = args[0]
        for b in args[1:]:
            r = self.call_intr(fty, "llvm.%s.%s" % (fn, intname(fty)), ["%s %s" % (fty, r), "%s %s" % (fty, b)])
        self.writef(ops[0], self.post(r, fty, [m for m in mods if m != "relu"] +
                                      (["relu"] if "relu" in mods else [])), t)

    def i_sqrt(self, mods, ops, opc):
        if "approx" in mods:
            return self.uninterp(mods, ops, opc, 1)
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        a = self.ftz_in(self.getf(ops[1], t), fty, "ftz" in mods)
        rm = self.rounding(mods)
        if rm != "rn":
            raise Unsupported(opc)
        r = self.call_intr(fty, "llvm.sqrt.%s" % intname(fty), ["%s %s" % (fty, a)])
        self.writef(ops[0], self.post(r, fty, mods), t)

    def i_rcp(self, mods, ops, opc):
        if "approx" in mods:
            return self.uninterp(mods, ops, opc, 1)
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        a = self.ftz_in(self.getf(ops[1], t), fty, "ftz" in mods)
        rm = self.rounding(mods)
        if rm == "rn":
            r = self.op("fdiv %s %s, %s" % (fty, fconst(fty, 1.0), a))
        else:
            name = "llvm.experimental.constrained.fdiv.%s" % intname(fty)
            self.declare(name, "declare %s @%s(%s, %s, metadata, metadata)" % (fty, name, fty, fty))
            r = self.op('call %s @%s(%s %s, %s %s, metadata !"%s", metadata !"fpexcept.ignore") strictfp'
                        % (fty, name, fty, fconst(fty, 1.0), fty, a, MODES[rm]))
        self.writef(ops[0], self.post(r, fty, mods), t)

    def i_rsqrt(self, mods, ops, opc):
        self.uninterp(mods, ops, opc, 1)

    def i_ex2(self, mods, ops, opc):
        self.uninterp(mods, ops, opc, 1)

    def i_lg2(self, mods, ops, opc):
        self.uninterp(mods, ops, opc, 1)

    def i_sin(self, mods, ops, opc):
        self.uninterp(mods, ops, opc, 1)

    def i_cos(self, mods, ops, opc):
        self.uninterp(mods, ops, opc, 1)

    def i_tanh(self, mods, ops, opc):
        self.uninterp(mods, ops, opc, 1)

    def i_copysign(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        a, b = self.getf(ops[1], t), self.getf(ops[2], t)
        r = self.call_intr(fty, "llvm.copysign.%s" % intname(fty), ["%s %s" % (fty, b), "%s %s" % (fty, a)])
        self.writef(ops[0], r, t)

    def i_testp(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        fty = FPTY[t]
        a = self.getf(ops[1], t)
        cls = {"finite": 0x1f8, "infinite": 0x204, "number": 0x3fc,
               "notanumber": 0x3, "normal": 0x108, "subnormal": 0x90}
        k = [m for m in mods if m in cls][0]
        r = self.call_intr("i1", "llvm.is.fpclass.%s" % intname(fty), ["%s %s" % (fty, a), "i32 %d" % cls[k]])
        self.write(ops[0], r, 1)

    # ---- conversions -------------------------------------------------------
    def i_cvt(self, mods, ops, opc):
        ts = self.types_of(mods)
        if len(ts) != 2:
            raise Unsupported(opc)
        dt, at = ts
        ftz = "ftz" in mods
        irnd = [m for m in mods if m in ("rni", "rzi", "rmi", "rpi")]
        frnd = [m for m in mods if m in MODES]
        if dt.endswith("x2"):
            if not is_float(dt) or at != "f32" or len(ops) != 3:
                raise Unsupported(opc)
            et = dt[:-2]
            hi = self.cvt_ff(self.ftz_in(self.getf(ops[1], at), "float", ftz), "float", FPTY[et], frnd, mods)
            lo = self.cvt_ff(self.ftz_in(self.getf(ops[2], at), "float", ftz), "float", FPTY[et], frnd, mods)
            hi = self.post(hi, FPTY[et], [m for m in mods if m in ("satfinite", "relu")])
            lo = self.post(lo, FPTY[et], [m for m in mods if m in ("satfinite", "relu")])
            hi = self.op("bitcast %s %s to i16" % (FPTY[et], hi))
            lo = self.op("bitcast %s %s to i16" % (FPTY[et], lo))
            hi = self.op("zext i16 %s to i32" % hi)
            hi = self.op("shl i32 %s, 16" % hi)
            lo = self.op("zext i16 %s to i32" % lo)
            self.write(ops[0], self.op("or i32 %s, %s" % (hi, lo)), 32)
            return
        if at.endswith("x2") or dt not in TBITS or at not in TBITS or \
                dt == "pred" or at == "pred" or dt == "b128":
            raise Unsupported(opc)
        db, ab = TBITS[dt], TBITS[at]
        if not is_float(dt) and not is_float(at):
            v = self.get(ops[1], ab)
            v = self.cvt_ii(v, at, dt, "sat" in mods)
            self.write(ops[0], v, db, is_signed(dt))
            return
        if is_float(at) and not is_float(dt):
            fty = FPTY[at]
            v = self.ftz_in(self.getf(ops[1], at), fty, ftz)
            if irnd and irnd[0] != "rzi":
                fn = {"rni": "roundeven", "rmi": "floor", "rpi": "ceil"}[irnd[0]]
                v = self.call_intr(fty, "llvm.%s.%s" % (fn, intname(fty)), ["%s %s" % (fty, v)])
            elif not irnd:
                raise Unsupported(opc)
            fn = "fptosi" if is_signed(dt) else "fptoui"
            r = self.call_intr("i%d" % db, "llvm.%s.sat.i%d.%s" % (fn, db, intname(fty)),
                               ["%s %s" % (fty, v)])
            self.write(ops[0], r, db, is_signed(dt))
            return
        if not is_float(at) and is_float(dt):
            fty = FPTY[dt]
            v = self.get(ops[1], ab)
            rm = frnd[0] if frnd else None
            if rm is None:
                raise Unsupported(opc)
            fn = "sitofp" if is_signed(at) else "uitofp"
            if rm == "rn":
                r = self.op("%s i%d %s to %s" % (fn, ab, v, fty))
            else:
                name = "llvm.experimental.constrained.%s.%s.i%d" % (fn, intname(fty), ab)
                self.declare(name, "declare %s @%s(i%d, metadata, metadata)" % (fty, name, ab))
                r = self.op('call %s @%s(i%d %s, metadata !"%s", metadata !"fpexcept.ignore") strictfp'
                            % (fty, name, ab, v, MODES[rm]))
            r = self.post(r, fty, mods)
            self.writef(ops[0], r, dt)
            return
        # float -> float
        sf, df = FPTY[at], FPTY[dt]
        v = self.ftz_in(self.getf(ops[1], at), sf, ftz)
        if sf == df:
            if irnd:
                fn = {"rni": "roundeven", "rmi": "floor", "rpi": "ceil", "rzi": "trunc"}[irnd[0]]
                v = self.call_intr(sf, "llvm.%s.%s" % (fn, intname(sf)), ["%s %s" % (sf, v)])
            elif frnd:
                raise Unsupported(opc)
        else:
            v = self.cvt_ff(v, sf, df, frnd, mods)
        v = self.post(v, df, mods)
        self.writef(ops[0], v, dt)

    def mbarrier(self, opcode, ops):
        """mbarrier.* as an uninterpreted call (see mbar.py)."""
        k = mbar.ptx_key(opcode)
        dst = None
        if ops and not ops[0].strip().startswith("["):
            dst = ops[0].strip()
            ops = ops[1:]
        args = []
        for o in ops:
            o = o.strip()
            if o.startswith("["):
                args.append(self.address(o, "generic"))
            else:
                w = self.regs.get(o, 32)
                args.append("i%d %s" % (w, self.get(o, w)))
        if dst is None:
            rty = "void"
        elif dst == "_":
            # arrive on shared::cluster has no state result
            rty = "void" if "shared::cluster" in opcode else "i64"
        else:
            rty = "i%d" % self.regs[dst]
        from llvmsig import parse_type
        self.declare(k, "declare %s @%s(%s) nounwind" % (rty, k, ", ".join(parse_type(a)[0] for a in args)))
        if rty == "void":
            self.emit("call void @%s(%s)" % (k, ", ".join(args)))
        else:
            r = self.op("call %s @%s(%s)" % (rty, k, ", ".join(args)))
            if dst != "_":
                self.write(dst, r, int(rty[1:]))

    def generic_op(self, name, ops):
        """Instruction modelled through a probe-derived opaque call (probe.py)."""
        self.used_probe = True
        csig = self.decls[name]
        flat = probe.flatten_operands(ops)
        if flat is None:
            raise Unsupported("generic op operands")
        nd = 0 if csig.ret == "void" else csig.ret.count("i64")
        dsts, args = flat[:nd], flat[nd:]
        if len(args) != len(csig.args):
            raise Unsupported("generic op arity")
        vals = []
        for (kind, text), carg in zip(args, csig.args):
            if text.strip() in self.handle_params and carg.ty == "i64":
                vals.append(self.handle_params[text.strip()])
                continue
            if kind == "addr" and carg.ty == "i64":
                base = text.split("+")[0].strip()
                if base not in self.regs and "+" not in text and base in self.name_map:
                    # a texture/surface reference named directly: its handle is
                    # what the source obtains with texsurf.handle on the global
                    # (texsurf.handle(@g) is modelled as the address of @g,
                    # which is what PTX uses as the handle: mov.u64 %rd, g)
                    vals.append("i64 " + self.symbol_addr(base))
                    continue
                if base not in self.regs or "+" in text:
                    raise Unsupported("generic op narrow address")
                w = self.regs[base]
                vals.append("i64 " + self.resize(self.read_reg(base, w), w, 64, False))
            elif kind == "addr":
                p = self.address("[" + text + "]", "generic")
                if p.rsplit(" ", 1)[0] != carg.ty:
                    raise Unsupported("generic op pointer type")
                vals.append(p)
            else:
                cw = int(carg.ty[1:]) if re.fullmatch(r"i\d+", carg.ty) else 64
                if text in self.regs:
                    w = self.regs[text]
                    v = self.read_reg(text, w)
                    v = self.resize(v, w, cw, False)
                else:
                    v = self.get(text, cw)
                vals.append("i%d %s" % (cw, v))
        if nd == 0:
            self.emit("call void @%s(%s)" % (name, ", ".join(vals)))
            return
        r = self.op("call %s @%s(%s)" % (csig.ret, name, ", ".join(vals)))
        for q, (kind, d) in enumerate(dsts):
            if d == "_":
                continue
            e = self.op("extractvalue %s %s, %d" % (csig.ret, r, q))
            w = self.regs.get(d)
            if w is None:
                raise Unsupported("generic op destination " + d)
            self.write(d, self.resize(e, 64, w, False), w)

    def cvt_narrow(self, opcode, ops):
        """Narrow-format conversion as an uninterpreted function (narrowcvt)."""
        k = narrowcvt.ptx_key(opcode)
        if k is None:
            raise Unsupported(opcode)
        d = ops[0].strip()
        if d not in self.regs:
            raise Unsupported("narrow cvt destination " + d)
        w = self.regs[d]
        args, tys = [], []
        flat = []
        for o in ops[1:]:
            vo = self.vec_ops(o)
            flat.extend(vo if vo else [o.strip()])   # {a, b, c, d}: one arg per lane
        for o in flat:
            if o not in self.regs:
                raise Unsupported("narrow cvt operand " + o)
            ow = self.regs[o]
            args.append("i%d %s" % (ow, self.read_reg(o, ow)))
            tys.append("i%d" % ow)
        self.declare(k, "declare i%d @%s(%s) nounwind willreturn memory(none)" % (w, k, ", ".join(tys)))
        self.write(d, self.op("call i%d @%s(%s)" % (w, k, ", ".join(args))), w)

    def cvt_ff(self, v, sf, df, frnd, mods):
        order = {"half": 1, "bfloat": 1, "float": 2, "double": 3}
        if sf in ("half", "bfloat") and df in ("half", "bfloat"):
            # both convert exactly to f32, so this rounds once
            v = self.op("fpext %s %s to float" % (sf, v))
            return self.cvt_ff(v, "float", df, frnd, mods)
        if order[df] > order[sf]:
            return self.op("fpext %s %s to %s" % (sf, v, df))
        rm = frnd[0] if frnd else None
        if rm is None:
            raise Unsupported("narrowing cvt without rounding")
        if rm == "rn":
            return self.op("fptrunc %s %s to %s" % (sf, v, df))
        name = "llvm.experimental.constrained.fptrunc.%s.%s" % (FP_SCALAR[df], FP_SCALAR[sf])
        self.declare(name, "declare %s @%s(%s, metadata, metadata)" % (df, name, sf))
        return self.op('call %s @%s(%s %s, metadata !"%s", metadata !"fpexcept.ignore") strictfp'
                       % (df, name, sf, v, MODES[rm]))

    def cvt_ii(self, v, at, dt, sat):
        ab, db = TBITS[at], TBITS[dt]
        sa, sd = is_signed(at), is_signed(dt)
        if sat and not (dt.startswith("b") or at.startswith("b")):
            # clamp the value (interpreted per `at`) to the range of `dt`
            w = max(ab, db) + 1
            x = self.op("%s i%d %s to i%d" % ("sext" if sa else "zext", ab, v, w))
            lo = -(1 << (db - 1)) if sd else 0
            hi = (1 << (db - 1)) - 1 if sd else (1 << db) - 1
            x = self.call_intr("i%d" % w, "llvm.smax.i%d" % w, ["i%d %s" % (w, x), "i%d %d" % (w, lo)])
            x = self.call_intr("i%d" % w, "llvm.smin.i%d" % w, ["i%d %s" % (w, x), "i%d %d" % (w, hi)])
            return self.op("trunc i%d %s to i%d" % (w, x, db))
        return self.resize(v, ab, db, sa)

    # ---- memory ------------------------------------------------------------
    SPACES = ("global", "shared", "local", "const", "param", "generic",
              "shared::cta", "shared::cluster", "param::func", "param::entry")

    def mem_info(self, mods):
        vec = 1
        for m in mods:
            if m in ("v2", "v4", "v8"):
                vec = int(m[1:])
        t = [m for m in mods if m in TBITS]
        if len(t) != 1:
            raise Unsupported("memory type")
        space = [m for m in mods if m in self.SPACES]
        return t[0], vec, space[0] if space else "generic"

    def check_mem_mods(self, mods, opc):
        # .mmio only orders the access for memory-mapped I/O: a plain access
        # under single-thread semantics
        bad = [m for m in mods if m in ("async", "bulk", "multimem", "tensor")
               or m.startswith("unified") or m.startswith("mbarrier")]
        if bad:
            raise Unsupported(opc)

    def i_ld(self, mods, ops, opc):
        self.check_mem_mods(mods, opc)
        t, vec, space = self.mem_info(mods)
        eb = TBITS[t] if t != "pred" else 8
        p = self.address(ops[1], space)
        dsts = self.vec_ops(ops[0]) or [ops[0]]
        if len(dsts) != vec:
            raise Unsupported("vector arity " + opc)
        total = eb * vec
        d0 = dsts[0].strip()
        if vec == 1 and eb == 64 and self.base_of.get(d0) == ("dyn", d0):
            # load as a pointer (keeping provenance); its integer value is
            # the address (Alive2's assembly mode makes this exact)
            pv = self.op("load ptr, %s, align 8" % p)
            self.emit("store ptr %s, ptr %s" % (pv, self.dyn_slot(d0)))
            v = self.op("ptrtoint ptr %s to i64" % pv)
        elif vec > 1 and eb >= 8 and SCALARIZE:
            # a vector access as one access per element (Alive2 relates these
            # to the source's element accesses far more easily than a single
            # wide integer); the first keeps the whole vector's alignment
            for k, d in enumerate(dsts):
                q = self.lane_ptr(p, k * eb // 8)
                x = self.op("load i%d, %s, align %d" % (eb, q, total // 8 if k == 0 else
                                                        lane_align(total // 8, k * eb // 8)))
                if d != "_":
                    self.write(d, x, eb, is_signed(t))
            return
        else:
            v = self.op("load i%d, %s, align %d" % (total, p, total // 8))
        for k, d in enumerate(dsts):
            if d == "_":
                continue
            x = self.op("lshr i%d %s, %d" % (total, v, eb * k)) if k else v
            x = self.resize(x, total, eb, False)
            self.write(d, x, eb, is_signed(t))

    def lane_ptr(self, p, off):
        """Typed pointer operand p ("ptr %x") advanced by off bytes."""
        if not off:
            return p
        ty, v = p.rsplit(" ", 1)
        return "%s %s" % (ty, self.op("getelementptr i8, %s, i64 %d" % (p, off)))

    # ---- non-bulk asynchronous copy (cp.async) ------------------------------
    CP_ASYNC_MODS = {"async", "ca", "cg", "shared", "shared::cta", "global",
                     "L2::cache_hint", "L2::64B", "L2::128B", "L2::256B"}

    def i_cp(self, mods, ops, opc):
        """cp.async.{ca,cg}.shared{::cta}.global [d], [s], cp-size{, src-size |
        , ignore-src}{, cache-policy}: copy src-size bytes (default cp-size)
        and zero-fill the rest up to cp-size; ignore-src copies zeros.  The
        copy is modelled as completing at once: reading the destination
        before cp.async.wait_* is a data race in PTX, so this only removes
        behaviours.  commit_group/wait_group/wait_all are then no-ops."""
        ms = set(mods)
        import tma
        tg = tma.ptx(mods, ops)
        if tg is not None:
            toks, canon = tg
            vals, tys = [], []
            for o in canon:
                br = re.fullmatch(r"\[\s*([%\w$]+)\s*\]", o)
                r = br.group(1) if br else o
                if r not in self.regs:
                    raise Unsupported("operand " + opc)
                w = self.regs[r]
                vals.append("i%d %s" % (w, self.get(r, w)))
                tys.append("i%d" % w)
            kname = tma.name_of(toks) + "." + "_".join(tys)
            self.declare(kname, "declare void @%s(%s) nounwind" % (kname, ", ".join(tys)))
            self.emit("call void @%s(%s)" % (kname, ", ".join(vals)))
            return
        if "bulk" in ms and "tensor" not in ms:
            return self.cp_async_bulk(ms, ops, opc)
        if ms == {"async", "commit_group"} or ms == {"async", "wait_group"} or \
                ms == {"async", "wait_all"}:
            return
        if "async" not in ms or not ms <= self.CP_ASYNC_MODS or \
                len(ms & {"ca", "cg"}) != 1 or "global" not in ms or \
                not ms & {"shared", "shared::cta"}:
            raise Unsupported("opcode " + opc)
        rest = [o.strip() for o in ops[3:]]
        if "L2::cache_hint" in ms:
            if not rest:
                raise Unsupported("opcode " + opc)
            rest = rest[:-1]      # cache-policy: a hint only
        size = self.imm(ops[2].strip())
        if size not in (4, 8, 16) or ("cg" in ms and size != 16) or len(rest) > 1:
            raise Unsupported("opcode " + opc)
        d = self.address(ops[0], "shared")
        s = self.address(ops[1], "global")
        if not rest:
            n = self.const(size, 32)
        elif rest[0] in self.regs and self.regs[rest[0]] == 1:
            ign = self.get(rest[0], 1)
            n = self.op("select i1 %s, i32 0, i32 %d" % (ign, size))
        else:
            n = self.get(rest[0], 32)
        self.cp_async_copy(size, d, s, n)

    def cp_async_bulk(self, ms, ops, opc):
        """Non-tensor cp.async.bulk without multicast/masks/validation, as a
        copy completing at once (see i_cp), plus an opaque complete-tx on the
        mbarrier for the global/shared::cta -> shared forms.  Bulk prefetch
        and the bulk-group operations are then no-ops."""
        hint = "L2::cache_hint" in ms
        ms = ms - {"L2::cache_hint"}
        if ms in ({"async", "bulk", "commit_group"}, {"async", "bulk", "wait_group"},
                  {"async", "bulk", "wait_group", "read"}):
            return
        if ms == {"async", "bulk", "prefetch", "L2", "global"}:
            return
        ops = [o.strip() for o in ops]
        mb = "mbarrier::complete_tx::bytes"
        if ms in ({"async", "bulk", "shared::cluster", "global", mb},
                  {"async", "bulk", "shared::cta", "global", mb}) and \
                len(ops) == 4 + hint:
            dsp, ssp, mbar = "shared", "global", ops[3]
        elif ms == {"async", "bulk", "shared::cluster", "shared::cta", mb} and \
                not hint and len(ops) == 4:
            dsp, ssp, mbar = "shared", "shared", ops[3]
        elif ms == {"async", "bulk", "global", "shared::cta", "bulk_group"} and \
                len(ops) == 3 + hint:
            dsp, ssp, mbar = "global", "shared", None
        else:
            raise Unsupported("opcode " + opc)
        # operand order is [dst], [src] for every form; a bare register or
        # symbol in brackets
        d = self.address(ops[0], dsp)
        sp = self.address(ops[1], ssp)
        n = self.get(ops[2], 32)
        mp = self.address(mbar, "shared") if mbar else None
        lines, decls = bulk_copy_ir(d, sp, n, mp, self.tmp)
        for l in lines:
            self.emit(l)
        self.extra_decls.update(decls)

    def cp_async_copy(self, size, d, s, n):
        """d, s: typed pointer operands; n: i32 number of bytes to copy."""
        lines, decls = cp_async_ir(size, d, s, n, self.tmp)
        for l in lines:
            self.emit(l)
        self.extra_decls.update(decls)

    def st_async(self, mods, ops, opc):
        """st.async.shared::cluster.mbarrier::complete_tx::bytes{.vN}.T [a], v,
        [mbar]: a store, then an opaque complete-tx of its size on mbar (as
        for cp.async.bulk); st.async{.mmio}.release.{sys,gpu}.global.T [a], v:
        a store (single-thread semantics).  multimem forms are not modelled."""
        ms = [m for m in mods if m != "async"]
        rest = [m for m in ms if m in TBITS or m in ("v2", "v4")]
        other = set(ms) - set(rest)
        mb = "mbarrier::complete_tx::bytes"
        if other == {"shared::cluster", mb} and len(ops) == 3:
            self.i_st(["shared::cluster"] + rest, ops[:2], opc)
            t, vec, _ = self.mem_info(rest)
            mp = self.address(ops[2], "shared")
            self.declare("ptx.cp.async.bulk.complete_tx",
                         "declare void @ptx.cp.async.bulk.complete_tx(ptr, i32) nounwind")
            self.emit("call void @ptx.cp.async.bulk.complete_tx(%s, i32 %d)"
                      % (mp, TBITS[t] * vec // 8))
            return
        if other - {"mmio"} in ({"release", "sys", "global"}, {"release", "gpu", "global"}) \
                and not ("mmio" in other and "gpu" in other) and len(ops) == 2:
            return self.i_st(["global"] + rest, ops, opc)
        raise Unsupported("opcode " + opc)

    def i_discard(self, mods, ops, opc):
        """discard{.global}.L2 [a], 128: the 128-byte line at a (128-byte
        aligned) becomes indeterminate; modelled as a store of poison."""
        if mods not in (["global", "L2"], ["L2"]) or len(ops) != 2 or \
                self.imm(ops[1].strip()) != 128:
            raise Unsupported("opcode " + opc)
        p = self.address(ops[0], "global" if "global" in mods else "generic")
        self.emit(discard_ir(p))

    def i_tcgen05(self, mods, ops, opc):
        """tcgen05.ld/st as uninterpreted calls (see tcgen05ls.py); the
        other tcgen05 instructions are probe-derived."""
        import tcgen05ls
        import tcgen05ops
        import tcgen05mma
        mm_ = tcgen05mma.ptx(mods, ops)
        if mm_ is not None:
            toks, canon = mm_
            vals, tys = [], []
            for o in canon:
                if o.startswith("{"):
                    rs = self.vec_ops(o)
                    acc = "poison"
                    for k, r in enumerate(rs):
                        acc = self.op("insertelement <%d x i32> %s, i32 %s, i32 %d" % (
                            len(rs), acc, self.get(r, 32), k))
                    vals.append("<%d x i32> %s" % (len(rs), acc))
                    tys.append("v%di32" % len(rs))
                    continue
                br = re.fullmatch(r"\[\s*([%\w$]+)\s*\]", o)
                r = br.group(1) if br else o
                if r in self.regs:
                    w = self.regs[r]
                    vals.append("i%d %s" % (w, self.get(r, w)))
                    tys.append("i%d" % w)
                elif self.imm(r) is not None:
                    vals.append("i64 %d" % self.imm(r))
                    tys.append("i64")
                else:
                    raise Unsupported("operand " + opc)
            kname = tcgen05mma.name_of(toks).replace("::", "__") + "." + "_".join(tys)
            self.declare(kname, "declare void @%s(%s) nounwind" % (kname, ", ".join(
                "<%s x i32>" % t[1:-3] if t.startswith("v") else t for t in tys)))
            self.emit("call void @%s(%s)" % (kname, ", ".join(vals)))
            return
        rr = tcgen05ls.red_ptx(mods)
        if rr is not None:
            shape, num, ty, op, ab, nan = rr
            n = tcgen05ls.nregs(shape, num)
            kname = tcgen05ls.red_key(*rr)
            self.declare(kname, tcgen05ls.red_decl(*rr))
            ops_ = [o.strip() for o in ops]
            regs = self.vec_ops(ops_[0])
            mm = re.fullmatch(r"\[\s*([%\w$]+)\s*\]", ops_[2]) if len(ops_) >= 3 else None
            if regs is None or len(regs) != n or not mm:
                raise Unsupported("operands " + opc)
            args = ["i32 %s" % self.get(mm.group(1), 32)]
            if shape == "16x32bx2":
                if len(ops_) != 4:
                    raise Unsupported("operands " + opc)
                args.append("i64 %s" % self.get(ops_[3], 64))
            st = "{ <%d x i32>, i32 }" % n
            v = self.op("call %s @%s(%s)" % (st, kname, ", ".join(args)))
            vec = self.op("extractvalue %s %s, 0" % (st, v))
            for k, r in enumerate(regs):
                self.write(r, self.op("extractelement <%d x i32> %s, i32 %d" % (n, vec, k)), 32)
            self.write(ops_[1], self.op("extractvalue %s %s, 1" % (st, v)), 32)
            return
        t = tcgen05ls.ptx(mods)
        if t is None:
            toks = tcgen05ops.ptx(mods)
            if toks is None:
                raise Unsupported("opcode " + opc)
            # operands as integers of their register width (addresses too)
            vals, tys = [], []
            for o in [o.strip() for o in ops]:
                mm = re.fullmatch(r"\[\s*([%\w$]+)\s*\]", o)
                r = mm.group(1) if mm else o
                if r not in self.regs or self.regs[r] == 1:
                    raise Unsupported("operand " + opc)
                w = self.regs[r]
                vals.append("i%d %s" % (w, self.get(r, w)))
                tys.append("i%d" % w)
            kname = tcgen05ops.name_of(toks) + ("." + "_".join(tys) if tys else "")
            self.declare(kname, "declare void @%s(%s) nounwind" % (kname, ", ".join(tys)))
            self.emit("call void @%s(%s)" % (kname, ", ".join(vals)))
            return
        kind, shape, num, packed = t
        n = tcgen05ls.nregs(shape, num)
        name = tcgen05ls.key(kind, shape, num, packed)
        self.declare(name, tcgen05ls.decl(kind, shape, num, packed))
        ops = [o.strip() for o in ops]
        # ld {regs}, [a]{, off};   st [a]{, off}, {regs}
        regs = self.vec_ops(ops[0] if kind == "ld" else ops[-1])
        addr = ops[1] if kind == "ld" else ops[0]
        m = re.fullmatch(r"\[\s*([%\w$]+)\s*\]", addr)
        if not m or regs is None or len(regs) != n:
            raise Unsupported("operands " + opc)
        args = ["i32 %s" % self.get(m.group(1), 32)]
        if shape == "16x32bx2":
            if len(ops) != 3:
                raise Unsupported("operands " + opc)
            args.append("i64 %s" % self.get(ops[2] if kind == "ld" else ops[1], 64))
        vt = "<%d x i32>" % n
        if kind == "ld":
            v = self.op("call %s @%s(%s)" % (vt, name, ", ".join(args)))
            for k, r in enumerate(regs):
                self.write(r, self.op("extractelement %s %s, i32 %d" % (vt, v, k)), 32)
        else:
            acc = "poison"
            for k, r in enumerate(regs):
                acc = self.op("insertelement %s %s, i32 %s, i32 %d" % (vt, acc, self.get(r, 32), k))
            self.emit("call void @%s(%s)" % (name, ", ".join(args + ["%s %s" % (vt, acc)])))

    def i_wgmma(self, mods, ops, opc):
        """wgmma.fence / commit_group / wait_group (.sync.aligned) only order
        the asynchronous warpgroup MMA (itself opaque): no-ops under
        single-thread semantics, like the other fences."""
        if mods in (["fence", "sync", "aligned"], ["commit_group", "sync", "aligned"],
                    ["wait_group", "sync", "aligned"]):
            return
        raise Unsupported("opcode " + opc)

    def i_multimem(self, mods, ops, opc):
        """multimem.st.async.release.{sys,gpu}.global.T [a], v: a store to every
        replica of a multimem address; in the single-memory model, a store to
        a (the same as the source's st.async with isMultimem).  multimem
        loads/reductions combine replicas and are not modelled."""
        rest = [m for m in mods if m in TBITS]
        other = set(mods) - set(rest)
        if other in ({"st", "async", "release", "sys", "global"},
                     {"st", "async", "release", "gpu", "global"}) and len(ops) == 2:
            return self.i_st(["global"] + rest, ops, opc)
        raise Unsupported("opcode " + opc)

    def st_bulk(self, mods, ops, opc):
        """st.bulk{.weak}{.shared::cta} [a], size, 0: zero `size` bytes
        (size a multiple of 8, a 8-byte aligned; initval must be 0)."""
        if not set(mods) <= {"bulk", "weak", "shared::cta"} or len(ops) != 3 or \
                self.imm(ops[2].strip()) != 0:
            raise Unsupported("opcode " + opc)
        p = self.address(ops[0], "shared" if "shared::cta" in mods else "generic")
        sz = ops[1].strip()
        if sz in self.regs and self.regs[sz] == 32:
            n = self.resize(self.get(sz, 32), 32, 64, False)
        else:
            n = self.get(sz, 64)
        lines, decls = st_bulk_ir(p, n, self.tmp)
        for l in lines:
            self.emit(l)
        self.extra_decls.update(decls)

    def i_spdecompress(self, mods, ops, opc):
        """spdecompress.elemsize.idxsize.sp::X:Y.xN {data}, {mdata}, {cdata}
        (see sparse.py)."""
        import sparse
        if len(mods) != 4 or mods[0] not in ("b8", "b16") or mods[1] not in ("b2", "b4") \
                or not sparse.SPF.fullmatch(mods[2]) or not re.fullmatch(r"x\d+", mods[3]):
            raise Unsupported("opcode " + opc)
        E, I = int(mods[0][1:]), int(mods[1][1:])
        X, Y = map(int, sparse.SPF.fullmatch(mods[2]).groups())
        num = int(mods[3][1:])
        if not sparse.check(E, I, X, Y, num):
            raise Unsupported("opcode " + opc)
        bundles = [self.vec_ops(o) or [o.strip()] for o in ops[:3]]
        if len(ops) != 3 or [len(b) for b in bundles] != [
                sparse.regs(Y * E * num), sparse.regs(X * I * num), sparse.regs(X * E * num)]:
            raise Unsupported("operand count " + opc)

        m = [self.get(r, 32) for r in bundles[1]]
        c = [self.get(r, 32) for r in bundles[2]]
        lines, v = sparse.spdecompress_ir(E, I, X, Y, num, m, c, self.tmp)
        for l in lines:
            self.emit(l)
        D = Y * E * num // 32
        iv = self.op("bitcast <%d x i%d> %s to <%d x i32>" % (Y * num, E, v, D))
        for k, r in enumerate(bundles[0]):
            self.write(r, self.op("extractelement <%d x i32> %s, i32 %d" % (D, iv, k)), 32)

    # stacksave/stackrestore: opaque calls, matching the source's
    # llvm.stacksave/llvm.stackrestore (renamed by tv.py)
    def i_stacksave(self, mods, ops, opc):
        if mods not in (["u64"], ["u32"]):
            raise Unsupported("opcode " + opc)
        self.declare("ptx.stacksave", "declare ptr @ptx.stacksave() nounwind")
        p = self.op("call ptr @ptx.stacksave()")
        bits = 64 if mods == ["u64"] else 32
        v = self.op("ptrtoint ptr %s to i%d" % (p, bits))
        self.write(ops[0], v, bits)

    def i_stackrestore(self, mods, ops, opc):
        if mods not in (["u64"], ["u32"]):
            raise Unsupported("opcode " + opc)
        self.declare("ptx.stackrestore", "declare void @ptx.stackrestore(ptr) nounwind")
        p = self.address("[%s]" % ops[0].strip(), "local")
        self.emit("call void @ptx.stackrestore(%s)" % p)

    def i_ldu(self, mods, ops, opc):
        self.i_ld(mods, ops, opc)

    def i_st(self, mods, ops, opc):
        if "bulk" in mods:
            return self.st_bulk(mods, ops, opc)
        if "async" in mods:
            return self.st_async(mods, ops, opc)
        self.check_mem_mods(mods, opc)
        t, vec, space = self.mem_info(mods)
        eb = TBITS[t]
        p = self.address(ops[0], space)
        srcs = self.vec_ops(ops[1]) or [ops[1]]
        m = re.fullmatch(r"\[\s*([\w$]+)\s*\]", ops[0].strip())
        if m and len(srcs) == 1 and eb == 64:
            try:
                slot = self.lookup(m.group(1))[0]
                key = self.base_of.get(srcs[0].strip())
                if key is not None:
                    self.slot_base[slot] = key
                else:
                    self.slot_base.pop(slot, None)
            except Unsupported:
                pass
        if len(srcs) != vec:
            raise Unsupported("vector arity " + opc)
        total = eb * vec
        if vec > 1 and eb >= 8 and SCALARIZE:
            for k, s_ in enumerate(srcs):
                q = self.lane_ptr(p, k * eb // 8)
                self.emit("store i%d %s, %s, align %d" % (
                    eb, self.get(s_, eb), q, total // 8 if k == 0 else lane_align(total // 8, k * eb // 8)))
            return
        acc = None
        for k, s in enumerate(srcs):
            x = self.resize(self.get(s, eb), eb, total, False)
            if k:
                x = self.op("shl i%d %s, %d" % (total, x, eb * k))
            acc = x if acc is None else self.op("or i%d %s, %s" % (total, acc, x))
        self.emit("store i%d %s, %s, align %d" % (total, acc, p, total // 8))

    def atomic_rmw(self, mods, ops, opc, has_result):
        self.check_mem_mods(mods, opc)
        t, vec, space = self.mem_info(mods)
        if vec != 1:
            raise Unsupported(opc)
        bits = TBITS[t]
        it = "i%d" % bits
        ops_ = ["add", "min", "max", "inc", "dec", "and", "or", "xor", "exch", "cas"]
        kind = [m for m in mods if m in ops_]
        if not kind:
            raise Unsupported(opc)
        kind = kind[0]
        o = ops if has_result else ["_"] + ops
        p = self.address(o[1], space)
        old = self.op("load %s, %s, align %d" % (it, p, bits // 8))
        b = self.get(o[2], bits)
        if kind == "cas":
            c = self.get(o[3], bits)
            eq = self.op("icmp eq %s %s, %s" % (it, old, b))
            new = self.op("select i1 %s, %s %s, %s %s" % (eq, it, c, it, old))
        elif kind == "exch":
            new = b
        elif is_float(t):
            fty = FPTY[t]
            fo = self.op("bitcast %s %s to %s" % (it, old, fty))
            fb = self.op("bitcast %s %s to %s" % (it, b, fty))
            if kind == "add":
                fr = self.op("fadd %s %s, %s" % (fty, fo, fb))
            elif kind in ("min", "max"):
                fr = self.call_intr(fty, "llvm.%snum.%s" % (kind, intname(fty)), ["%s %s" % (fty, fo), "%s %s" % (fty, fb)])
            else:
                raise Unsupported(opc)
            new = self.op("bitcast %s %s to %s" % (fty, fr, it))
        elif kind == "add":
            new = self.op("add %s %s, %s" % (it, old, b))
        elif kind in ("and", "or", "xor"):
            new = self.op("%s %s %s, %s" % (kind, it, old, b))
        elif kind in ("min", "max"):
            fn = ("s" if is_signed(t) else "u") + kind
            new = self.call_intr(it, "llvm.%s.%s" % (fn, it), ["%s %s" % (it, old), "%s %s" % (it, b)])
        elif kind == "inc":
            ge = self.op("icmp uge %s %s, %s" % (it, old, b))
            inc = self.op("add %s %s, 1" % (it, old))
            new = self.op("select i1 %s, %s 0, %s %s" % (ge, it, it, inc))
        elif kind == "dec":
            z = self.op("icmp eq %s %s, 0" % (it, old))
            gt = self.op("icmp ugt %s %s, %s" % (it, old, b))
            c = self.op("or i1 %s, %s" % (z, gt))
            dec = self.op("sub %s %s, 1" % (it, old))
            new = self.op("select i1 %s, %s %s, %s %s" % (c, it, b, it, dec))
        else:
            raise Unsupported(opc)
        self.emit("store %s %s, %s, align %d" % (it, new, p, bits // 8))
        if has_result:
            self.write(o[0], old, bits)

    def i_atom(self, mods, ops, opc):
        self.atomic_rmw(mods, ops, opc, True)

    def i_red(self, mods, ops, opc):
        self.atomic_rmw(mods, ops, opc, False)

    def i_prefetch(self, mods, ops, opc):
        pass

    def i_prefetchu(self, mods, ops, opc):
        pass

    # ---- calls -------------------------------------------------------------
    def i_call(self, mods, ops, opc):
        rets = []
        if ops and ops[0].startswith("("):
            rets = [x.strip() for x in ops[0][1:-1].split(",") if x.strip()]
            ops = ops[1:]
        callee = ops[0].strip()
        args = [x.strip() for x in ops[1][1:-1].split(",") if x.strip()] \
            if len(ops) > 1 and ops[1].startswith("(") else []
        indirect = callee.startswith("%")
        if indirect:
            key = self.base_of.get(callee)
            target = None
            if key is not None and key[0] == "sym" and callee in self.exact and \
                    key[1] in self.name_map:
                target = self.name_map[key[1]]     # mov %rd, func: a known function
            cands = [x for x in self.indirect_sigs
                     if getattr(x, "callee", None) == target and len(self.sized_args(x.args)) == len(args)
                     and (x.ret == "void" or self.zero_sized(x.ret)) == (not rets)]
            sigs = {(x.ret, tuple(a.ty for a in x.args), x.vararg): x for x in cands}
            if not sigs and target is not None and target in self.decls and \
                    self.decls[target].vararg:
                # a variadic function called through a prototype: the PTX has
                # the fixed arguments plus the buffer; use the call site's
                # return type (it may differ from the declaration's)
                import copy
                dsig = self.decls[target]
                rts = {x.ret for x in self.indirect_sigs if getattr(x, "callee", None) == target}
                if len(rts) == 1:
                    csig = copy.copy(dsig)
                    csig.ret = next(iter(rts))
                    csig.ret_attrs = ""
                    return self.vararg_call(target, csig, rets, args)
            if len(sigs) != 1:
                raise Unsupported("indirect call with ambiguous source signature")
            csig = next(iter(sigs.values()))
            if csig.vararg:
                raise Unsupported("vararg")
            if target is not None:
                fname = "@" + target
            else:
                fv = self.get(callee, 64)
                if key is not None and callee in self.exact:
                    b = self.base_ptr(key)
                    fp = "%s %s" % b if b else None
                else:
                    fp = self.rebase(key, fv) if key is not None else None
                fname = fp.rsplit(" ", 1)[1] if fp else self.op("inttoptr i64 %s to ptr" % fv)
        else:
            if len(ops) > 2:
                raise Unsupported("call with prototype")
            fname = self.name_map.get(callee)
            if fname is None or fname not in self.decls:
                raise Unsupported("unknown callee " + callee)
            csig = self.decls[fname]
            if csig.vararg:
                return self.vararg_call(fname, csig, rets, args)
        cargs = self.sized_args(csig.args)
        if csig.vararg or len(args) != len(cargs):
            raise Unsupported("call arity")
        vals = []
        for a, carg in zip(args, cargs):
            slot, pp = self.lookup(a)
            if carg.byval_type():
                vals.append("%s %s %s" % (carg.ty, carg.attrs, slot))
                continue
            if carg.ty.startswith("ptr"):
                v = self.ptr_from_slot(carg.ty, slot)
            else:
                v = self.load_value(carg.ty, slot, pp)
            vals.append("%s %s %s" % (carg.ty, carg.attrs, v))
        if len(cargs) != len(csig.args):
            it = iter(vals)
            vals = [next(it) if a in cargs else "%s zeroinitializer" % a.ty for a in csig.args]
        self.callees.append(fname)
        call = "call %s %s(%s)" % (csig.ret, fname if indirect else "@" + fname, ", ".join(vals))
        if indirect and fname.startswith("@"):
            self.callees.append(fname[1:])
        if csig.ret == "void":
            if rets:
                raise Unsupported("void call with retval")
            self.emit(call)
            return
        if not rets and self.zero_sized(csig.ret):
            self.op(call)
            return
        r = self.op(call)
        if len(rets) != 1:
            raise Unsupported("call without retval param")
        slot, pp = self.lookup(rets[0])
        from llvmsig import Arg
        self.store_value(csig.ret, r, Arg(csig.ret, csig.ret_attrs, None), slot, pp, True)

    def vararg_call(self, fname, csig, rets, args):
        """Call to a variadic function: NVPTX passes the variable arguments in
        a buffer (natural alignment, in order) whose address is the last
        parameter.  The variable arguments' types come from the source call."""
        fixed = self.sized_args(csig.args)
        if len(args) != len(fixed) + 1:
            raise Unsupported("vararg call arity")
        # the source call sites of fname, in order (all, including those
        # without variable arguments)
        sites = [x for x in self.indirect_sigs if getattr(x, "callee", None) == fname
                 and len(x.args) >= len(csig.args)]
        kinds = {tuple(a.ty for a in x.args) for x in sites}
        if not hasattr(self, "vcall_idx"):
            self.vcall_idx = {}
        k = self.vcall_idx.get(fname, 0)
        self.vcall_idx[fname] = k + 1
        if len(kinds) == 1:
            site = sites[0]
        elif k < len(sites):
            # different variable argument types at different call sites:
            # match the k-th call in the PTX with the k-th in the source
            # (calls have side effects, so their order is preserved; a wrong
            # match gives a counterexample, not a proof)
            site = sites[k]
        else:
            raise Unsupported("vararg call with ambiguous source types")
        vtys = [a.ty for a in site.args[len(csig.args):]]
        vals = []
        for a, carg in zip(args, fixed):
            slot, pp = self.lookup(a)
            if carg.byval_type():
                vals.append("%s %s %s" % (carg.ty, carg.attrs, slot))
                continue
            if carg.ty.startswith("ptr"):
                v = self.ptr_from_slot(carg.ty, slot)
            else:
                v = self.load_value(carg.ty, slot, pp)
            vals.append("%s %s %s" % (carg.ty, carg.attrs, v))
        bslot, _ = self.lookup(args[-1])
        buf = self.ptr_from_slot("ptr", bslot)
        st = layout.parse("{ %s }" % ", ".join(vtys), self.typedefs) if vtys else None
        for t, off in zip(vtys, layout.offsets(st) if vtys else []):
            p = self.gep(buf, off)
            vals.append("%s %s" % (t, self.load_mem(layout.parse(t, self.typedefs), p, 0)))
        fty = "%s (%s)" % (csig.ret, ", ".join([a.ty for a in csig.args] + ["..."]))
        self.callees.append(fname)
        call = "call %s @%s(%s)" % (fty, fname, ", ".join(vals))
        if csig.ret == "void":
            self.emit(call)
            return
        r = self.op(call)
        slot, pp = self.lookup(rets[0])
        from llvmsig import Arg
        self.store_value(csig.ret, r, Arg(csig.ret, csig.ret_attrs, None), slot, pp, True)

    def lookup(self, sym):
        for sc in reversed(self.scopes):
            if sym in sc:
                return sc[sym]
        raise Unsupported("unknown param " + sym)

    # ---- cross-thread operations: uninterpreted calls ------------------------
    # These have no single-thread semantics; they are modelled as calls to
    # the (renamed) NVVM intrinsic they implement, so the check verifies that
    # the backend emitted the right instruction with the right operands.
    def ucall(self, ret, name, args, attrs="nounwind"):
        fname = "nvvm." + name
        from llvmsig import parse_type
        self.declare(fname, "declare %s @%s(%s) %s" % (
            ret, fname, ", ".join(parse_type(a)[0] for a in args), attrs))
        if ret == "void":
            self.emit("call void @%s(%s)" % (fname, ", ".join(args)))
            return None
        return self.op("call %s @%s(%s)" % (ret, fname, ", ".join(args)))

    def pred_operand(self, s):
        s = s.strip()
        if s.startswith("!"):
            return self.op("xor i1 %s, true" % self.get(s[1:], 1))
        return self.get(s, 1)

    def i_bar(self, mods, ops, opc):
        self.barrier(mods, ops, opc, True)

    def i_barrier(self, mods, ops, opc):
        self.barrier(mods, ops, opc, "aligned" in mods)

    def barrier(self, mods, ops, opc, aligned):
        al = ".aligned" if aligned else ""
        if mods[:2] == ["warp", "sync"]:
            self.ucall("void", "bar.warp.sync", ["i32 " + self.get(ops[0], 32)])
            return
        kind = mods[0] if mods else "sync"
        if kind == "red":
            return self.barrier_red(mods, ops, opc, al)
        if kind not in ("sync", "arrive"):
            raise Unsupported(opc)
        a = self.get(ops[0], 32)
        if len(ops) == 1:
            if kind != "sync":
                raise Unsupported(opc)
            self.ucall("void", "barrier.cta.sync%s.all" % al, ["i32 " + a])
        else:
            b = self.get(ops[1], 32)
            self.ucall("void", "barrier.cta.%s%s.count" % (kind, al), ["i32 " + a, "i32 " + b])

    def barrier_red(self, mods, ops, opc, al):
        # bar.red.{and,or}.pred d, a{, b}, {!}c   /  bar.red.popc.u32 d, a{, b}, {!}c
        op = [m for m in mods if m in ("and", "or", "popc")]
        if not op:
            raise Unsupported(opc)
        op = op[0]
        a = self.get(ops[1], 32)
        args = ["i32 " + a]
        if len(ops) == 4:
            args.append("i32 " + self.get(ops[2], 32))
        args.append("i1 " + self.pred_operand(ops[-1]))
        name = "barrier.cta.red.%s%s.%s" % (op, al, "count" if len(ops) == 4 else "all")
        if op == "popc":
            self.write(ops[0], self.ucall("i32", name, args), 32)
        else:
            self.write(ops[0], self.ucall("i1", name, args), 1)

    def i_movmatrix(self, mods, ops, opc):
        # movmatrix.sync.aligned.m8n8.trans.b16 d, a  (cross-lane data movement)
        name = "movmatrix." + ".".join(mods)
        self.write(ops[0], self.ucall("i32", name, ["i32 " + self.get(ops[1], 32)]), 32)

    def i_fns(self, mods, ops, opc):
        args = ["i32 " + self.get(o, 32) for o in ops[1:4]]
        self.write(ops[0], self.ucall("i32", "fns", args, "nounwind willreturn memory(none)"), 32)

    def i_setmaxnreg(self, mods, ops, opc):
        kind = [m for m in mods if m in ("inc", "dec")]
        if not kind:
            raise Unsupported(opc)
        self.ucall("void", "setmaxnreg.%s.sync.aligned.u32" % kind[0], ["i32 " + self.get(ops[0], 32)])

    def i_shfl(self, mods, ops, opc):
        sync = "sync" in mods
        mode = [m for m in mods if m in ("up", "down", "bfly", "idx")][0]
        dsts = ops[0].split("|")
        args = [self.get(ops[1], 32), self.get(ops[2], 32), self.get(ops[3], 32)]
        if sync:
            args = [self.get(ops[4], 32)] + args
        name = "shfl.%s%s.i32%s" % ("sync." if sync else "", mode, "p" if len(dsts) > 1 else "")
        argv = ["i32 " + a for a in args]
        if len(dsts) > 1:
            r = self.ucall("{ i32, i1 }", name, argv)
            v = self.op("extractvalue { i32, i1 } %s, 0" % r)
            p = self.op("extractvalue { i32, i1 } %s, 1" % r)
            self.write(dsts[0].strip(), v, 32)
            self.write(dsts[1].strip(), p, 1)
        else:
            self.write(dsts[0].strip(), self.ucall("i32", name, argv), 32)

    def i_vote(self, mods, ops, opc):
        sync = "sync" in mods
        mode = [m for m in mods if m in ("all", "any", "uni", "ballot")][0]
        a = self.pred_operand(ops[1])
        args = ["i1 " + a]
        if sync:
            args = ["i32 " + self.get(ops[2], 32)] + args
        name = "vote.%s%s" % (mode, ".sync" if sync else "")
        if mode == "ballot":
            self.write(ops[0], self.ucall("i32", name, args), 32)
        else:
            self.write(ops[0], self.ucall("i1", name, args), 1)

    def i_activemask(self, mods, ops, opc):
        self.write(ops[0], self.ucall("i32", "activemask", []), 32)

    def i_redux(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        kind = [m for m in mods if m in ("add", "min", "max", "and", "or", "xor")]
        if t == "f32" and kind and kind[0] in ("min", "max"):
            name = "redux.sync.f" + kind[0] + (".abs" if "abs" in mods else "") + \
                (".NaN" if "NaN" in mods else "")
            a = self.getf(ops[1], "f32")
            fname = "nvvm." + name
            self.declare(fname, "declare float @%s(float, i32) nounwind" % fname)
            r = self.op("call float @%s(float %s, i32 %s)" % (fname, a, self.get(ops[2], 32)))
            self.writef(ops[0], r, "f32")
            return
        if not kind or t not in ("u32", "s32", "b32"):
            raise Unsupported(opc)
        k = kind[0]
        if k in ("min", "max") and t == "u32":
            k = "u" + k
        r = self.ucall("i32", "redux.sync." + k,
                       ["i32 " + self.get(ops[1], 32), "i32 " + self.get(ops[2], 32)])
        self.write(ops[0], r, 32)

    def i_match(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        it = "i%d" % bits
        a = self.get(ops[1], bits)
        m = self.get(ops[2], 32)
        if "any" in mods:
            self.write(ops[0], self.ucall("i32", "match.any.sync." + it, ["i32 " + m, "%s %s" % (it, a)]), 32)
        else:
            dsts = ops[0].split("|")
            r = self.ucall("{ i32, i1 }", "match.all.sync.%sp" % it, ["i32 " + m, "%s %s" % (it, a)])
            self.write(dsts[0].strip(), self.op("extractvalue { i32, i1 } %s, 0" % r), 32)
            if len(dsts) > 1:
                self.write(dsts[1].strip(), self.op("extractvalue { i32, i1 } %s, 1" % r), 1)

    def i_szext(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        a, b = self.get(ops[1], 32), self.get(ops[2], 32)
        if "wrap" in mods:
            n = self.op("and i32 %s, 31" % b)
        else:
            n = self.call_intr("i32", "llvm.umin.i32", ["i32 " + b, "i32 32"])
        z = self.op("icmp eq i32 %s, 0" % n)
        full = self.op("icmp uge i32 %s, 32" % n)
        nn = self.op("select i1 %s, i32 1, i32 %s" % (self.op("or i1 %s, %s" % (z, full)), n))
        sh = self.op("sub i32 32, %s" % nn)
        x = self.op("shl i32 %s, %s" % (a, sh))
        x = self.op("%s i32 %s, %s" % ("ashr" if is_signed(t) else "lshr", x, sh))
        x = self.op("select i1 %s, i32 %s, i32 %s" % (full, a, x))
        self.write(ops[0], self.op("select i1 %s, i32 0, i32 %s" % (z, x)), 32)

    def i_bmsk(self, mods, ops, opc):
        a, b = self.get(ops[1], 32), self.get(ops[2], 32)
        if "wrap" in mods:
            a = self.op("and i32 %s, 31" % a)
            b = self.op("and i32 %s, 31" % b)
        else:
            a = self.call_intr("i32", "llvm.umin.i32", ["i32 " + a, "i32 32"])
            b = self.call_intr("i32", "llvm.umin.i32", ["i32 " + b, "i32 32"])
        a64 = self.op("zext i32 %s to i64" % a)
        b64 = self.op("zext i32 %s to i64" % b)
        m = self.op("shl i64 1, %s" % b64)
        m = self.op("sub i64 %s, 1" % m)
        m = self.op("shl i64 %s, %s" % (m, a64))
        self.write(ops[0], self.op("trunc i64 %s to i32" % m), 32)

    def i_isspacep(self, mods, ops, opc):
        space = [m for m in mods if m in ("global", "shared", "local", "const", "param")]
        if "shared::cluster" in mods:
            space = ["shared.cluster"]
        elif "shared::cta" in mods:
            space = ["shared"]
        if not space:
            raise Unsupported(opc)
        a = self.get(ops[1], 64)
        key = self.base_of.get(ops[1].strip())
        p = self.rebase(key, a) if key is not None else None
        p = p.rsplit(" ", 1)[1] if p and p.startswith("ptr ") else \
            self.op("inttoptr i64 %s to ptr" % a)
        self.write(ops[0], self.ucall("i1", "isspacep." + space[0], ["ptr " + p]), 1)

    def i_elect(self, mods, ops, opc):
        dsts = ops[0].split("|")
        r = self.ucall("{ i32, i1 }", "elect.sync", ["i32 " + self.get(ops[1], 32)])
        if dsts[0].strip() != "_":
            self.write(dsts[0].strip(), self.op("extractvalue { i32, i1 } %s, 0" % r), 32)
        if len(dsts) > 1:
            self.write(dsts[1].strip(), self.op("extractvalue { i32, i1 } %s, 1" % r), 1)

    def i_brx(self, mods, ops, opc):
        if "idx" not in mods:
            raise Unsupported(opc)
        targets = self.jump_tables.get(ops[1].strip())
        if not targets:
            raise Unsupported("brx.idx without .branchtargets")
        i = self.get(ops[0], 32)
        dflt = "brxbad%d" % self.n
        self.n += 1
        cases = " ".join("i32 %d, label %%%s" % (k, self.label(t)) for k, t in enumerate(targets))
        self.emit("switch i32 %s, label %%%s [ %s ]" % (i, dflt, cases))
        # an index outside the table is undefined behaviour
        self.new_block(dflt)
        self.emit("unreachable")

    def dyn_slot(self, reg):
        slot = "%%dyn.%s" % re.sub(r"[^\w.]", "_", reg.lstrip("%"))
        if slot not in self.slots_used:
            self.slots_used.add(slot)
            self.allocas.append("%s = alloca ptr" % slot)
        return slot

    def i_alloca(self, mods, ops, opc):
        t = self.types_of(mods)[0]
        bits = TBITS[t]
        size = self.get(ops[1], bits)
        al = self.imm(ops[2]) if len(ops) > 2 else 1
        p = self.op("alloca i8, i%d %s, align %d" % (bits, size, al))
        self.emit("store ptr %s, ptr %s" % (p, self.dyn_slot(ops[0].strip())))
        self.write(ops[0], self.op("ptrtoint ptr %s to i%d" % (p, bits)), bits)

    def i_clmad(self, mods, ops, opc):
        a, b, c = (self.get(o, 64) for o in ops[1:4])
        if "lo" in mods:
            p = self.call_intr("i64", "llvm.clmul.i64", ["i64 " + a, "i64 " + b])
        else:
            wa = self.op("zext i64 %s to i128" % a)
            wb = self.op("zext i64 %s to i128" % b)
            w = self.call_intr("i128", "llvm.clmul.i128", ["i128 " + wa, "i128 " + wb])
            w = self.op("lshr i128 %s, 64" % w)
            p = self.op("trunc i128 %s to i64" % w)
        self.write(ops[0], self.op("xor i64 %s, %s" % (p, c)), 64)

    def i_nanosleep(self, mods, ops, opc):
        self.ucall("void", "nanosleep", ["i32 " + self.get(ops[0], 32)])


# PTXTV_SCALARIZE=1: vector ld/st as one access per element (tv.py does the
# same to the source); a fallback for timeouts, since it helps when source
# and PTX use the same element width and hurts when they do not
SCALARIZE = os.environ.get("PTXTV_SCALARIZE", "0") == "1"


def lane_align(A, off):
    a = A
    while off % a:
        a //= 2
    return a


def cp_async_ir(size, d, s, n, fresh):
    """IR for one cp.async of cp-size `size`: n (an i32) bytes from s, zeros
    up to `size`, both addresses aligned to `size`, n <= size.  d, s are
    typed pointer operands ("ptr %x"); fresh() returns a new value name."""
    L, decls = [], {"llvm.assume": "declare void @llvm.assume(i1)"}

    def op(text):
        t = fresh()
        L.append("%s = %s" % (t, text))
        return t
    ok = op("icmp ule i32 %s, %d" % (n, size))
    L.append("call void @llvm.assume(i1 %s)" % ok)            # src-size <= cp-size
    for p in (d, s):                                            # aligned to cp-size
        a = op("ptrtoint %s to i64" % p)
        z = op("icmp eq i64 %s, 0" % op("and i64 %s, %d" % (a, size - 1)))
        L.append("call void @llvm.assume(i1 %s)" % z)
    # bytes [0, n) from s, zeros after: one masked byte-vector load
    vt = "<%d x i8>" % size
    mname = "llvm.masked.load.v%di8.p0" % size
    decls[mname] = "declare %s @%s(ptr, <%d x i1>, %s)" % (vt, mname, size, vt)
    ins = op("insertelement <%d x i32> poison, i32 %s, i64 0" % (size, n))
    spl = op("shufflevector <%d x i32> %s, <%d x i32> poison, <%d x i32> zeroinitializer"
             % (size, ins, size, size))
    idx = ", ".join("i32 %d" % k for k in range(size))
    mask = op("icmp ult <%d x i32> <%s>, %s" % (size, idx, spl))
    sp = s.split(" ", 1)
    v = op("call %s @%s(%s align %d %s, <%d x i1> %s, %s zeroinitializer)"
           % (vt, mname, sp[0], size, sp[1], size, mask, vt))
    L.append("store %s %s, %s, align %d" % (vt, v, d, size))
    return L, decls


def bulk_copy_ir(d, s, n, mbar, fresh):
    """IR for a non-tensor cp.async.bulk: copy n (an i32, a multiple of 16)
    bytes from s to d (both 16-byte aligned), then, if mbar is given, an
    opaque complete-tx of n bytes on it."""
    L = []
    decls = {"llvm.assume": "declare void @llvm.assume(i1)",
             "llvm.memcpy.p0.p0.i64": "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)"}

    def op(text):
        t = fresh()
        L.append("%s = %s" % (t, text))
        return t
    n64 = op("zext i32 %s to i64" % n)
    for v in (n64, op("ptrtoint %s to i64" % d), op("ptrtoint %s to i64" % s)):
        z = op("icmp eq i64 %s, 0" % op("and i64 %s, 15" % v))
        L.append("call void @llvm.assume(i1 %s)" % z)
    L.append("call void @llvm.memcpy.p0.p0.i64(%s, %s, i64 %s, i1 false)" % (d, s, n64))
    if mbar:
        decls["ptx.cp.async.bulk.complete_tx"] = \
            "declare void @ptx.cp.async.bulk.complete_tx(ptr, i32) nounwind"
        L.append("call void @ptx.cp.async.bulk.complete_tx(%s, i32 %s)" % (mbar, n))
    return L, decls


def discard_ir(p):
    """discard.L2: 128 bytes at p (128-byte aligned) become indeterminate."""
    return "store <128 x i8> poison, %s, align 128" % p


def st_bulk_ir(p, n, fresh):
    """IR for st.bulk: zero n (an i64) bytes at p (a typed pointer operand);
    n must be a multiple of 8 and p 8-byte aligned."""
    L = []
    decls = {"llvm.assume": "declare void @llvm.assume(i1)",
             "llvm.memset.p0.i64": "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)"}

    def op(text):
        t = fresh()
        L.append("%s = %s" % (t, text))
        return t
    for v in (n, op("ptrtoint %s to i64" % p)):
        z = op("icmp eq i64 %s, 0" % op("and i64 %s, 7" % v))
        L.append("call void @llvm.assume(i1 %s)" % z)
    L.append("call void @llvm.memset.p0.i64(%s, i8 0, i64 %s, i1 false)" % (p, n))
    return L, decls


def lift_function(ptx_text, sig, name_map, decls, globals_, ptx_name,
                  typedefs=None, indirect_sigs=None):
    funcs = parse_module(ptx_text)
    if ptx_name not in funcs:
        raise Unsupported("function not found in PTX: " + ptx_name)
    L = Lifter(funcs[ptx_name], sig, name_map, decls, globals_, typedefs, indirect_sigs)
    try:
        L.lift()
    except ValueError as e:  # layout of an unsupported LLVM type
        raise Unsupported(str(e))
    return L


def render(L, define_line):
    # The lifted code computes integer addresses from pointer arguments
    # (ptrtoint), which Alive2 counts as a capture; capture attributes are a
    # source-level promise that does not constrain the machine code.
    define_line = re.sub(r"\s*captures\([^)]*\)|\s*\bnocapture\b", "", define_line)
    out = [define_line.rstrip().rstrip("{").rstrip() + " {"]
    for label, lines in L.blocks:
        out.append(label + ":")
        out.extend(lines)
    out.append("}")
    return "\n".join(out)
