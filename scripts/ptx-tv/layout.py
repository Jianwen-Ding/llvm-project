"""NVPTX64 data layout for LLVM types given as text.

datalayout: e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64
"""
import re

from llvmsig import parse_type, split_top

FP_BITS = {"half": 16, "bfloat": 16, "float": 32, "double": 64, "fp128": 128}


class Ty:
    def __init__(self, kind, text, **kw):
        self.kind, self.text = kind, text
        self.__dict__.update(kw)

    def __repr__(self):
        return self.text


def parse(text, typedefs, _depth=0):
    text = text.strip()
    if _depth > 50:
        raise ValueError("recursive type")
    m = re.fullmatch(r"i(\d+)", text)
    if m:
        return Ty("int", text, bits=int(m.group(1)))
    if text in FP_BITS:
        return Ty("fp", text, bits=FP_BITS[text])
    if text.startswith("ptr"):
        return Ty("ptr", text, bits=64)
    if text.startswith("<{") or text.startswith("{"):
        packed = text.startswith("<{")
        inner = text[2:-2] if packed else text[1:-1]
        elems = [parse(e, typedefs, _depth + 1) for e in split_top(inner)] if inner.strip() else []
        return Ty("struct", text, elems=elems, packed=packed)
    m = re.fullmatch(r"<\s*(\d+)\s+x\s+(.+)>", text)
    if m:
        return Ty("vec", text, n=int(m.group(1)), elem=parse(m.group(2), typedefs, _depth + 1))
    m = re.fullmatch(r"\[\s*(\d+)\s+x\s+(.+)\]", text)
    if m:
        return Ty("arr", text, n=int(m.group(1)), elem=parse(m.group(2), typedefs, _depth + 1))
    if text.startswith("%"):
        if text not in typedefs:
            raise ValueError("unknown named type " + text)
        t = parse(typedefs[text], typedefs, _depth + 1)
        t.text = text
        return t
    raise ValueError("unsupported type " + text)


def pow2ceil(x):
    p = 1
    while p < x:
        p *= 2
    return p


def store_size(t):
    if t.kind in ("int", "fp", "ptr"):
        return (t.bits + 7) // 8
    if t.kind == "vec":
        return (t.n * bits(t.elem) + 7) // 8
    return alloc_size(t)


def bits(t):
    if t.kind in ("int", "fp", "ptr"):
        return t.bits
    if t.kind == "vec":
        return t.n * bits(t.elem)
    raise ValueError("bits of aggregate")


def align(t):
    if t.kind == "int":
        b = t.bits
        for w, a in ((8, 1), (16, 2), (32, 4), (64, 8), (128, 16), (256, 32)):
            if b <= w:
                return a
        return 32
    if t.kind == "fp":
        return t.bits // 8
    if t.kind == "ptr":
        return 8
    if t.kind == "vec":
        return pow2ceil(max(store_size(t), 1))
    if t.kind == "arr":
        return align(t.elem)
    if t.kind == "struct":
        if t.packed or not t.elems:
            return 1
        return max(align(e) for e in t.elems)
    raise ValueError(t.kind)


def alloc_size(t):
    if t.kind == "arr":
        return t.n * alloc_size(t.elem)
    if t.kind == "struct":
        off = 0
        for e in t.elems:
            if not t.packed:
                off = (off + align(e) - 1) // align(e) * align(e)
            off += alloc_size(e)
        a = align(t)
        return (off + a - 1) // a * a
    s = store_size(t)
    a = align(t)
    return (s + a - 1) // a * a


def offsets(t):
    off, out = 0, []
    for e in t.elems:
        if not t.packed:
            off = (off + align(e) - 1) // align(e) * align(e)
        out.append(off)
        off += alloc_size(e)
    return out


def is_sub_byte(t):
    if t.kind == "int":
        return t.bits % 8 != 0
    if t.kind == "vec":
        return bits(t.elem) % 8 != 0
    return False


def has_sub_byte(t):
    if t.kind == "struct":
        return any(has_sub_byte(e) for e in t.elems)
    if t.kind == "arr":
        return has_sub_byte(t.elem)
    return is_sub_byte(t)
