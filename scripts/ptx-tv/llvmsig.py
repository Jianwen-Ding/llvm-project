"""Minimal textual parser for LLVM function signatures (types, args, attrs).

Only what ptx-tv needs: split an LLVM `define`/`declare` line into its
return type, argument types, argument attributes and argument names.
"""
import re

SIMPLE_TYPES = {"void", "half", "bfloat", "float", "double", "fp128",
                "x86_fp80", "ppc_fp128", "label", "metadata", "token",
                "x86_amx"}


class TypeParseError(Exception):
    pass


def _skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def parse_type(s, i=0):
    """Parse an LLVM type starting at s[i]; return (type_text, end_index)."""
    i = _skip_ws(s, i)
    start = i
    if i >= len(s):
        raise TypeParseError("eof")
    c = s[i]
    if c == "<" and s.startswith("<{", i):
        i = _match(s, i + 1, "{", "}") + 1
        if i >= len(s) or s[i] != ">":
            raise TypeParseError("packed struct")
        i += 1
    elif c == "<":
        i = _match(s, i, "<", ">")
        i += 1
    elif c == "[":
        i = _match(s, i, "[", "]") + 1
    elif c == "{":
        i = _match(s, i, "{", "}") + 1
    elif c == "%":
        m = re.compile(r'%("[^"]*"|[-a-zA-Z$._0-9]+)').match(s, i)
        if not m:
            raise TypeParseError("named type")
        i = m.end()
    else:
        m = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*").match(s, i)
        if not m:
            raise TypeParseError("type at %r" % s[i:i + 20])
        w = m.group(0)
        i = m.end()
        if re.fullmatch(r"i\d+", w) or w in SIMPLE_TYPES:
            pass
        elif w == "ptr":
            m2 = re.compile(r"\s*addrspace\(\d+\)").match(s, i)
            if m2:
                i = m2.end()
        elif w == "target":
            m2 = re.compile(r'\("[^"]*"[^)]*\)').match(s, i)
            if not m2:
                raise TypeParseError("target type")
            i = m2.end()
        else:
            raise TypeParseError("not a type: " + w)
    t = s[start:i].strip()
    # function / old-style pointer suffixes are not supported
    return t, i


def _match(s, i, open_c, close_c):
    depth = 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            i = s.index('"', i + 1)
        elif ch in "<[{(":
            depth += 1
        elif ch in ">]})":
            depth -= 1
            if depth == 0:
                if ch != close_c:
                    raise TypeParseError("mismatched brackets")
                return i
        i += 1
    raise TypeParseError("unbalanced")


def split_top(s, sep=","):
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            j = s.index('"', i + 1)
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if ch in "<[{(":
            depth += 1
        elif ch in ">]})":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    if "".join(cur).strip():
        out.append("".join(cur))
    return [x.strip() for x in out]


class Arg:
    def __init__(self, ty, attrs, name):
        self.ty, self.attrs, self.name = ty, attrs, name

    def has(self, a):
        return re.search(r"(^|\s)" + re.escape(a) + r"\b", self.attrs) is not None

    def byval_type(self):
        m = re.search(r"\bbyval\(", self.attrs)
        if not m:
            return None
        t, _ = parse_type(self.attrs, m.end())
        return t

    def __repr__(self):
        return "Arg(%s|%s|%s)" % (self.ty, self.attrs, self.name)


class Sig:
    pass


def parse_sig(line):
    """Parse a `define ...` or `declare ...` line."""
    m = re.search(r'@("[^"]*"|[-a-zA-Z$._0-9]+)\(', line)
    if not m:
        raise TypeParseError("no function name")
    sig = Sig()
    sig.name = m.group(1)
    head = line[:m.start()].split(None, 1)[1] if " " in line[:m.start()] else ""
    toks = head.split()
    sig.ret = None
    for k in range(len(toks)):
        cand = " ".join(toks[k:])
        try:
            t, e = parse_type(cand)
            if e == len(cand):
                sig.ret = t
                sig.ret_attrs = " ".join(toks[:k])
                break
        except TypeParseError:
            continue
    if sig.ret is None:
        raise TypeParseError("ret type")
    # args
    i = m.end() - 1
    j = _match(line, i, "(", ")")
    argtext = line[i + 1:j]
    sig.rest = line[j + 1:]
    sig.args = []
    sig.vararg = False
    for a in split_top(argtext):
        if a == "...":
            sig.vararg = True
            continue
        t, e = parse_type(a)
        rest = a[e:].strip()
        name = None
        mm = re.search(r'(%("[^"]*"|[-a-zA-Z$._0-9]+))$', rest)
        if mm:
            name = mm.group(1)
            rest = rest[:mm.start()].strip()
        sig.args.append(Arg(t, rest, name))
    return sig


def int_bits(t):
    m = re.fullmatch(r"i(\d+)", t)
    return int(m.group(1)) if m else None
