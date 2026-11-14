#!/usr/bin/env python3
"""Write a Markdown report (RESULTS.md) from a ptx-tv JSONL results file."""
import collections
import json
import re
import sys


def unsupported_key(d):
    m = re.match(r"modifier \.(\S+) in", d)
    if m:
        return "modifier ." + m.group(1)
    m = re.match(r"(?:opcode )?([a-z0-9_]+(?:\.[a-z0-9_]+)?)", d)
    return m.group(1) if m else d[:40]


def main(path, out, title, config):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    fn = [r for r in rows if "function" in r]
    files = {r["file"] for r in rows}
    file_fail = collections.Counter(r["status"] for r in rows if "function" not in r)
    st = collections.Counter(r["status"] for r in fn)
    n = len(fn)
    L = ["# %s" % title, "", config, ""]
    L.append("Test files: %d; files with no usable nvptx64 llc RUN line or other "
             "file-level failure: %d (%s)." % (
                 len(files), sum(file_fail.values()),
                 ", ".join("%s %d" % kv for kv in file_fail.most_common())))
    L.append("Functions checked: %d." % n)
    L.append("")
    L.append("| status | functions | share |")
    L.append("|---|---:|---:|")
    for k, v in st.most_common():
        L.append("| `%s` | %d | %.1f%% |" % (k, v, 100.0 * v / n))
    concl = st["correct"] + st["mismatch"] + st["mismatch-known"] + st["mismatch-param-oob"]
    nprobe = sum(1 for r in fn if r["status"] == "correct" and r.get("probe"))
    nweak = sum(1 for r in fn if r["status"] == "correct" and r.get("alive_extra"))
    L.append("")
    L.append("%d of the correct results were obtained only with a weaker check "
             "(`--disable-poison-input`: inputs assumed not to be poison) after the "
             "full check timed out." % nweak)
    L.append("")
    L.append("Of the correct results, %d rely on probe-derived opaque instructions "
             "(probe.py): they check the surrounding code and operand plumbing, not "
             "the instruction selection of the intrinsic itself." % nprobe)
    L.append("")
    nop = sum(1 for r in fn if r["status"] == "correct" and r.get("copy_model") == "opaque")
    if nop:
        L.append("%d of these used the opaque copy model (`PTXTV_OPAQUE_COPIES=1`) "
                 "after the check with cp.async/st.bulk copy semantics timed out." % nop)
        L.append("")
    L.append("Conclusive (proved correct or counterexample): %d (%.1f%%)." % (concl, 100.0 * concl / n))
    L.append("")
    oob = [r for r in fn if r["status"] == "mismatch-param-oob"]
    if oob:
        L.append("## Counterexamples with out-of-bounds `.param` accesses")
        L.append("")
        L.append("All are finding 1 (sub-byte vectors) except `param-vectorize-device.ll` "
                 "`callee_St4x*`, the lower-severity over-read into alignment padding.")
        L.append("")
        for r in sorted(oob, key=lambda r: (r["file"], r["function"])):
            L.append("* `%s` `%s`: %s" % (r["file"], r["function"], "; ".join(r.get("param_oob", [])[:2])))
        L.append("")
    mm = [r for r in fn if r["status"] == "mismatch"]
    L.append("## Unclassified counterexamples")
    L.append("")
    if not mm:
        L.append("None.")
    for r in sorted(mm, key=lambda r: (r["file"], r["function"])):
        L.append("* `%s` `%s`" % (r["file"], r["function"]))
    L.append("")
    kn = collections.Counter(r.get("known") for r in fn if r["status"] == "mismatch-known")
    if kn:
        L.append("## Counterexamples of known modelling classes (not defects)")
        L.append("")
        L.append("| class | functions |")
        L.append("|---|---:|")
        for k, v in kn.most_common():
            L.append("| `%s` | %d |" % (k, v))
        L.append("")
    un = collections.Counter(unsupported_key(r.get("detail", "")) for r in fn
                             if r["status"] == "lift-unsupported")
    if un:
        L.append("## Not lifted (top reasons)")
        L.append("")
        L.append("| reason | functions |")
        L.append("|---|---:|")
        for k, v in un.most_common(25):
            L.append("| `%s` | %d |" % (k, v))
        L.append("")
    al = collections.Counter(
        re.sub(r"'[^']*'", "'...'", re.search(r"ERROR: [^\n]*", r.get("detail", "")).group(0)
               if re.search(r"ERROR: [^\n]*", r.get("detail", "")) else r.get("detail", "")[:60])
        for r in fn if r["status"] in ("alive-unsupported", "alive-error"))
    if al:
        L.append("## Alive2 could not handle the source")
        L.append("")
        for k, v in al.most_common(10):
            L.append("* %d: %s" % (v, k.strip()[:100]))
        L.append("")
    open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ptx-tv results",
         sys.argv[4] if len(sys.argv) > 4 else "")
