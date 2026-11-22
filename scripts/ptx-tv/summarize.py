#!/usr/bin/env python3
"""Summarize ptx-tv JSONL results: status counts and top unsupported reasons."""
import collections
import json
import re
import sys


def norm(detail):
    d = detail.split("\n")[0]
    d = re.sub(r"%[a-zA-Z_]+\d+", "%R", d)
    d = re.sub(r"\b\d+\b", "N", d)
    return d[:90]


def main():
    rows = []
    for path in sys.argv[1:]:
        for l in open(path):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    fn_rows = [r for r in rows if "function" in r]
    file_rows = [r for r in rows if "function" not in r]
    st = collections.Counter(r["status"] for r in fn_rows)
    print("files: %d (file-level failures: %d)" % (
        len({r["file"] for r in rows}), len(file_rows)))
    for k, v in collections.Counter(r["status"] for r in file_rows).most_common():
        print("  file %-22s %d" % (k, v))
    print("functions: %d" % len(fn_rows))
    for k, v in st.most_common():
        print("  %-24s %6d  %5.1f%%" % (k, v, 100.0 * v / max(len(fn_rows), 1)))
    for status in ("lift-unsupported", "lift-crash", "lift-invalid-ir",
                   "alive-unsupported", "alive-error"):
        c = collections.Counter(norm(r.get("detail", "")) if status != "alive-unsupported" and status != "alive-error"
                                else norm(re.sub(r"(?s).*?(ERROR[^\n]*).*", r"\1", r.get("detail", "")))
                                for r in fn_rows if r["status"] == status)
        if not c:
            continue
        print("\ntop %s:" % status)
        for k, v in c.most_common(30):
            print("  %5d  %s" % (v, k))
    mism = [r for r in fn_rows if r["status"] == "mismatch"]
    if mism:
        print("\nmismatches:")
        for r in mism:
            print("  %s :: %s  (%s)" % (r["file"], r["function"], r.get("dir", "")))


if __name__ == "__main__":
    main()


def known_summary(paths):
    c = collections.Counter()
    for path in paths:
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "mismatch-known":
                    c[r.get("known")] += 1
    if c:
        print("\nknown false-alarm classes:")
        for k, v in c.most_common():
            print("  %5d  %s" % (v, k))


if __name__ == "__main__":
    known_summary(sys.argv[1:])
