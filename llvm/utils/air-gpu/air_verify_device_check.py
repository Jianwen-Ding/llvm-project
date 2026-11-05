#!/usr/bin/env python3
"""Does llc's check of AIR agree with the device, on modules made to be odd?

Each module in verify-cases/ is written with -air-skip-device-checks and run on the
device behind the AIR service (one request at a time, under the device lock, with
the GPU timeout), and written normally, when llc's checks may report it. A rule of
AIRVerify.cpp is right only if the device rejects what it reports, and the other
way round, but for cases marked "stricter": invalid AIR the device takes anyway. The output is one line per module, and the exit status is 1 on a
disagreement.

    python3 air_verify_device_check.py --llc build/bin/llc
"""
import argparse, base64, fcntl, glob, os, subprocess, sys, tempfile

ap = argparse.ArgumentParser()
ap.add_argument('--llc', default='llc')
ap.add_argument('--queue', default=os.environ.get('AIR_QUEUE', '/air'))
ap.add_argument('--lock', default=os.environ.get('AIR_DEVICE_LOCK', '/scratch/air-device.lock'))
args = ap.parse_args()
sys.path.insert(0, os.environ.get('AIR_CLIENT', '/air/client'))
from client import submit  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))


def request(job):
    with open(args.lock, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return submit(job, args.queue, timeout=180)


bad = 0
for path in sorted(glob.glob(os.path.join(here, 'verify-cases', '*.ll'))):
    name = os.path.basename(path)[:-3]
    with tempfile.TemporaryDirectory() as tmp:
        air = os.path.join(tmp, 'x.air')
        common = [args.llc, '-mtriple=air', '-filetype=obj', '-air-output=air', path, '-o', air]
        checked = subprocess.run(common, capture_output=True, text=True)
        backend = checked.returncode == 0
        unchecked = subprocess.run(common + ['-air-skip-device-checks'], capture_output=True, text=True)
        if unchecked.returncode != 0:
            print(f'{name:14} llc-fails-before-writing  backend={"ok" if backend else "error"}  '
                  f'{unchecked.stderr.strip().splitlines()[-1][:100] if unchecked.stderr.strip() else ""}')
            bad += backend
            continue
        data = base64.b64encode(open(air, 'rb').read()).decode()
        reply = request({'op': 'execute', 'air': data, 'functionality': 'AIR check: ' + name,
                         'gpu_timeout_ms': 2000, 'compile_timeout_ms': 20000,
                         'buffers': [{'size': 256}, {'size': 256}],
                         'commands': [{'kernel': 'k', 'bindings': [{'index': 0, 'buffer': 0}, {'index': 1, 'buffer': 1}],
                                       'grid': [1, 1, 1], 'group': [1, 1, 1]}], 'readback': [0]})
        device = bool(reply.get('ok'))
        # (a "stricter" case: not valid AIR although the device takes it, without an error)
        stricter = open(path).readline().startswith('; stricter:')
        agree = device == backend or (stricter and device and not backend)
        bad += not agree
        print(f'{name:14} device={"ok" if device else "REJECT"} backend={"ok" if backend else "error"} '
              f'{"" if agree else "DISAGREE "}'
              f'{str(reply.get("error", ""))[:90].strip()!r} | {checked.stderr.strip()[-110:]!r}', flush=True)
sys.exit(1 if bad else 0)
