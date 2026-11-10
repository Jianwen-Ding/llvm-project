#!/usr/bin/env python3
"""Cost MLIR kernels on AGX with the AGX Performance Simulator.

Each kernel here goes through convert-gpu-to-air (with the given mlir-opt),
mlir-translate, the AIR target's portable passes, opt -O3 for AGX (which
runs agx-lower-air first) and llc -O3, as agx-build-fatbin.py builds them;
agx_cost gives one SIMD-group's cycles. With two mlir-opts, the second
column is the second's and the change is printed.

  bench.py --bin BUILD/bin --cost agx_cost [--base OLD_MLIR_OPT] [KERNEL.mlir ...]
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = ('builtin.module(gpu.module(canonicalize,'
            'convert-gpu-to-air{use-bare-ptr-memref-call-conv=1},canonicalize))')


def run(cmd):
  return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def lower(mlir_opt, bindir, src, tmp):
  """The kernel as LLVM IR (upstream AIR)."""
  out = run([mlir_opt, src, '--pass-pipeline=' + PIPELINE])
  # The gpu.module's body as a builtin module: the lines after its header,
  # up to its closing brace (the module's own is the last).
  lines = out.rstrip().splitlines()
  start = next(i for i, l in enumerate(lines) if l.strip().startswith('gpu.module'))
  body = '\n'.join(lines[start + 1:-2])
  m = os.path.join(tmp, 'k.mlir')
  open(m, 'w').write('module {' + body + '}\n')
  ll = os.path.join(tmp, 'k.ll')
  open(ll, 'w').write(run([os.path.join(bindir, 'mlir-translate'),
                           '--mlir-to-llvmir', m]))
  return ll


def cost(bindir, costbin, ll, tmp, cpu, args, optimize=True):
  llc, opt = os.path.join(bindir, 'llc'), os.path.join(bindir, 'opt')
  air, agx = os.path.join(tmp, 'air.ll'), os.path.join(tmp, 'agx.ll')
  run([llc, '-mtriple=air', '-air-portable', ll, '-o', air])
  passes = ['-O3'] if optimize else ['-passes=agx-lower-air']
  run([opt, '-mtriple=agx', '-mcpu=' + cpu] + passes + ['-S', air, '-o', agx])
  level = ['-O3'] if optimize else []
  asm = run([llc, '-mtriple=agx', '-mcpu=' + cpu] + level + [agx, '-o', '-'])
  obj, binf = os.path.join(tmp, 'k.o'), os.path.join(tmp, 'k.bin')
  run([llc, '-mtriple=agx', '-mcpu=' + cpu] + level +
      ['-filetype=obj', agx, '-o', obj])
  run([os.path.join(bindir, 'llvm-objcopy'), '-O', 'binary',
       '--only-section=.text', obj, binf])
  j = json.loads(run([costbin, binf, '--allow-unknown'] + args)
                 .strip().splitlines()[-1])
  insts = [l for l in asm.splitlines()
           if re.match(r'^\t[a-z]', l) and not l.strip().startswith('trap')]
  gprs = re.search(r'general purpose registers: (\d+)', asm)
  return {'cycles': j['cycles'], 'instructions': len(insts),
          'gprs': int(gprs.group(1)) if gprs else 0, 'asm': asm}


def launch_args(src):
  for l in open(src):
    if l.startswith('// agx-cost:'):
      return l.split(':', 1)[1].split()
  return []


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('--bin', required=True, help='LLVM build bin directory')
  ap.add_argument('--cost', required=True, help="the simulator's agx_cost")
  ap.add_argument('--base', help='an mlir-opt to compare with')
  ap.add_argument('-mcpu', default='agx-g14c')
  ap.add_argument('--asm', help='write each assembly to this directory')
  ap.add_argument('--no-O3', dest='optimize', action='store_false',
                  help='only agx-lower-air between the portable passes and llc')
  ap.add_argument('kernels', nargs='*')
  a = ap.parse_args()
  kernels = a.kernels or sorted(glob.glob(os.path.join(HERE, '*.mlir')))
  new_opt = os.path.join(a.bin, 'mlir-opt')
  opts = ([('base', a.base)] if a.base else []) + [('new', new_opt)]
  print('%-16s' % 'kernel' + ''.join('%28s' % n for n, _ in opts) +
        ('%10s' % 'change' if a.base else ''))
  for k in kernels:
    row = []
    for name, mo in opts:
      with tempfile.TemporaryDirectory() as tmp:
        r = cost(a.bin, a.cost, lower(mo, a.bin, k, tmp), tmp, a.mcpu,
                 launch_args(k), a.optimize)
      if a.asm:
        os.makedirs(a.asm, exist_ok=True)
        open(os.path.join(a.asm, '%s.%s.s' % (os.path.basename(k)[:-5], name)),
             'w').write(r['asm'])
      row.append(r)
    line = '%-16s' % os.path.basename(k)[:-5]
    for r in row:
      line += '%28s' % ('%.1f cyc %3d inst %2d gpr' % (
          r['cycles'], r['instructions'], r['gprs']))
    if a.base:
      line += '%+9.1f%%' % (100.0 * (row[1]['cycles'] / row[0]['cycles'] - 1))
    print(line)


if __name__ == '__main__':
  sys.exit(main())
