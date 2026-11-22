#!/bin/bash
# show.sh FILE FUNC : run ptx-tv on one function and print source, PTX, Alive2 verdict
f=$1; fn=$2
W=${PTXTV_SHOW_DIR:-/tmp/ptxtv-show}; mkdir -p $W
D=$(dirname "$0")
r=$(PTXTV_WALL_TIMEOUT=${PTXTV_WALL_TIMEOUT:-60} python3 $D/tv.py --workdir $W -j1 --smt-to ${SMT:-10000} --keep --func "$fn" /home/user/llvm-project/llvm/test/CodeGen/NVPTX/$f)
dir=$(echo "$r" | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(r.get('dir','')); print(r['status'], r.get('known',''), r.get('param_oob',''), file=sys.stderr)")
[ -z "$dir" ] && exit 0
echo "--- src"; grep -v '^;\|^$\|^target\|^source_filename\|^attributes' $dir/src.ll | head -${SRCN:-25}
echo "--- ptx"; awk -v n="$fn(" 'index($0,n){f=1} f{print} /End function/{if(f)exit}' $dir/../out.ptx | grep -v '^\s*//\|^\s*$' | head -${PTXN:-40}
echo "--- alive"; /home/user/alive2/build/alive-tv $dir/src.ll $dir/tgt.ll --tgt-is-asm --disable-undef-input --smt-to=${SMT:-10000} 2>&1 | grep -A${AN:-14} "^ERROR\|^Example" | head -${AN2:-30}
