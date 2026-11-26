; Bare AIR bitcode (no wrapper, no container) to pure AGX machine code (no
; container): -o *.agx writes the instructions alone, the object's .text.
; RUN: llc -mtriple=air64-apple-macosx -filetype=obj -air-output=air %S/Inputs/saxpy.ll -o %t.air
; RUN: %python -c "import struct,sys; d=open(sys.argv[1],'rb').read(); o,n=struct.unpack_from('<II',d,8); open(sys.argv[2],'wb').write(d[o:o+n])" %t.air %t.raw.air
; RUN: %python -c "import sys; assert open(sys.argv[1],'rb').read(4) == b'BC\xc0\xde'" %t.raw.air
; RUN: llvm-air2agx -mcpu=agx-g14g %t.raw.air -o %t.agx
; RUN: llvm-air2agx -mcpu=agx-g14g %t.air -o %t.o
; RUN: llvm-objcopy -O binary --only-section=.text %t.o %t.text
; RUN: cmp %t.agx %t.text
; RUN: llvm-air2agx -mcpu=agx-g14g --emit=agx --kernel saxpy %t.raw.air -o %t.k.agx
; RUN: cmp %t.k.agx %t.agx
; RUN: not llvm-air2agx -mcpu=agx-g14g --emit=agx --kernel nosuch %t.raw.air -o %t.n.agx 2>&1 | FileCheck %s --check-prefix=NOKERNEL
; RUN: not llvm-air2agx -mcpu=agx-g14g --kernel saxpy %t.raw.air -o %t.n.o 2>&1 | FileCheck %s --check-prefix=NEEDSAGX

; NOKERNEL: error: no kernel nosuch
; NEEDSAGX: error: --kernel needs --emit=agx
