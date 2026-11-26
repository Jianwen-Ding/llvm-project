; A .metallib as the AIR target writes it for Metal's driver (typed-pointer
; bitcode behind the wrapper, in the container) compiles to AGX in one
; llvm-air2agx run, as the module's portable AIR does through opt -O3 and llc.
; RUN: llc -mtriple=air -filetype=obj %S/Inputs/saxpy.ll -o %t.metallib
; RUN: llvm-air2agx --list %t.metallib | FileCheck %s --check-prefix=LIST
; RUN: llvm-air2agx -mcpu=agx-g14g --emit=asm %t.metallib -o - | FileCheck %s --check-prefix=ASM
; RUN: llvm-air2agx -mcpu=agx-g14g --emit=llvm %t.metallib -o - | FileCheck %s --check-prefix=IR
; RUN: llvm-air2agx -mcpu=agx-g14g %t.metallib -o %t.o
; RUN: llvm-readobj --sections %t.o | FileCheck %s --check-prefix=OBJ

; The same kernel through the .air file Apple's metallib tool would link.
; RUN: llc -mtriple=air64-apple-macosx -filetype=obj -air-output=air %S/Inputs/saxpy.ll -o %t.air
; RUN: llvm-air2agx -mcpu=agx-g14g %t.air -o %t.air.o
; RUN: cmp %t.o %t.air.o

; Each kernel's instructions, for agx_cost.
; RUN: llvm-air2agx -mcpu=agx-g14g --emit=code %t.metallib -o %t.code
; RUN: llvm-objcopy -O binary --only-section=.text %t.o %t.text
; RUN: cmp %t.code.saxpy.bin %t.text

; Several processors are one job each, in parallel, named after the input.
; RUN: rm -rf %t.dir && mkdir %t.dir && cp %t.air %t.dir/k.air && cp %t.metallib %t.dir/k.metallib
; RUN: llvm-air2agx -mcpu=agx-g13g,agx-g14g -j 2 --output-dir %t.dir/out %t.dir/k.metallib %t.dir/k.air --time 2>&1 | FileCheck %s --check-prefix=TIME
; RUN: ls %t.dir/out | FileCheck %s --check-prefix=NAMES
; RUN: cmp %t.dir/out/k.agx-g14g.o %t.o

; LIST: entry saxpy, type 2, AIR 2.8, {{.*}} bytes of bitcode, hash ok

; ASM-LABEL: saxpy:
; ASM: device_load {{.*}}, i32,
; ASM: device_store

; IR: target triple = "agx"
; The Metal pipeline packs the scalars into one device buffer, after x and y.
; IR: define agx_kernel void @saxpy(ptr addrspace(1) {{.*}}, ptr addrspace(1) {{.*}}, ptr addrspace(1) {{[^,]*}})
; IR: call i32 @llvm.agx.thread.position.in.grid.x()
; IR-NOT: !air.

; OBJ: Name: .text
; OBJ: Name: .agx.kernels

; TIME-COUNT-4: kernel(s): read {{.*}} ms, opt {{.*}} ms, codegen {{.*}} ms
; TIME: 4 job(s) in

; Two inputs of one name: the second's outputs take a number.
; NAMES: k.1.agx-g13g.o
; NAMES-NEXT: k.1.agx-g14g.o
; NAMES-NEXT: k.agx-g13g.o
; NAMES-NEXT: k.agx-g14g.o
