; A module that asks for IEEE binary32 arithmetic (the module flag
; air.ieee-fp32, as ZLUDA's strict floating point mode writes it) gets AGX's
; ieee-fp32 feature in its functions whatever the target machine's features:
; the bfloat addition, whose AIR emulation agx-lower-air restores as one
; operation, keeps denormals (it is computed exactly, not on the flushing f32
; ALU), so 0x0001 + 0x0001 is 0x0002.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -S %t.air.ll | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %t.air.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: sed -e '/air.ieee-fp32/d' -e 's/!llvm.module.flags = !{!0}//' %s > %t.noflag.ll
; RUN: llc -mtriple=air -air-portable %t.noflag.ll -o %t.noflag.air.ll
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -S %t.noflag.air.ll | FileCheck %s --check-prefix=NOFLAG

; CHECK: define agx_kernel void @k({{.*}} #[[A:[0-9]+]]
; CHECK: attributes #[[A]] = {{.*}}"target-features"="+ieee-fp32"
; NOFLAG-NOT: +ieee-fp32

; The IEEE lowering: the hardware's add of the extended operands, then a
; branch to the exact computation where an operand or the result is denormal.
; ASM-LABEL: k:
; ASM: fadd32
; ASM: jmp_exec_none
; ASM: stop
define void @k(ptr addrspace(1) %out, bfloat %a, bfloat %b) "air-kernel" {
  %s = fadd bfloat %a, %b
  store bfloat %s, ptr addrspace(1) %out
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"air.ieee-fp32", i32 1}
