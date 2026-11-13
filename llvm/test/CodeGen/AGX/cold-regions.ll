; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-cold-regions=false < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 < %s | FileCheck %s --check-prefix=OFF

; An if-body entered with an unlikely edge (here double precision's general
; code, behind its short path) moves to the end of the kernel: skipping it is
; a jmp_exec_any not taken instead of a jmp_exec_none taken (a taken jump
; costs about 34 cycles on G14C, one not taken 1), and the body comes back
; with an unconditional jump to the join. Not with ieee-fp32, whose exact
; paths are taken for zero operands.

; CHECK-LABEL: dmul:
; CHECK: jmp_exec_any [[COLD:\.LBB[0-9_]+]]
; CHECK-NOT: jmp_exec_none
; CHECK: [[JOIN:\.LBB[0-9_]+]]:
; CHECK: pop_exec
; CHECK: device_store
; CHECK: stop
; CHECK: [[COLD]]:
; CHECK: jmp_exec_any [[JOIN]]
; CHECK-NEXT: jmp_exec_none [[JOIN]]
; OFF-LABEL: dmul:
; OFF: jmp_exec_none
; OFF-NOT: jmp_exec_any
; OFF: stop
define agx_kernel void @dmul(ptr addrspace(1) %p, double %a, double %b) {
  %r = fmul double %a, %b
  store double %r, ptr addrspace(1) %p
  ret void
}

; An if-body of no known probability stays in line.
; CHECK-LABEL: maybe:
; CHECK: jmp_exec_none
; CHECK: stop
define agx_kernel void @maybe(ptr addrspace(1) %p, i32 %x) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %t, %x
  br i1 %c, label %then, label %end
then:
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %v = load i32, ptr addrspace(1) %a
  %w = mul i32 %v, %v
  %w2 = mul i32 %w, %v
  %w3 = add i32 %w2, %x
  store i32 %w3, ptr addrspace(1) %a
  br label %end
end:
  ret void
}
