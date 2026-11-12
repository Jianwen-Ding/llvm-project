; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-generic-group-max=0 < %s | FileCheck %s --check-prefix=OFF

; Generic accesses whose pointers have one tag at run time (selects on one
; condition of a private and a device address, as InstCombine leaves GEPs of a
; select) share one tag test: the run is copied for each memory.

; CHECK-LABEL: run:
; CHECK:         jmp_exec_none
; CHECK-NOT:     jmp_exec_none
; CHECK:         device_load
; CHECK:         device_load
; CHECK:         device_store
; CHECK:         jmp_exec_none
; CHECK-NOT:     jmp_exec_none
; CHECK:         stack_load
; CHECK:         stack_load
; CHECK:         stack_store
; CHECK-NOT:     jmp_exec_none
; CHECK:         stop
; OFF-LABEL: run:
; OFF-COUNT-6:   jmp_exec_none
define agx_kernel void @run(ptr addrspace(1) %g, ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %loc = alloca [4 x float], align 4, addrspace(5)
  %k = and i32 %i, 3
  %e = getelementptr float, ptr addrspace(5) %loc, i32 %k
  store float 1.0, ptr addrspace(5) %e
  %lg = addrspacecast ptr addrspace(5) %loc to ptr
  %gg = addrspacecast ptr addrspace(1) %g to ptr
  %c = icmp ult i32 %i, %n
  %l1 = getelementptr i8, ptr %lg, i64 4
  %g1 = getelementptr i8, ptr %gg, i64 4
  %p0 = select i1 %c, ptr %lg, ptr %gg
  %p1 = select i1 %c, ptr %l1, ptr %g1
  %a = load float, ptr %p0
  %b = load float, ptr %p1
  %s = fadd float %a, %b
  store float %s, ptr %p1
  %o = getelementptr float, ptr addrspace(1) %out, i32 %i
  %m = getelementptr float, ptr addrspace(5) %loc, i32 %i
  %x = load float, ptr addrspace(5) %m
  %r = fadd float %s, %x
  store float %r, ptr addrspace(1) %o
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
