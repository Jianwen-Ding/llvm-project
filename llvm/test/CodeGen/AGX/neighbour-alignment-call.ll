; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-vector-accesses < %s | FileCheck %s

; A later access proves nothing where an instruction between may not return.
; CHECK-LABEL: @call(
; CHECK: load i8, ptr addrspace(1) %q2, align 1
define void @call(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %q2 = getelementptr i8, ptr addrspace(1) %p, i64 2
  %b2 = load i8, ptr addrspace(1) %q2, align 1
  call void @may_not_return()
  %d = load half, ptr addrspace(1) %p, align 2
  %x2 = uitofp i8 %b2 to half
  %r = fadd half %x2, %d
  store half %r, ptr addrspace(1) %o
  ret void
}

declare void @may_not_return()
