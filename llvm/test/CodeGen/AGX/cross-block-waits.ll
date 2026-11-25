; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-max-unskipped-instrs=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-cross-block-waits=0 < %s | FileCheck %s --check-prefix=BLOCK

; Guarded loads whose results are first used after the joins stay in flight
; across the ifs (L15-E428): one wait, before the sum, not one at the end of
; each guarded side. @exit_unused's load is unused on the path that ends the
; kernel: its block waits before the branch, not the exit before its stop
; (the side is long enough to be jumped over at a limit of 4 instructions,
; L15-E433, E434).

; CHECK-LABEL: guarded:
; CHECK:       device_load
; CHECK-NOT:   wait
; CHECK:       pop_exec
; CHECK-NOT:   wait
; CHECK:       device_load
; CHECK-NOT:   wait
; CHECK:       pop_exec
; CHECK:       wait 0
; CHECK-NEXT:  fadd32
; CHECK-NOT:   wait
; CHECK:       stop
; BLOCK-LABEL: guarded:
; BLOCK:       device_load
; BLOCK-NEXT:  wait 0
; BLOCK:       device_load
; BLOCK-NEXT:  wait 0
define agx_kernel void @guarded(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %a, i32 %b) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %ca = icmp ult i32 %x, %a
  br i1 %ca, label %la, label %ja
la:
  %pa = getelementptr inbounds float, ptr addrspace(1) %p, i32 %x
  %va = load float, ptr addrspace(1) %pa, align 4
  br label %ja
ja:
  %ra = phi float [ %va, %la ], [ 0.0, %entry ]
  %cb = icmp ult i32 %x, %b
  br i1 %cb, label %lb, label %jb
lb:
  %x1 = add i32 %x, 1
  %pb = getelementptr inbounds float, ptr addrspace(1) %p, i32 %x1
  %vb = load float, ptr addrspace(1) %pb, align 4
  br label %jb
jb:
  %rb = phi float [ %vb, %lb ], [ 0.0, %ja ]
  %s = fadd float %ra, %rb
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i32 %x
  store float %s, ptr addrspace(1) %po, align 4
  ret void
}

; CHECK-LABEL: exit_unused:
; CHECK:       device_load
; CHECK:       wait 0
; CHECK-NEXT:  jmp_exec_none
; CHECK:       pop_exec
; CHECK-NEXT:  stop
define agx_kernel void @exit_unused(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %n) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr inbounds float, ptr addrspace(1) %p, i32 %x
  %v = load float, ptr addrspace(1) %pa, align 4
  %c = icmp ult i32 %x, %n
  br i1 %c, label %st, label %end
st:
  %w = fmul float %v, %v
  %w2 = fadd float %w, 3.0
  %w3 = fmul float %w2, %w
  %w4 = fadd float %w3, %v
  %w5 = fmul float %w4, %w2
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i32 %x
  store float %w5, ptr addrspace(1) %po, align 4
  br label %end
end:
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
