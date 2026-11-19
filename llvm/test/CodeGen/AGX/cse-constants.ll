; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-cse-constants=0 < %s | FileCheck %s --check-prefix=OFF
;
; A constant read only as an operand is reused from any dominating block, not
; only from its own or an immediate predecessor (MachineCSE's heuristic for
; instructions as cheap as a move): one mov_imm of 1024 for the three guards.
; The zeros the masked loads keep where they do not run stay one per load
; (L15-E451).

; CHECK-LABEL: guards:
; CHECK:       mov_imm r4, 1024
; CHECK-NOT:   mov_imm {{r[0-9]+}}, 1024
; CHECK:       if_icmp r0l, ult, r6, r4, 1
; CHECK:       stop
; OFF-LABEL:   guards:
; OFF:         mov_imm r4, 1024
; OFF:         mov_imm r6, 1024
; OFF-NEXT:    mov_imm r4, 0
; OFF-NEXT:    if_icmp r0l, ult, r5, r6, 1

target triple = "agx"

declare i32 @llvm.agx.thread.position.in.grid.x()

; three guarded loads, each guard an unsigned compare with 1024
define agx_kernel void @guards(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %a, i32 %b, i32 %c) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %ia = add i32 %t, %a
  %ga = icmp ult i32 %ia, 1024
  br i1 %ga, label %la, label %ja
la:
  %pa = getelementptr float, ptr addrspace(1) %p, i32 %ia
  %va = load float, ptr addrspace(1) %pa
  br label %ja
ja:
  %xa = phi float [ %va, %la ], [ 0.0, %entry ]
  %ib = add i32 %t, %b
  %gb = icmp ult i32 %ib, 1024
  br i1 %gb, label %lb, label %jb
lb:
  %pb = getelementptr float, ptr addrspace(1) %p, i32 %ib
  %vb = load float, ptr addrspace(1) %pb
  br label %jb
jb:
  %xb = phi float [ %vb, %lb ], [ 0.0, %ja ]
  %ic = add i32 %t, %c
  %gc = icmp ult i32 %ic, 1024
  br i1 %gc, label %lc, label %jc
lc:
  %pc = getelementptr float, ptr addrspace(1) %p, i32 %ic
  %vc = load float, ptr addrspace(1) %pc
  br label %jc
jc:
  %xc = phi float [ %vc, %lc ], [ 0.0, %jb ]
  %s1 = fadd float %xa, %xb
  %s2 = fadd float %s1, %xc
  %pq = getelementptr float, ptr addrspace(1) %q, i32 %t
  store float %s2, ptr addrspace(1) %pq
  ret void
}
