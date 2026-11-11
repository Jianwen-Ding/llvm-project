; RUN: llc -mtriple=agx -agx-fold-condition-chains=0 -agx-preisel-simplifycfg=0 -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -agx-fold-condition-chains=0 -agx-preisel-simplifycfg=0 -mcpu=agx-g14c -verify-machineinstrs \
; RUN:   -agx-fold-widened-branch-conds=0 < %s | FileCheck %s --check-prefix=OFF

; A boolean phi (kept as i16 by type legalization) that a later if tests:
; the if compares its 16 bits, instead of widening them into a 32-bit
; register first.

; CHECK-LABEL: phi_cond:
; CHECK-NOT: bitop 0101, {{r[0-9]+}}, {{r[0-9]+[lh]}}, 0
; CHECK: if_icmp r0l, nueq, {{r[0-9]+[lh]}}, 0, 1
; OFF-LABEL: phi_cond:
; OFF: bitop 0101, [[W:r[0-9]+]], {{r[0-9]+[lh]}}, 0
; OFF-NEXT: if_icmp r0l, nueq, [[W]], 0, 1
define agx_kernel void @phi_cond(ptr addrspace(1) %p, float %a, float %b) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %lt = fcmp olt float %x, %a
  br i1 %lt, label %done, label %check
check:
  %eq = fcmp oeq float %x, %b
  br label %done
done:
  %c = phi i1 [ true, %entry ], [ %eq, %check ]
  br i1 %c, label %then, label %end
then:
  store float %a, ptr addrspace(1) %q, align 4
  br label %end
end:
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
