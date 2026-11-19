; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-align-versioning < %s | FileCheck %s

; A loop of guarded taps at inttoptr(k_j + p) (a convolution's, inside-the-
; image tests): versionGuards' copy for SIMD-groups passing every test takes
; the taps from one base, the fit tests joined to the copy's own test (its
; ballot), with no third copy (L15-E408).

; CHECK-LABEL: @guarded(
; CHECK:       call i32 @llvm.agx.simd.ballot(i1
; CHECK:       %onebase = inttoptr
; CHECK-NOT:   onebase:
; CHECK:       ret void

target triple = "agx"

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @guarded(ptr addrspace(1) %out, i64 %in, i64 %stride, i32 %c, i64 %o0, i64 %o1, i64 %o2, i64 %o3, i1 %g0, i1 %g1, i1 %g2, i1 %g3) {
entry:
  %tid = call i32 @llvm.agx.thread.position.in.grid.x()
  %t = sext i32 %tid to i64
  %t4 = shl nsw i64 %t, 2
  %b = add i64 %in, %t4
  %k0 = add i64 %b, %o0
  %k1 = add i64 %b, %o1
  %k2 = add i64 %b, %o2
  %k3 = add i64 %b, %o3
  br label %loop

loop:
  %p = phi i64 [ 0, %entry ], [ %p.next, %latch ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %s = phi float [ 0.0, %entry ], [ %s3, %latch ]
  br label %t0
t0:
  br i1 %g0, label %do0, label %join0
do0:
  %a0 = add i64 %k0, %p
  %q0 = inttoptr i64 %a0 to ptr addrspace(1)
  %v0 = load float, ptr addrspace(1) %q0, align 4
  %f0 = fadd float %s, %v0
  br label %join0
join0:
  %s0 = phi float [ %s, %t0 ], [ %f0, %do0 ]
  br label %t1
t1:
  br i1 %g1, label %do1, label %join1
do1:
  %a1 = add i64 %k1, %p
  %q1 = inttoptr i64 %a1 to ptr addrspace(1)
  %v1 = load float, ptr addrspace(1) %q1, align 4
  %f1 = fadd float %s0, %v1
  br label %join1
join1:
  %s1 = phi float [ %s0, %t1 ], [ %f1, %do1 ]
  br label %t2
t2:
  br i1 %g2, label %do2, label %join2
do2:
  %a2 = add i64 %k2, %p
  %q2 = inttoptr i64 %a2 to ptr addrspace(1)
  %v2 = load float, ptr addrspace(1) %q2, align 4
  %f2 = fadd float %s1, %v2
  br label %join2
join2:
  %s2 = phi float [ %s1, %t2 ], [ %f2, %do2 ]
  br label %t3
t3:
  br i1 %g3, label %do3, label %join3
do3:
  %a3 = add i64 %k3, %p
  %q3 = inttoptr i64 %a3 to ptr addrspace(1)
  %v3 = load float, ptr addrspace(1) %q3, align 4
  %f3 = fadd float %s2, %v3
  br label %join3
join3:
  %s3 = phi float [ %s2, %t3 ], [ %f3, %do3 ]
  br label %latch
latch:
  %p.next = add i64 %p, %stride
  %i.next = add i32 %i, 1
  %c1 = icmp slt i32 %i.next, %c
  br i1 %c1, label %loop, label %exit

exit:
  %o = getelementptr float, ptr addrspace(1) %out, i32 %tid
  store float %s3, ptr addrspace(1) %o, align 4
  ret void
}
