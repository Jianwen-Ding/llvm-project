; REQUIRES: asserts
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -debug-only=agx-mem-cluster -o /dev/null < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -debug-only=agx-mem-cluster -agx-sched-heavy-stores-apart=128 -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=LIGHT --allow-empty
;
; A region of more than 64 instructions for every 32-bit word it stores
; clusters no device stores (L15-E648): held together, each store waits for
; every value of the run, all live at once (logpow_rows with E648's shorter
; pow: 34 -> 54 GPRs). Here exp, log, sin and cos of four vectors.
; CHECK: agx-mem-cluster: heavy stores apart in %bb.{{[0-9]+}}
; LIGHT-NOT: heavy stores apart
define agx_kernel void @hs(ptr addrspace(1) %p, ptr addrspace(1) %q) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %t, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %jn, %loop ]
  %i0 = add i32 %row, %j
  %i0x = add i32 %i0, 0
  %a0 = getelementptr float, ptr addrspace(1) %p, i32 %i0x
  %v0 = load <4 x float>, ptr addrspace(1) %a0, align 16
  %i1x = add i32 %i0, 4
  %a1 = getelementptr float, ptr addrspace(1) %p, i32 %i1x
  %v1 = load <4 x float>, ptr addrspace(1) %a1, align 16
  %i2x = add i32 %i0, 8
  %a2 = getelementptr float, ptr addrspace(1) %p, i32 %i2x
  %v2 = load <4 x float>, ptr addrspace(1) %a2, align 16
  %i3x = add i32 %i0, 12
  %a3 = getelementptr float, ptr addrspace(1) %p, i32 %i3x
  %v3 = load <4 x float>, ptr addrspace(1) %a3, align 16
  %e0 = call <4 x float> @llvm.exp.v4f32(<4 x float> %v0)
  %l0 = call <4 x float> @llvm.log.v4f32(<4 x float> %v0)
  %n0 = call <4 x float> @llvm.sin.v4f32(<4 x float> %v0)
  %o0 = call <4 x float> @llvm.cos.v4f32(<4 x float> %v0)
  %s0a = fadd <4 x float> %e0, %l0
  %s0b = fadd <4 x float> %n0, %o0
  %s0 = fadd <4 x float> %s0a, %s0b
  %b0 = getelementptr float, ptr addrspace(1) %q, i32 %i0x
  store <4 x float> %s0, ptr addrspace(1) %b0, align 16
  %e1 = call <4 x float> @llvm.exp.v4f32(<4 x float> %v1)
  %l1 = call <4 x float> @llvm.log.v4f32(<4 x float> %v1)
  %n1 = call <4 x float> @llvm.sin.v4f32(<4 x float> %v1)
  %o1 = call <4 x float> @llvm.cos.v4f32(<4 x float> %v1)
  %s1a = fadd <4 x float> %e1, %l1
  %s1b = fadd <4 x float> %n1, %o1
  %s1 = fadd <4 x float> %s1a, %s1b
  %b1 = getelementptr float, ptr addrspace(1) %q, i32 %i1x
  store <4 x float> %s1, ptr addrspace(1) %b1, align 16
  %e2 = call <4 x float> @llvm.exp.v4f32(<4 x float> %v2)
  %l2 = call <4 x float> @llvm.log.v4f32(<4 x float> %v2)
  %n2 = call <4 x float> @llvm.sin.v4f32(<4 x float> %v2)
  %o2 = call <4 x float> @llvm.cos.v4f32(<4 x float> %v2)
  %s2a = fadd <4 x float> %e2, %l2
  %s2b = fadd <4 x float> %n2, %o2
  %s2 = fadd <4 x float> %s2a, %s2b
  %b2 = getelementptr float, ptr addrspace(1) %q, i32 %i2x
  store <4 x float> %s2, ptr addrspace(1) %b2, align 16
  %e3 = call <4 x float> @llvm.exp.v4f32(<4 x float> %v3)
  %l3 = call <4 x float> @llvm.log.v4f32(<4 x float> %v3)
  %n3 = call <4 x float> @llvm.sin.v4f32(<4 x float> %v3)
  %o3 = call <4 x float> @llvm.cos.v4f32(<4 x float> %v3)
  %s3a = fadd <4 x float> %e3, %l3
  %s3b = fadd <4 x float> %n3, %o3
  %s3 = fadd <4 x float> %s3a, %s3b
  %b3 = getelementptr float, ptr addrspace(1) %q, i32 %i3x
  store <4 x float> %s3, ptr addrspace(1) %b3, align 16
  %jn = add nuw nsw i32 %j, 16
  %c = icmp ult i32 %jn, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare <4 x float> @llvm.exp.v4f32(<4 x float>)
declare <4 x float> @llvm.log.v4f32(<4 x float>)
declare <4 x float> @llvm.sin.v4f32(<4 x float>)
declare <4 x float> @llvm.cos.v4f32(<4 x float>)
