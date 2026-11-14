; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-known-grid=0 < %s | FileCheck %s --check-prefix=OFF

; A region whose latency schedule needs more registers than its original
; order is reverted where that lowers the threads a threadgroup may have;
; not where the known grid (64 threads: 2 SIMD-groups) resides on one core
; with the new registers anyway: the loads issue ahead of their adds
; (L15-E319).

; CHECK-LABEL: sum16:
; CHECK-COUNT-6: device_load
; CHECK: wait
; OFF-LABEL: sum16:
; OFF: device_load
; OFF-NEXT: wait
define agx_kernel void @sum16(ptr addrspace(1) %p, ptr addrspace(1) %o) #0 {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %b = mul i32 %t, 64
  %base = getelementptr <4 x float>, ptr addrspace(1) %p, i32 %b
  %a0 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 0
  %v0 = load <4 x float>, ptr addrspace(1) %a0, align 16
  %e0_0 = extractelement <4 x float> %v0, i32 0
  %s0_0 = fadd float 0.0, %e0_0
  %e0_1 = extractelement <4 x float> %v0, i32 1
  %s0_1 = fadd float %s0_0, %e0_1
  %e0_2 = extractelement <4 x float> %v0, i32 2
  %s0_2 = fadd float %s0_1, %e0_2
  %e0_3 = extractelement <4 x float> %v0, i32 3
  %s0_3 = fadd float %s0_2, %e0_3
  %a1 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 1
  %v1 = load <4 x float>, ptr addrspace(1) %a1, align 16
  %e1_0 = extractelement <4 x float> %v1, i32 0
  %s1_0 = fadd float %s0_3, %e1_0
  %e1_1 = extractelement <4 x float> %v1, i32 1
  %s1_1 = fadd float %s1_0, %e1_1
  %e1_2 = extractelement <4 x float> %v1, i32 2
  %s1_2 = fadd float %s1_1, %e1_2
  %e1_3 = extractelement <4 x float> %v1, i32 3
  %s1_3 = fadd float %s1_2, %e1_3
  %a2 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 2
  %v2 = load <4 x float>, ptr addrspace(1) %a2, align 16
  %e2_0 = extractelement <4 x float> %v2, i32 0
  %s2_0 = fadd float %s1_3, %e2_0
  %e2_1 = extractelement <4 x float> %v2, i32 1
  %s2_1 = fadd float %s2_0, %e2_1
  %e2_2 = extractelement <4 x float> %v2, i32 2
  %s2_2 = fadd float %s2_1, %e2_2
  %e2_3 = extractelement <4 x float> %v2, i32 3
  %s2_3 = fadd float %s2_2, %e2_3
  %a3 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 3
  %v3 = load <4 x float>, ptr addrspace(1) %a3, align 16
  %e3_0 = extractelement <4 x float> %v3, i32 0
  %s3_0 = fadd float %s2_3, %e3_0
  %e3_1 = extractelement <4 x float> %v3, i32 1
  %s3_1 = fadd float %s3_0, %e3_1
  %e3_2 = extractelement <4 x float> %v3, i32 2
  %s3_2 = fadd float %s3_1, %e3_2
  %e3_3 = extractelement <4 x float> %v3, i32 3
  %s3_3 = fadd float %s3_2, %e3_3
  %a4 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 4
  %v4 = load <4 x float>, ptr addrspace(1) %a4, align 16
  %e4_0 = extractelement <4 x float> %v4, i32 0
  %s4_0 = fadd float %s3_3, %e4_0
  %e4_1 = extractelement <4 x float> %v4, i32 1
  %s4_1 = fadd float %s4_0, %e4_1
  %e4_2 = extractelement <4 x float> %v4, i32 2
  %s4_2 = fadd float %s4_1, %e4_2
  %e4_3 = extractelement <4 x float> %v4, i32 3
  %s4_3 = fadd float %s4_2, %e4_3
  %a5 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 5
  %v5 = load <4 x float>, ptr addrspace(1) %a5, align 16
  %e5_0 = extractelement <4 x float> %v5, i32 0
  %s5_0 = fadd float %s4_3, %e5_0
  %e5_1 = extractelement <4 x float> %v5, i32 1
  %s5_1 = fadd float %s5_0, %e5_1
  %e5_2 = extractelement <4 x float> %v5, i32 2
  %s5_2 = fadd float %s5_1, %e5_2
  %e5_3 = extractelement <4 x float> %v5, i32 3
  %s5_3 = fadd float %s5_2, %e5_3
  %a6 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 6
  %v6 = load <4 x float>, ptr addrspace(1) %a6, align 16
  %e6_0 = extractelement <4 x float> %v6, i32 0
  %s6_0 = fadd float %s5_3, %e6_0
  %e6_1 = extractelement <4 x float> %v6, i32 1
  %s6_1 = fadd float %s6_0, %e6_1
  %e6_2 = extractelement <4 x float> %v6, i32 2
  %s6_2 = fadd float %s6_1, %e6_2
  %e6_3 = extractelement <4 x float> %v6, i32 3
  %s6_3 = fadd float %s6_2, %e6_3
  %a7 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 7
  %v7 = load <4 x float>, ptr addrspace(1) %a7, align 16
  %e7_0 = extractelement <4 x float> %v7, i32 0
  %s7_0 = fadd float %s6_3, %e7_0
  %e7_1 = extractelement <4 x float> %v7, i32 1
  %s7_1 = fadd float %s7_0, %e7_1
  %e7_2 = extractelement <4 x float> %v7, i32 2
  %s7_2 = fadd float %s7_1, %e7_2
  %e7_3 = extractelement <4 x float> %v7, i32 3
  %s7_3 = fadd float %s7_2, %e7_3
  %a8 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 8
  %v8 = load <4 x float>, ptr addrspace(1) %a8, align 16
  %e8_0 = extractelement <4 x float> %v8, i32 0
  %s8_0 = fadd float %s7_3, %e8_0
  %e8_1 = extractelement <4 x float> %v8, i32 1
  %s8_1 = fadd float %s8_0, %e8_1
  %e8_2 = extractelement <4 x float> %v8, i32 2
  %s8_2 = fadd float %s8_1, %e8_2
  %e8_3 = extractelement <4 x float> %v8, i32 3
  %s8_3 = fadd float %s8_2, %e8_3
  %a9 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 9
  %v9 = load <4 x float>, ptr addrspace(1) %a9, align 16
  %e9_0 = extractelement <4 x float> %v9, i32 0
  %s9_0 = fadd float %s8_3, %e9_0
  %e9_1 = extractelement <4 x float> %v9, i32 1
  %s9_1 = fadd float %s9_0, %e9_1
  %e9_2 = extractelement <4 x float> %v9, i32 2
  %s9_2 = fadd float %s9_1, %e9_2
  %e9_3 = extractelement <4 x float> %v9, i32 3
  %s9_3 = fadd float %s9_2, %e9_3
  %a10 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 10
  %v10 = load <4 x float>, ptr addrspace(1) %a10, align 16
  %e10_0 = extractelement <4 x float> %v10, i32 0
  %s10_0 = fadd float %s9_3, %e10_0
  %e10_1 = extractelement <4 x float> %v10, i32 1
  %s10_1 = fadd float %s10_0, %e10_1
  %e10_2 = extractelement <4 x float> %v10, i32 2
  %s10_2 = fadd float %s10_1, %e10_2
  %e10_3 = extractelement <4 x float> %v10, i32 3
  %s10_3 = fadd float %s10_2, %e10_3
  %a11 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 11
  %v11 = load <4 x float>, ptr addrspace(1) %a11, align 16
  %e11_0 = extractelement <4 x float> %v11, i32 0
  %s11_0 = fadd float %s10_3, %e11_0
  %e11_1 = extractelement <4 x float> %v11, i32 1
  %s11_1 = fadd float %s11_0, %e11_1
  %e11_2 = extractelement <4 x float> %v11, i32 2
  %s11_2 = fadd float %s11_1, %e11_2
  %e11_3 = extractelement <4 x float> %v11, i32 3
  %s11_3 = fadd float %s11_2, %e11_3
  %a12 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 12
  %v12 = load <4 x float>, ptr addrspace(1) %a12, align 16
  %e12_0 = extractelement <4 x float> %v12, i32 0
  %s12_0 = fadd float %s11_3, %e12_0
  %e12_1 = extractelement <4 x float> %v12, i32 1
  %s12_1 = fadd float %s12_0, %e12_1
  %e12_2 = extractelement <4 x float> %v12, i32 2
  %s12_2 = fadd float %s12_1, %e12_2
  %e12_3 = extractelement <4 x float> %v12, i32 3
  %s12_3 = fadd float %s12_2, %e12_3
  %a13 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 13
  %v13 = load <4 x float>, ptr addrspace(1) %a13, align 16
  %e13_0 = extractelement <4 x float> %v13, i32 0
  %s13_0 = fadd float %s12_3, %e13_0
  %e13_1 = extractelement <4 x float> %v13, i32 1
  %s13_1 = fadd float %s13_0, %e13_1
  %e13_2 = extractelement <4 x float> %v13, i32 2
  %s13_2 = fadd float %s13_1, %e13_2
  %e13_3 = extractelement <4 x float> %v13, i32 3
  %s13_3 = fadd float %s13_2, %e13_3
  %a14 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 14
  %v14 = load <4 x float>, ptr addrspace(1) %a14, align 16
  %e14_0 = extractelement <4 x float> %v14, i32 0
  %s14_0 = fadd float %s13_3, %e14_0
  %e14_1 = extractelement <4 x float> %v14, i32 1
  %s14_1 = fadd float %s14_0, %e14_1
  %e14_2 = extractelement <4 x float> %v14, i32 2
  %s14_2 = fadd float %s14_1, %e14_2
  %e14_3 = extractelement <4 x float> %v14, i32 3
  %s14_3 = fadd float %s14_2, %e14_3
  %a15 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 15
  %v15 = load <4 x float>, ptr addrspace(1) %a15, align 16
  %e15_0 = extractelement <4 x float> %v15, i32 0
  %s15_0 = fadd float %s14_3, %e15_0
  %e15_1 = extractelement <4 x float> %v15, i32 1
  %s15_1 = fadd float %s15_0, %e15_1
  %e15_2 = extractelement <4 x float> %v15, i32 2
  %s15_2 = fadd float %s15_1, %e15_2
  %e15_3 = extractelement <4 x float> %v15, i32 3
  %s15_3 = fadd float %s15_2, %e15_3
  %q = getelementptr float, ptr addrspace(1) %o, i32 %t
  store float %s15_3, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
attributes #0 = { "agx-max-threads-per-threadgroup"="64" "agx-threads-per-grid"="64,1,1" "agx-threads-per-threadgroup"="64,1,1" }
