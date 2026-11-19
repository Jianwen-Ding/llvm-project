; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-max-pending-loads=8 < %s | FileCheck %s --check-prefix=EIGHT

; Twelve independent loads stay in flight together (at most sixteen, L15-E437):
; the first wait comes before their first use, not after the eighth load.

; CHECK-LABEL: twelve:
; CHECK-COUNT-12: device_load
; CHECK-NOT: device_load
; CHECK: wait
; EIGHT-LABEL: twelve:
; EIGHT-COUNT-8: device_load
; EIGHT: wait
; EIGHT: device_load
define agx_kernel void @twelve(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %i0 = add i32 %t, 0
  %a0 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i0
  %v0 = load float, ptr addrspace(1) %a0, align 4
  %i1 = add i32 %t, 4096
  %a1 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i1
  %v1 = load float, ptr addrspace(1) %a1, align 4
  %i2 = add i32 %t, 8192
  %a2 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i2
  %v2 = load float, ptr addrspace(1) %a2, align 4
  %i3 = add i32 %t, 12288
  %a3 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i3
  %v3 = load float, ptr addrspace(1) %a3, align 4
  %i4 = add i32 %t, 16384
  %a4 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i4
  %v4 = load float, ptr addrspace(1) %a4, align 4
  %i5 = add i32 %t, 20480
  %a5 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i5
  %v5 = load float, ptr addrspace(1) %a5, align 4
  %i6 = add i32 %t, 24576
  %a6 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i6
  %v6 = load float, ptr addrspace(1) %a6, align 4
  %i7 = add i32 %t, 28672
  %a7 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i7
  %v7 = load float, ptr addrspace(1) %a7, align 4
  %i8 = add i32 %t, 32768
  %a8 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i8
  %v8 = load float, ptr addrspace(1) %a8, align 4
  %i9 = add i32 %t, 36864
  %a9 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i9
  %v9 = load float, ptr addrspace(1) %a9, align 4
  %i10 = add i32 %t, 40960
  %a10 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i10
  %v10 = load float, ptr addrspace(1) %a10, align 4
  %i11 = add i32 %t, 45056
  %a11 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i11
  %v11 = load float, ptr addrspace(1) %a11, align 4
  %s1 = fadd float %v0, %v1
  %s2 = fadd float %s1, %v2
  %s3 = fadd float %s2, %v3
  %s4 = fadd float %s3, %v4
  %s5 = fadd float %s4, %v5
  %s6 = fadd float %s5, %v6
  %s7 = fadd float %s6, %v7
  %s8 = fadd float %s7, %v8
  %s9 = fadd float %s8, %v9
  %s10 = fadd float %s9, %v10
  %s11 = fadd float %s10, %v11
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i32 %t
  store float %s11, ptr addrspace(1) %po, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
