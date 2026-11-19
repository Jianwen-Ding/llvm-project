; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-align-versioning < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-tap-bases-min=0 < %s | FileCheck %s --check-prefix=OFF

; The PTX front end's integer addresses of a convolution's taps, inttoptr(k_j + p)
; with p stepped by the loop and five invariant k_j: loop strength reduction
; keeps a 64-bit pointer for each, each stepped by a 64-bit add. A copy of the
; loop addresses them from inttoptr(k_0 + p) and 32-bit element indices
; (k_j - k_0) / 4, where every difference is a multiple of 4 whose quotient fits
; 32 signed bits for the whole SIMD-group (L15-E395); the loop as it was
; elsewhere. @index's p is an extended 32-bit index scaled by 4, which loop
; strength reduction already takes as the accesses' index register: no copy
; (L15-E396).

; CHECK-LABEL: @taps(
; CHECK:       call i32 @llvm.agx.simd.ballot
; CHECK:       loop.onebase:
; CHECK:       %onebase = inttoptr
; CHECK-COUNT-4: getelementptr i32, ptr addrspace(1) %onebase, i64
; ASM-LABEL:   taps:
; ASM:         Loop Header
; ASM-COUNT-5: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}
; ASM:         jmp_exec_any
; ASM:         Loop Header
; ASM:         iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}
; ASM-NOT:     iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}
; ASM:         jmp_exec_any
; OFF-LABEL:   taps:
; OFF:         Loop Header
; OFF-COUNT-5: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}
; OFF-NOT:     Loop Header

target triple = "agx"

declare float @llvm.fma.f32(float, float, float)
declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @taps(ptr addrspace(1) %out, i64 %in, i64 %stride, i32 %c, i64 %o0, i64 %o1, i64 %o2, i64 %o3, i64 %o4) {
entry:
  %tid = call i32 @llvm.agx.thread.position.in.grid.x()
  %t = sext i32 %tid to i64
  %t4 = shl nsw i64 %t, 2
  %b = add i64 %in, %t4
  %k0 = add i64 %b, %o0
  %k1 = add i64 %b, %o1
  %k2 = add i64 %b, %o2
  %k3 = add i64 %b, %o3
  %k4 = add i64 %b, %o4
  br label %loop

loop:
  %p = phi i64 [ 0, %entry ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s4, %loop ]
  %a0 = add i64 %k0, %p
  %q0 = inttoptr i64 %a0 to ptr addrspace(1)
  %v0 = load float, ptr addrspace(1) %q0, align 4
  %s0 = call float @llvm.fma.f32(float %v0, float %v0, float %s)
  %a1 = add i64 %k1, %p
  %q1 = inttoptr i64 %a1 to ptr addrspace(1)
  %v1 = load float, ptr addrspace(1) %q1, align 4
  %s1 = call float @llvm.fma.f32(float %v1, float %v1, float %s0)
  %a2 = add i64 %k2, %p
  %q2 = inttoptr i64 %a2 to ptr addrspace(1)
  %v2 = load float, ptr addrspace(1) %q2, align 4
  %s2 = call float @llvm.fma.f32(float %v2, float %v2, float %s1)
  %a3 = add i64 %k3, %p
  %q3 = inttoptr i64 %a3 to ptr addrspace(1)
  %v3 = load float, ptr addrspace(1) %q3, align 4
  %s3 = call float @llvm.fma.f32(float %v3, float %v3, float %s2)
  %a4 = add i64 %k4, %p
  %q4 = inttoptr i64 %a4 to ptr addrspace(1)
  %v4 = load float, ptr addrspace(1) %q4, align 4
  %s4 = call float @llvm.fma.f32(float %v4, float %v4, float %s3)
  %p.next = add i64 %p, %stride
  %i.next = add i32 %i, 1
  %c1 = icmp slt i32 %i.next, %c
  br i1 %c1, label %loop, label %exit

exit:
  %o = getelementptr float, ptr addrspace(1) %out, i32 %tid
  store float %s4, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: @index(
; CHECK-NOT:   onebase
; OFF-LABEL:   index:
; CHECK:       ret void
define agx_kernel void @index(ptr addrspace(1) %out, i64 %in, i32 %stride, i32 %c, i64 %o0, i64 %o1, i64 %o2, i64 %o3, i64 %o4) {
entry:
  %tid = call i32 @llvm.agx.thread.position.in.grid.x()
  %k0 = add i64 %in, %o0
  %k1 = add i64 %in, %o1
  %k2 = add i64 %in, %o2
  %k3 = add i64 %in, %o3
  %k4 = add i64 %in, %o4
  br label %loop

loop:
  %j = phi i32 [ %tid, %entry ], [ %j.next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s4, %loop ]
  %je = sext i32 %j to i64
  %p = shl nsw i64 %je, 2
  %a0 = add i64 %k0, %p
  %q0 = inttoptr i64 %a0 to ptr addrspace(1)
  %v0 = load float, ptr addrspace(1) %q0, align 4
  %s0 = call float @llvm.fma.f32(float %v0, float %v0, float %s)
  %a1 = add i64 %k1, %p
  %q1 = inttoptr i64 %a1 to ptr addrspace(1)
  %v1 = load float, ptr addrspace(1) %q1, align 4
  %s1 = call float @llvm.fma.f32(float %v1, float %v1, float %s0)
  %a2 = add i64 %k2, %p
  %q2 = inttoptr i64 %a2 to ptr addrspace(1)
  %v2 = load float, ptr addrspace(1) %q2, align 4
  %s2 = call float @llvm.fma.f32(float %v2, float %v2, float %s1)
  %a3 = add i64 %k3, %p
  %q3 = inttoptr i64 %a3 to ptr addrspace(1)
  %v3 = load float, ptr addrspace(1) %q3, align 4
  %s3 = call float @llvm.fma.f32(float %v3, float %v3, float %s2)
  %a4 = add i64 %k4, %p
  %q4 = inttoptr i64 %a4 to ptr addrspace(1)
  %v4 = load float, ptr addrspace(1) %q4, align 4
  %s4 = call float @llvm.fma.f32(float %v4, float %v4, float %s3)
  %j.next = add i32 %j, %stride
  %i.next = add i32 %i, 1
  %c1 = icmp slt i32 %i.next, %c
  br i1 %c1, label %loop, label %exit

exit:
  %o = getelementptr float, ptr addrspace(1) %out, i32 %tid
  store float %s4, ptr addrspace(1) %o, align 4
  ret void
}
