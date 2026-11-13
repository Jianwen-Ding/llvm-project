; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-alignment-versioning=0 < %s | FileCheck %s --check-prefix=OFF

; A loop over a half row known aligned to 2 bytes only: a copy for a 4-byte
; aligned base (a test of the kernel argument, uniform) whose loads at
; offsets that are multiples of 4 merge into 4-byte (and wider) loads, and
; the loop as it was otherwise (L15-E236). Bytes are left alone.

; CHECK-LABEL: rowsum:
; CHECK: and {{r[0-9]+}}, u0, 3
; CHECK: Loop Header
; CHECK: device_load {{[0-9]}}, i16, x,
; CHECK: Loop Header
; CHECK: device_load {{[0-9]}}, i32, xy,
; CHECK: stop
; OFF-LABEL: rowsum:
; OFF-NOT: and {{r[0-9]+}}, u0, 3
; OFF-NOT: i32, xy
; OFF: stop
define agx_kernel void @rowsum(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %l
l:
  %i = phi i32 [ 0, %entry ], [ %i1, %l ]
  %s = phi float [ 0.0, %entry ], [ %s4, %l ]
  %b = shl nuw nsw i32 %i, 2
  %z0 = zext i32 %b to i64
  %g0 = getelementptr inbounds half, ptr addrspace(1) %p, i64 %z0
  %v0 = load half, ptr addrspace(1) %g0, align 2
  %g1 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 1
  %v1 = load half, ptr addrspace(1) %g1, align 2
  %g2 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 2
  %v2 = load half, ptr addrspace(1) %g2, align 2
  %g3 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 3
  %v3 = load half, ptr addrspace(1) %g3, align 2
  %g4 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 4
  %v4 = load half, ptr addrspace(1) %g4, align 2
  %g6 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 6
  %v6 = load half, ptr addrspace(1) %g6, align 2
  %e0 = fpext half %v0 to float
  %e1 = fpext half %v1 to float
  %e2 = fpext half %v2 to float
  %e3 = fpext half %v3 to float
  %e4 = fpext half %v4 to float
  %e6 = fpext half %v6 to float
  %s0 = fadd float %s, %e0
  %s1 = fadd float %s0, %e1
  %s2 = fadd float %s1, %e2
  %s3 = fadd float %s2, %e3
  %s5 = fadd float %s3, %e4
  %s4 = fadd float %s5, %e6
  %i1 = add nuw nsw i32 %i, 2
  %c = icmp ult i32 %i1, 256
  br i1 %c, label %l, label %exit
exit:
  store float %s4, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: bytes:
; CHECK-NOT: and {{r[0-9]+}}, u0, 3
; CHECK: stop
define agx_kernel void @bytes(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %l
l:
  %i = phi i32 [ 0, %entry ], [ %i1, %l ]
  %s = phi i32 [ 0, %entry ], [ %t3, %l ]
  %z0 = zext i32 %i to i64
  %g0 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %z0
  %v0 = load i8, ptr addrspace(1) %g0, align 1
  %g1 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 1
  %v1 = load i8, ptr addrspace(1) %g1, align 1
  %g2 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 2
  %v2 = load i8, ptr addrspace(1) %g2, align 1
  %g3 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 3
  %v3 = load i8, ptr addrspace(1) %g3, align 1
  %e0 = zext i8 %v0 to i32
  %e1 = zext i8 %v1 to i32
  %e2 = zext i8 %v2 to i32
  %e3 = zext i8 %v3 to i32
  %t0 = add i32 %s, %e0
  %t1 = add i32 %t0, %e1
  %t2 = add i32 %t1, %e2
  %t3 = add i32 %t2, %e3
  %i1 = add nuw nsw i32 %i, 4
  %c = icmp ult i32 %i1, 1024
  br i1 %c, label %l, label %exit
exit:
  store i32 %t3, ptr addrspace(1) %o, align 4
  ret void
}
