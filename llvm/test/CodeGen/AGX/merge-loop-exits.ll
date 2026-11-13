; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-loop-exits=0 < %s | FileCheck %s --check-prefix=OFF

; Mandelbrot's escape loop exits from its header (|z|^2 >= 4) and from its
; latch (the iteration count): structurized, an if inside the loop every
; iteration. The latch's arithmetic runs in the header, the loop exits once
; on either test, and a block after it picks the exit (L15-E245).

declare float @llvm.fma.f32(float, float, float)

; CHECK-LABEL: escape:
; CHECK: Loop Header
; CHECK-NOT: if_
; CHECK: while_icmp
; CHECK-NEXT: jmp_exec_any
; OFF-LABEL: escape:
; OFF: Loop Header
; OFF: if_fcmp
; OFF: while_icmp
define agx_kernel void @escape(ptr addrspace(1) %o, i32 %maxit, float %cr, float %ci) {
entry:
  %pos = icmp sgt i32 %maxit, 0
  br i1 %pos, label %h, label %j
h:
  %it = phi i32 [ 0, %entry ], [ %it1, %t ]
  %zr = phi float [ 0.0, %entry ], [ %zr1, %t ]
  %zi = phi float [ 0.0, %entry ], [ %zi1, %t ]
  %a = fmul float %zr, %zr
  %m = call float @llvm.fma.f32(float %zi, float %zi, float %a)
  %e1 = fcmp uge float %m, 4.0
  br i1 %e1, label %xh, label %t
t:
  %b = fmul float %zi, %zi
  %d = fsub float %b, %a
  %zi1 = fadd float %cr, %d
  %two = fadd float %zi, %zi
  %zr1 = call float @llvm.fma.f32(float %two, float %zr, float %ci)
  %it1 = add nuw nsw i32 %it, 1
  %e2 = icmp eq i32 %maxit, %it1
  br i1 %e2, label %xt, label %h
xt:
  br label %j
xh:
  br label %j
j:
  %r = phi i32 [ 0, %entry ], [ %maxit, %xt ], [ %it, %xh ]
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
}

; A latch with a division (not safe to run where the loop left by the
; header's test) keeps both exits.
; CHECK-LABEL: divides:
; CHECK: Loop Header
; CHECK: if_
define agx_kernel void @divides(ptr addrspace(1) %o, i32 %n, i32 %k) {
entry:
  br label %h
h:
  %i = phi i32 [ 0, %entry ], [ %i1, %t ]
  %s = phi i32 [ 0, %entry ], [ %s1, %t ]
  %e1 = icmp eq i32 %i, %k
  br i1 %e1, label %xh, label %t
t:
  %q = udiv i32 %s, %i
  %s1 = add i32 %q, 3
  %i1 = add i32 %i, 1
  %e2 = icmp eq i32 %i1, %n
  br i1 %e2, label %xt, label %h
xt:
  br label %j
xh:
  br label %j
j:
  %r = phi i32 [ %s1, %xt ], [ %s, %xh ]
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
}
