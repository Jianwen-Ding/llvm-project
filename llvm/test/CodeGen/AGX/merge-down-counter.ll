; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-down-counters=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-down-counters-by-step=0 < %s | FileCheck %s --check-prefix=BYSTEPOFF

; LSR counts this loop down to zero beside the index it steps: an isub each
; iteration. The counter only the exit test uses, and the index's step odd,
; the exit is the index reaching its start + n (exact mod 2^32): one add per
; iteration fewer (L15-E206). An even step keeps the counter (s c' = s n
; would also hold at other iterations mod 2^32): even2.

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.maxnum.f32(float, float)
declare i32 @llvm.umax.i32(i32, i32)

; CHECK-LABEL: rowmax:
; CHECK: Loop Header
; CHECK-NOT: isub
; CHECK: while_icmp r0l, nueq, r{{[0-9]+}}, r{{[0-9]+}}, 1
; OFF-LABEL: rowmax:
; OFF: Loop Header
; OFF: isub
define agx_kernel void @rowmax(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %n, %r
  %umax = call i32 @llvm.umax.i32(i32 %n, i32 1)
  br label %l
l:
  %j = phi i32 [ 0, %entry ], [ %j1, %l ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %l ]
  %ix = add i32 %j, %row
  %e = sext i32 %ix to i64
  %p = getelementptr float, ptr addrspace(1) %x, i64 %e
  %v = load float, ptr addrspace(1) %p, align 4
  %m1 = call float @llvm.maxnum.f32(float %m, float %v)
  %j1 = add nuw i32 %j, 1
  %c = icmp eq i32 %j1, %umax
  br i1 %c, label %done, label %l
done:
  %q = getelementptr float, ptr addrspace(1) %o, i32 %r
  store float %m1, ptr addrspace(1) %q, align 4
  ret void
}

; CHECK-LABEL: even2:
; CHECK: Loop Header
; CHECK: isub
define agx_kernel void @even2(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %n, %r
  %umax = call i32 @llvm.umax.i32(i32 %n, i32 1)
  br label %l
l:
  %j = phi i32 [ 0, %entry ], [ %j1, %l ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %l ]
  %j2 = shl i32 %j, 1
  %ix = add i32 %j2, %row
  %e = sext i32 %ix to i64
  %p = getelementptr float, ptr addrspace(1) %x, i64 %e
  %v = load float, ptr addrspace(1) %p, align 4
  %m1 = call float @llvm.maxnum.f32(float %m, float %v)
  %j1 = add nuw i32 %j, 1
  %c = icmp eq i32 %j1, %umax
  br i1 %c, label %done, label %l
done:
  %q = getelementptr float, ptr addrspace(1) %o, i32 %r
  store float %m1, ptr addrspace(1) %q, align 4
  ret void
}

; An unrolled loop's counter counts down by the unroll factor (4 here) from
; a constant it divides: c' = 0 after 64 / 4 iterations, so the exit is the
; index reaching its start + 16 steps (L15-E439).

; CHECK-LABEL: colsum4:
; CHECK:       [[LOOP:\.LBB[0-9_]+]]:
; CHECK-NOT:   isub
; CHECK:       while_icmp r0l, nueq,
; CHECK-NEXT:  jmp_exec_any [[LOOP]]
; BYSTEPOFF-LABEL: colsum4:
; BYSTEPOFF:   isub
define agx_kernel void @colsum4(ptr addrspace(1) %x, ptr addrspace(1) %o) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %ivn, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %a3, %loop ]
  %i = trunc i64 %iv to i32
  %r0 = add i32 %i, 0
  %s0 = shl i32 %r0, 10
  %e0 = add i32 %s0, %t
  %z0 = zext i32 %e0 to i64
  %p0 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z0
  %v0 = load float, ptr addrspace(1) %p0, align 4
  %a0 = fadd float %acc, %v0
  %r1 = add i32 %i, 1
  %s1 = shl i32 %r1, 10
  %e1 = add i32 %s1, %t
  %z1 = zext i32 %e1 to i64
  %p1 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z1
  %v1 = load float, ptr addrspace(1) %p1, align 4
  %a1 = fadd float %a0, %v1
  %r2 = add i32 %i, 2
  %s2 = shl i32 %r2, 10
  %e2 = add i32 %s2, %t
  %z2 = zext i32 %e2 to i64
  %p2 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z2
  %v2 = load float, ptr addrspace(1) %p2, align 4
  %a2 = fadd float %a1, %v2
  %r3 = add i32 %i, 3
  %s3 = shl i32 %r3, 10
  %e3 = add i32 %s3, %t
  %z3 = zext i32 %e3 to i64
  %p3 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z3
  %v3 = load float, ptr addrspace(1) %p3, align 4
  %a3 = fadd float %a2, %v3
  %ivn = add nuw nsw i64 %iv, 4
  %c = icmp eq i64 %ivn, 64
  br i1 %c, label %done, label %loop
done:
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i32 %t
  store float %a3, ptr addrspace(1) %po, align 4
  ret void
}

attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" }
