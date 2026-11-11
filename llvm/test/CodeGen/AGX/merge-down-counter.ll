; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-down-counters=0 < %s | FileCheck %s --check-prefix=OFF

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
