; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-dead-after-narrowing=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's naive matmul after LSR: a 64-bit down counter (at most 2^20, the
; memref's size) beside two 32-bit indices. Narrowed to 32 bits, it left its
; old sign extensions dead in the loop, uses that kept the counter from
; merging into the index stepped by 1 (L15-E206); deleted, the exit is the
; index reaching its end, one isub per iteration fewer (L15-E231).

declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.grid.y()
declare float @llvm.fma.f32(float, float, float)
declare i64 @llvm.umin.i64(i64, i64)

; CHECK-LABEL: matmul:
; CHECK: Loop Header
; CHECK-NOT: isub
; CHECK: while_icmp r0l, nueq, r{{[0-9]+}}, r{{[0-9]+}}, 1
; OFF-LABEL: matmul:
; OFF: Loop Header
; OFF: isub
define agx_kernel void @matmul(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i64 %n) {
entry:
  %n32 = trunc i64 %n to i32
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %y = call i32 @llvm.agx.thread.position.in.grid.y()
  %row = mul i32 %y, %n32
  %empty = icmp slt i64 %n, 1
  br i1 %empty, label %exit, label %pre
pre:
  %cnt = call i64 @llvm.umin.i64(i64 %n, i64 1048576)
  br label %l
l:
  %ia = phi i32 [ %ia1, %l ], [ %row, %pre ]
  %ib = phi i32 [ %ib1, %l ], [ %x, %pre ]
  %c64 = phi i64 [ %c1, %l ], [ %cnt, %pre ]
  %acc = phi float [ 0.0, %pre ], [ %r, %l ]
  %za = zext i32 %ia to i64
  %pa = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %a, i64 %za
  %va = load float, ptr addrspace(1) %pa, align 4
  %zb = zext i32 %ib to i64
  %pb = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %b, i64 %zb
  %vb = load float, ptr addrspace(1) %pb, align 4
  %r = call float @llvm.fma.f32(float %va, float %vb, float %acc)
  %c1 = add nsw i64 %c64, -1
  %ib1 = add i32 %ib, %n32
  %ia1 = add i32 %ia, 1
  %done = icmp eq i64 %c1, 0
  br i1 %done, label %exit, label %l
exit:
  %s = phi float [ 0.0, %entry ], [ %r, %l ]
  %ic = add i32 %row, %x
  %zc = zext i32 %ic to i64
  %pc = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %c, i64 %zc
  store float %s, ptr addrspace(1) %pc, align 4
  ret void
}
