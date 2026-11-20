; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-relative-index-selects=0 < %s | FileCheck %s --check-prefix=OFF
;
; An unrolled argmax: the index chain p = c ? i + k : p (k = 0..3) kept as
; d = p - i, whose selects take the constants k, d stepping by -4 once an
; iteration and p = d + i formed after the loop (with i's exit value)
; (L15-E490).

; IR-LABEL: @argmax_rows(
; IR: %rel = phi i32 [ {{.*}}, %entry ], [ %rel.next, %loop ]
; IR: select i1 %{{.*}}, i32 0, i32 %rel
; IR: select i1 %{{.*}}, i32 1, i32
; IR: select i1 %{{.*}}, i32 2, i32
; IR: [[L:%.*]] = select i1 %{{.*}}, i32 3, i32
; IR: %rel.next = sub i32 [[L]], 4
; IR: exit:
; IR: [[OUT:%.*]] = add i32 256, %rel.next
; IR: store i32 [[OUT]]
; CHECK-LABEL: argmax_rows:
; CHECK: Loop Header
; CHECK-NOT: or
; CHECK: fcmpsel gt, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, 3,
; CHECK: isub {{r[0-9]+}}, {{r[0-9]+}}, 4
; CHECK: jmp_exec_any
; OFF-LABEL: argmax_rows:
; OFF: Loop Header
; OFF: or {{r[0-9]+}}, {{r[0-9]+}}, 3
; OFF: jmp_exec_any
define agx_kernel void @argmax_rows(ptr addrspace(1) %x, ptr addrspace(1) %o) {
entry:
  %t = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 8
  %row64 = zext i32 %row to i64
  %base = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %x, i64 %row64
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %bi = phi i32 [ 0, %entry ], [ %b3, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m3, %loop ]
  %g0 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %base, i64 %iv
  %v0 = load float, ptr addrspace(1) %g0, align 4
  %c0 = fcmp ogt float %v0, %m
  %m0 = select i1 %c0, float %v0, float %m
  %i0 = trunc nuw nsw i64 %iv to i32
  %b0 = select i1 %c0, i32 %i0, i32 %bi
  %iv1 = or disjoint i64 %iv, 1
  %g1 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %base, i64 %iv1
  %v1 = load float, ptr addrspace(1) %g1, align 4
  %c1 = fcmp ogt float %v1, %m0
  %m1 = select i1 %c1, float %v1, float %m0
  %i1 = trunc nuw nsw i64 %iv1 to i32
  %b1 = select i1 %c1, i32 %i1, i32 %b0
  %iv2 = or disjoint i64 %iv, 2
  %g2 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %base, i64 %iv2
  %v2 = load float, ptr addrspace(1) %g2, align 4
  %c2 = fcmp ogt float %v2, %m1
  %m2 = select i1 %c2, float %v2, float %m1
  %i2 = trunc nuw nsw i64 %iv2 to i32
  %b2 = select i1 %c2, i32 %i2, i32 %b1
  %iv3 = or disjoint i64 %iv, 3
  %g3 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %base, i64 %iv3
  %v3 = load float, ptr addrspace(1) %g3, align 4
  %c3 = fcmp ogt float %v3, %m2
  %m3 = select i1 %c3, float %v3, float %m2
  %i3 = trunc nuw nsw i64 %iv3 to i32
  %b3 = select i1 %c3, i32 %i3, i32 %b2
  %iv.next = add nuw nsw i64 %iv, 4
  %done = icmp eq i64 %iv.next, 256
  br i1 %done, label %exit, label %loop

exit:
  %t64 = zext i32 %t to i64
  %po = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %o, i64 %t64
  store i32 %b3, ptr addrspace(1) %po, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
