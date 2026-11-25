; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-merge-phi-operations=0 < %s | FileCheck %s --check-prefix=OFF

; x + 1 on both sides of a branch into a phi (jacobi2d's): one x + 1 at the
; join, with the flags both had; the side that still uses its own keeps it
; (L15-E421).

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: @both(
; CHECK: j:
; CHECK-NEXT: %r{{[0-9]*}} = add i32 %x, 1
; CHECK-NOT: phi
; OFF-LABEL: @both(
; OFF: %r = phi i32
define agx_kernel void @both(ptr addrspace(1) %p, i32 %n) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %x, %n
  br i1 %c, label %a, label %b
a:
  %s1 = add nsw i32 %x, 1
  %u = icmp slt i32 %s1, %n
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %x
  %z = zext i1 %u to i32
  store i32 %z, ptr addrspace(1) %q, align 4
  br label %j
b:
  %s2 = add i32 %x, 1
  br label %j
j:
  %r = phi i32 [ %s1, %a ], [ %s2, %b ]
  %o = getelementptr i32, ptr addrspace(1) %p, i32 %r
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}

; Different operands stay a phi.
; CHECK-LABEL: @different(
; CHECK: phi i32
define agx_kernel void @different(ptr addrspace(1) %p, i32 %n) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %x, %n
  br i1 %c, label %a, label %b
a:
  %s1 = add i32 %x, 1
  br label %j
b:
  %s2 = add i32 %x, 2
  br label %j
j:
  %r = phi i32 [ %s1, %a ], [ %s2, %b ]
  %o = getelementptr i32, ptr addrspace(1) %p, i32 %r
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}
