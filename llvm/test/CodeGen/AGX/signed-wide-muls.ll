; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-signed-wide-muls=0 < %s | FileCheck %s --check-prefix=OFF

; A sign-extended i32 times a zero-extended non-negative one (zext nneg, the
; PTX front end's int dimension) is one signed 32 x 32 -> 64 imadd, also when
; the zext is in another block (L15-E414).

; CHECK-LABEL: same_block:
; CHECK: imadd {{r[0-9]+_r[0-9]+}}, {{r[0-9]+}}.sx, u3.sx, 0
; CHECK-NOT: asr
; OFF-LABEL: same_block:
; OFF: imadd
; OFF: asr
; OFF: imadd
; CHECK-LABEL: other_block:
; CHECK: imadd {{r[0-9]+_r[0-9]+}}, {{r[0-9]+}}.sx, u3.sx, 0
; CHECK-NOT: asr
; CHECK: jmp_exec_any
; OFF-LABEL: other_block:
; OFF: asr

declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @same_block(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %at = add i32 %a, %t
  %x = sext i32 %at to i64
  %y = zext nneg i32 %b to i64
  %m = mul i64 %x, %y
  %q = getelementptr i64, ptr addrspace(1) %p, i32 %t
  store i64 %m, ptr addrspace(1) %q, align 8
  ret void
}
define agx_kernel void @other_block(ptr addrspace(1) %p, i32 %a, i32 %b, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %y = zext nneg i32 %b to i64
  br label %l
l:
  %i = phi i32 [ %t, %entry ], [ %i1, %l ]
  %x = sext i32 %i to i64
  %m = mul i64 %x, %y
  %q = getelementptr float, ptr addrspace(1) %p, i64 %m
  store float 1.0, ptr addrspace(1) %q, align 4
  %i1 = add i32 %i, 64
  %c = icmp slt i32 %i1, %n
  br i1 %c, label %l, label %e
e:
  ret void
}
