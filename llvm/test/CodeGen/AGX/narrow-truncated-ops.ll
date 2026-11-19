; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; A 64-bit operation in another block than its users, which all truncate it
; to 32 bits (as E427's narrowed compares leave them), is the 32-bit operation
; of its operands' low words (L15-E429). @shifted's lshr needs the high word:
; it stays 64-bit.

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: @offsets(
; CHECK:       [[A:%.*]] = add i32 %x, -1
; CHECK:       [[B:%.*]] = or i32 [[A]], %y
; CHECK:       icmp ult i32 [[B]], 1024
; CHECK-NOT:   i64
; CHECK:       ret void
define agx_kernel void @offsets(ptr addrspace(1) %o, i32 %y, i1 %c) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %xz = zext i32 %x to i64
  %yz = zext i32 %y to i64
  %a = add nsw i64 %xz, -1
  %b = or i64 %a, %yz
  br i1 %c, label %t, label %e
t:
  %bt = trunc i64 %b to i32
  %in = icmp ult i32 %bt, 1024
  %v = zext i1 %in to i32
  store i32 %v, ptr addrspace(1) %o, align 4
  br label %e
e:
  ret void
}

; CHECK-LABEL: @shifted(
; CHECK:       lshr i64
; CHECK:       ret void
define agx_kernel void @shifted(ptr addrspace(1) %o, i64 %p, i1 %c) {
entry:
  %s = lshr i64 %p, 8
  br i1 %c, label %t, label %e
t:
  %st = trunc i64 %s to i32
  store i32 %st, ptr addrspace(1) %o, align 4
  br label %e
e:
  ret void
}
