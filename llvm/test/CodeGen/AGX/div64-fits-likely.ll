; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s --check-prefix=ASM

; A front end's test that a 64-bit division's operands fit in 32 bits, with
; the 32-bit division on one side and the 64-bit one on the other, is marked
; likely to fit (the outermost such test on the division's path: the
; bypass's own inner test is left alone), and AGXColdRegions moves the long
; case out of line.

; CHECK-LABEL: @div(
; CHECK: br i1 %fits, label %fast, label %slow, !prof [[P:![0-9]+]]
; CHECK: [[P]] = !{!"branch_weights", i32 1048575, i32 1}
; ASM-LABEL: div:
; ASM: jmp_exec_any [[COLD:\.LBB[0-9_]+]]
; ASM: stop
; ASM: [[COLD]]:
define agx_kernel void @div(ptr addrspace(1) %p, i64 %a, i64 %b) {
entry:
  %or = or i64 %a, %b
  %fits = icmp ult i64 %or, 4294967296
  br i1 %fits, label %fast, label %slow
fast:
  %a32 = trunc i64 %a to i32
  %b32 = trunc i64 %b to i32
  %q32 = udiv i32 %a32, %b32
  %q1 = zext i32 %q32 to i64
  br label %join
slow:
  %q2 = sdiv i64 %a, %b
  br label %join
join:
  %q = phi i64 [ %q1, %fast ], [ %q2, %slow ]
  store i64 %q, ptr addrspace(1) %p
  ret void
}
