; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-inline-single-call-max-instrs=2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=SMALL
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-inline-single-call-max-depth=1 -agx-inline-single-call-report -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=DEPTH
; On the call path, a function with a single call site is inlined into its
; caller; one called from two sites stays a call.

; CHECK-LABEL: kernel:
; CHECK-NOT: call once
; CHECK-NOT: call inner
; CHECK: call twice
; CHECK: call twice
; CHECK-NOT: {{^}}once:
; CHECK-NOT: {{^}}inner:
; CHECK-LABEL: twice:

; Larger than the bound: kept.
; SMALL-LABEL: kernel:
; SMALL: call once
; SMALL: call twice
; SMALL-LABEL: once:

; inner (into once) and once (into kernel) would make two levels.
; DEPTH-DAG: inline single call: inner ({{[0-9]+}}) into once ({{[0-9]+}}, depth 1)
; DEPTH-DAG: inline single call: once ({{[0-9]+}}) into kernel ({{[0-9]+}}, depth 2) skipped

target triple = "agx-unknown-unknown"

define agx_kernel void @kernel(i32 %n) {
entry:
  %a = call i32 @once(i32 %n)
  %b = call i32 @twice(i32 %a)
  %c = call i32 @twice(i32 %b)
  store volatile i32 %c, ptr addrspace(1) inttoptr (i64 8192 to ptr addrspace(1))
  ret void
}

define internal i32 @once(i32 %x) {
entry:
  %y = mul i32 %x, %x
  %z = call i32 @inner(i32 %y)
  %w = add i32 %z, 7
  ret i32 %w
}

define internal i32 @inner(i32 %x) {
entry:
  %y = xor i32 %x, 12345
  %z = mul i32 %y, 3
  ret i32 %z
}

define internal i32 @twice(i32 %x) {
entry:
  %y = mul i32 %x, 5
  %z = add i32 %y, 11
  ret i32 %z
}
