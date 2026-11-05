; AIR as bitcode (a .air file) through llc for AGX: an alloca passed to a call
; stays in AIR's thread address space while the module is read, whatever AGX's
; layout says, and compiles as the same module in text does.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: llvm-as %t.air.ll -o %t.air
; RUN: llc -mtriple=agx -mcpu=agx-g14g %t.air -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g -filetype=obj %t.air -o %t.bc.o
; RUN: llc -mtriple=agx -mcpu=agx-g14g -filetype=obj %t.air.ll -o %t.text.o
; RUN: cmp %t.bc.o %t.text.o

target triple = "air"

; CHECK-LABEL: k:
define void @k(ptr addrspace(1) %p) {
  %expected = alloca i32, align 4
  store i32 0, ptr %expected
  %ok = call i1 @cas(ptr addrspace(1) %p, ptr %expected)
  ret void
}

define internal i1 @cas(ptr addrspace(1) %p, ptr %expected) noinline {
  %e = load i32, ptr %expected
  %r = cmpxchg ptr addrspace(1) %p, i32 %e, i32 1 monotonic monotonic
  %ok = extractvalue { i32, i1 } %r, 1
  ret i1 %ok
}
