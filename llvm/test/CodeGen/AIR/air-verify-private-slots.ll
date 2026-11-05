; A pointer in private memory used with different pointee types, which Metal rejects (Unexpected bitcode
; file!, on an M2 Max; utils/air-gpu/verify-cases/ptr-slot-*). An i1 in private memory, which Metal also
; rejects as it comes, is legalized to i8 with its accesses, and stays volatile.
; RUN: split-file %s %t
; RUN: llc -mtriple=air %t/i1.ll -o - | FileCheck %s --check-prefix=I1
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/two.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PTR
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/gep.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PTR
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/one.ll -o /dev/null
; I1: alloca i8
; I1-NOT: alloca i1
; I1: store volatile i8
; I1: load volatile i8
; PTR: a pointer in private memory used with different pointee types
;--- i1.ll
target triple = "air"
define void @k(ptr addrspace(1) %o, i32 %n) "air-kernel" {
  %s = alloca i1, align 1
  %c = icmp eq i32 %n, 0
  store volatile i1 %c, ptr %s, align 1
  %l = load volatile i1, ptr %s, align 1
  %z = zext i1 %l to i32
  store i32 %z, ptr addrspace(1) %o
  ret void
}
;--- two.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %s = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %o, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  store i32 7, ptr addrspace(1) %p, align 4
  %q = load ptr addrspace(1), ptr %s, align 8
  store i8 1, ptr addrspace(1) %q, align 1
  ret void
}
;--- gep.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %s = alloca ptr addrspace(1), align 8
  %g = getelementptr inbounds i8, ptr addrspace(1) %o, i64 4
  store ptr addrspace(1) %g, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  store i32 7, ptr addrspace(1) %p, align 4
  ret void
}
;--- one.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %s = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %o, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  store i32 7, ptr addrspace(1) %p, align 4
  ret void
}
