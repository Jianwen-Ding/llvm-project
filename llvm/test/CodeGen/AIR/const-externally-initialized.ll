; Metal reads a constant-space array marked externally_initialized wrongly: the Metal pipeline
; clears the mark, and the portable AIR, for other consumers, keeps it.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE
; CHECK: @g = addrspace(2) global [4 x i16]
; CHECK-NOT: externally_initialized
; PORTABLE: @g = addrspace(2) externally_initialized global [4 x i16]
target triple = "air"
@g = addrspace(2) externally_initialized global [4 x i16] [i16 10, i16 20, i16 30, i16 40], align 8
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %p = getelementptr inbounds nuw i8, ptr addrspace(2) @g, i64 2
  %v = load i16, ptr addrspace(2) %p
  store i16 %v, ptr addrspace(1) %o
  ret void
}
