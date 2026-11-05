; What the device rejects is an error here, not a library that fails on the device
; ("Unexpected bitcode file!"). Each rule was found by reducing a library the device
; rejected; -air-skip-device-checks writes the library anyway, to show that again.
; RUN: split-file %s %t
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/gep.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GEP
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/cmp.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CMP
; RUN: llc -mtriple=air -filetype=obj -air-output=air -air-skip-device-checks %t/gep.ll -o /dev/null
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/ok.ll -o /dev/null

; GEP: the device would reject this library: {{.*}}a getelementptr of a type other than its base's pointee
; CMP: the device would reject this library: {{.*}}a comparison of pointers of different types

;--- gep.ll
target triple = "air"
@bar = internal addrspace(3) global i64 undef, align 8
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %p = getelementptr inbounds i8, ptr addrspace(3) @bar, i64 4
  %v = load i32, ptr addrspace(3) %p
  store i32 %v, ptr addrspace(1) %o
  ret void
}

;--- cmp.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %a = alloca [2 x i32], align 4
  %c = icmp slt ptr %a, null
  %z = zext i1 %c to i32
  store i32 %z, ptr addrspace(1) %o
  ret void
}

;--- ok.ll
target triple = "air"
@w = internal addrspace(3) global [4 x i32] undef, align 4
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %p = getelementptr inbounds [4 x i32], ptr addrspace(3) @w, i64 0, i64 1
  %v = load i32, ptr addrspace(3) %p
  store i32 %v, ptr addrspace(1) %o
  ret void
}
