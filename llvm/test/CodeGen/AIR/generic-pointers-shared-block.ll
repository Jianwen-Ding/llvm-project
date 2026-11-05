; A kernel whose threadgroup block is named by "air-shared-block" holds offsets in it in
; generic addresses: the block's address comes off going to generic and back on coming out.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
target triple = "air"
@b = internal addrspace(3) global [16 x i8] undef, align 16

; CHECK-LABEL: define void @k(
; CHECK: sub i64
; CHECK: or i64 %{{.*}}, -9223372036854775808
; CHECK: and i64 %{{.*}}, 9223372036854775807
; CHECK: add i64 %{{.*}}, %{{.*}}
define void @k(ptr addrspace(1) %o, i64 %i) "air-kernel" "air-shared-block"="b" {
  %q = getelementptr i8, ptr addrspace(3) @b, i64 %i
  %g = addrspacecast ptr addrspace(3) %q to ptr addrspace(4)
  %v = ptrtoint ptr addrspace(4) %g to i64
  store i64 %v, ptr addrspace(1) %o
  %p = inttoptr i64 %v to ptr addrspace(4)
  %s = addrspacecast ptr addrspace(4) %p to ptr addrspace(3)
  store i32 7, ptr addrspace(3) %s
  ret void
}
