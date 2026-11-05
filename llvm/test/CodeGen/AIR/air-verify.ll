; llc reports what the device rejects, and nothing else. The modules are those of
; llvm/utils/air-gpu/verify-cases: air_verify_device_check.py shows that the device
; rejects the first group and runs the second, which the documents would exclude.
; RUN: split-file %s %t
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/agxcall.ll -o /dev/null 2>&1 | FileCheck %t/agxcall.ll
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/bare.ll -o /dev/null 2>&1 | FileCheck %t/bare.ll
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/duploc.ll -o /dev/null 2>&1 | FileCheck %t/duploc.ll
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/noarg.ll -o /dev/null 2>&1 | FileCheck %t/noarg.ll
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/undescribed.ll -o /dev/null 2>&1 | FileCheck %t/undescribed.ll
; RUN: not llc -mtriple=air -filetype=obj -air-output=air %t/unresolved.ll -o /dev/null 2>&1 | FileCheck %t/unresolved.ll
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/as5.ll -o %t/as5.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/as5.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/ascast.ll -o %t/ascast.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/ascast.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/bufas.ll -o %t/bufas.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/bufas.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/dbl.ll -o %t/dbl.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/dbl.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/i1mem.ll -o %t/i1mem.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/i1mem.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/int128.ll -o %t/int128.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/int128.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/int3.ll -o %t/int3.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/int3.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air -air-verify-written %t/ok.ll -o %t/ok.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/ok.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/retval.ll -o %t/retval.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/retval.air
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t/tgbig.ll -o %t/tgbig.air
; RUN: opt -mtriple=air -passes=air-verify-metal -disable-output %t/tgbig.air

;--- agxcall.ll
; CHECK: the device would reject this library: {{.*}}which is AGX's
target triple = "air"
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  store i32 %v, ptr addrspace(1) %o
  ret void
}

;--- bare.ll
; CHECK: the device would reject this library: {{.*}}does not describe the kernel's arguments
target triple = "air"
define void @k(ptr addrspace(1) %o) {
  store i32 7, ptr addrspace(1) %o
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k}

;--- duploc.ll
; CHECK: the device would reject this library: {{.*}}two buffer arguments at location 0
target triple = "air"
define void @k(ptr addrspace(1) %o, ptr addrspace(1) %p) {
  store i32 7, ptr addrspace(1) %o
  store i32 8, ptr addrspace(1) %p
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"p"}

;--- noarg.ll
; CHECK: the device would reject this library: {{.*}}which the kernel does not have
target triple = "air"
define void @k() {
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}

;--- undescribed.ll
; CHECK: the device would reject this library: {{.*}}does not describe argument 1
target triple = "air"
define void @k(ptr addrspace(1) %o, i32 %x) {
  store i32 %x, ptr addrspace(1) %o
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}

;--- unresolved.ll
; CHECK: the device would reject this library: {{.*}}which neither the module nor AIR defines
target triple = "air"
declare i32 @nothing_defines_this(i32)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = call i32 @nothing_defines_this(i32 1)
  store i32 %v, ptr addrspace(1) %o
  ret void
}

;--- as5.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %a = inttoptr i64 64 to ptr addrspace(5)
  %v = load i32, ptr addrspace(5) %a
  store i32 %v, ptr addrspace(1) %o
  ret void
}

;--- ascast.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %c = addrspacecast ptr addrspace(1) %o to ptr addrspace(2)
  %v = load i32, ptr addrspace(2) %c
  store i32 %v, ptr addrspace(1) %o
  ret void
}

;--- bufas.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) {
  store i32 7, ptr addrspace(1) %o
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}

;--- dbl.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load float, ptr addrspace(1) %o
  %d = fpext float %v to double
  %a = fadd double %d, 1.0
  %t = fptrunc double %a to float
  store float %t, ptr addrspace(1) %o
  ret void
}

;--- i1mem.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i1 true, ptr addrspace(1) %o
  ret void
}

;--- int128.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load i64, ptr addrspace(1) %o
  %w = zext i64 %v to i128
  %m = mul i128 %w, %w
  %s = lshr i128 %m, 64
  %t = trunc i128 %s to i64
  store i64 %t, ptr addrspace(1) %o
  ret void
}

;--- int3.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load i32, ptr addrspace(1) %o
  %t = trunc i32 %v to i3
  %a = add i3 %t, 1
  %z = zext i3 %a to i32
  store i32 %z, ptr addrspace(1) %o
  ret void
}

;--- ok.ll
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i32 7, ptr addrspace(1) %o
  ret void
}

;--- retval.ll
target triple = "air"
define i32 @k(ptr addrspace(1) %o) {
  store i32 7, ptr addrspace(1) %o
  ret i32 1
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}

;--- tgbig.ll
target triple = "air"
@t = internal addrspace(3) global [65536 x i8] undef, align 16
define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i8 1, ptr addrspace(3) @t
  %v = load i8, ptr addrspace(3) @t
  store i8 %v, ptr addrspace(1) %o
  ret void
}
