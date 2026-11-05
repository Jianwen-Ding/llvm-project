; Instruction metadata in AIR for Metal: a kind the file does not have is dropped:
; agx.atomic.ordering, the hint of llc -air-portable for AGX, made the file
; unreadable ("Invalid ID"). A kind it has keeps its name.
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o %t.portable.ll
; RUN: FileCheck %s --check-prefix=PORTABLE < %t.portable.ll
; RUN: llc -mtriple=air -filetype=obj -air-output=air %t.portable.ll -o %t.portable.air
; RUN: llvm-dis %t.portable.air -o - | FileCheck %s --check-prefix=METAL

; PORTABLE: !agx.atomic.ordering
; METAL-LABEL: define {{.*}}@ordered(
; METAL: call i32 @air.atomic.global.cmpxchg.weak
; METAL-NOT: agx.atomic.ordering

; CHECK-LABEL: define {{.*}}@kinds(
; CHECK: fadd float {{.*}}!fpmath
; CHECK-NOT: agx.atomic.ordering

target triple = "air"

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)

define void @ordered(ptr addrspace(1) %p) "air-kernel" {
  %r = cmpxchg ptr addrspace(1) %p, i32 0, i32 1 acq_rel monotonic
  ret void
}

define void @kinds(ptr addrspace(1) %p) "air-kernel" {
  %x = load float, ptr addrspace(1) %p
  %y = fadd float %x, %x, !fpmath !0, !agx.atomic.ordering !1
  store float %y, ptr addrspace(1) %p
  ret void
}

!0 = !{float 2.5}
!1 = !{i32 4, i32 0}
