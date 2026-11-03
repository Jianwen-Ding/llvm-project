; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; The hardware's f32 compares flush denormals, so a min / max that only compares
; or other min / max read needs no flush of its own (fmul by 1.0); one that is
; stored does. nsz leaves out the ordering of zeros (the or and second fcmpsel).

target triple = "agx-unknown-unknown"
declare float @llvm.minimumnum.f32(float, float)
declare float @llvm.maximumnum.f32(float, float)

; CHECK-LABEL: slab:
; CHECK-NOT: fmul32
; CHECK-NOT: or{{[[:space:]]}}
; CHECK: stop
define agx_kernel void @slab(float %a, float %b, float %c, ptr addrspace(1) %p) #0 {
  %min = call nsz float @llvm.minimumnum.f32(float %a, float %b)
  %max = call nsz float @llvm.maximumnum.f32(float %min, float %c)
  %cmp = fcmp olt float %max, %c
  %r = select i1 %cmp, i32 1, i32 2
  store volatile i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: stored:
; CHECK: fmul32
; CHECK: stop
define agx_kernel void @stored(float %a, float %b, ptr addrspace(1) %p) #0 {
  %min = call nsz float @llvm.minimumnum.f32(float %a, float %b)
  store volatile float %min, ptr addrspace(1) %p
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) }
