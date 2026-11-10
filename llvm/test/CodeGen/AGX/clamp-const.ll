; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; maximumnum / minimumnum with a nonzero constant need no fix for zeros of
; two signs (equal operands then have the same bits): a clamp to [-128, 127]
; is one compare-select per bound.

; CHECK-LABEL: s8:
; CHECK: fcmpsel gtn
; CHECK-NOT: fcmpsel eq
; CHECK: fcmpsel ltn
; CHECK-NOT: fcmpsel eq
; CHECK: device_store
define agx_kernel void @s8(ptr addrspace(1) %p, float %a) {
  %l = call float @llvm.maximumnum.f32(float %a, float -128.0)
  %c = call float @llvm.minimumnum.f32(float %l, float 127.0)
  store float %c, ptr addrspace(1) %p
  ret void
}

; maximumnum(x, +0) is (x > 0) ? x : +0 (a NaN, -0 and +0 give +0): one
; fcmpsel with immediates.
; CHECK-LABEL: z:
; CHECK: fcmpsel gt, [[R:r[0-9]+]], [[X:u[0-9]+]], 0.0, [[X]], 0
; CHECK-NEXT: device_store
define agx_kernel void @z(ptr addrspace(1) %p, float %a) {
  %m = call float @llvm.maximumnum.f32(float %a, float 0.0)
  store float %m, ptr addrspace(1) %p
  ret void
}

; A zero bound of minimumnum keeps the fix: minimumnum(-0, +0) is -0.
; CHECK-LABEL: zmin:
; CHECK: fcmpsel ltn
; CHECK: fcmpsel eq
; CHECK: stop
define agx_kernel void @zmin(ptr addrspace(1) %p, float %a) {
  %m = call float @llvm.minimumnum.f32(float %a, float 0.0)
  store float %m, ptr addrspace(1) %p
  ret void
}
