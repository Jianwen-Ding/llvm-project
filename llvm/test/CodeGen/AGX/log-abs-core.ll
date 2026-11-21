; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-log-abs-core=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-log-offset-split=0 < %s | FileCheck %s --check-prefix=FIELD
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-log-offset-split=0 -agx-field-mask-any-low=0 < %s | FileCheck %s --check-prefix=NOFIELD

; log's polynomial is of |x| (every result of a negative x or a NaN is
; replaced), as pow's is: log(x) + pow(x, y) computes one (L15-E613), one
; exponent and one mantissa of x.
; CHECK-LABEL: logpow:
; CHECK: bfeil {{.*}}, 24, mask 0xFF
; CHECK-NOT: bfeil {{.*}}, 24, mask 0xFF
; CHECK: device_store
; OFF-LABEL: logpow:
; OFF: bfeil {{.*}}, 24
; OFF: bfeil {{.*}}, 24
; OFF: device_store
define agx_kernel void @logpow(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %t to i64
  %pp = getelementptr float, ptr addrspace(1) %p, i64 %z
  %x = load float, ptr addrspace(1) %pp
  %l = call float @llvm.log.f32(float %x)
  %w = call float @llvm.pow.f32(float %x, float 0.75)
  %r = fadd float %l, %w
  %qq = getelementptr float, ptr addrspace(1) %q, i64 %z
  store float %r, ptr addrspace(1) %qq
  ret void
}

; A log alone: no and of the sign bit first. Its doubled bits shift the sign
; out (L15-E615); without that, the exponent of |x| is the field of x's bits
; from 23 under the sign, one bfeil.
; CHECK-LABEL: lone:
; CHECK-NOT: and
; CHECK: bfeil {{.*}}, 24, mask 0xFF
; CHECK: device_store
; FIELD-LABEL: lone:
; FIELD-NOT: and
; FIELD: bfeil {{.*}}, 23, mask 0xFF
; FIELD: device_store
; NOFIELD-LABEL: lone:
; NOFIELD: and
; NOFIELD: bfeil {{.*}}, 23
; NOFIELD: device_store
define agx_kernel void @lone(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %t to i64
  %pp = getelementptr float, ptr addrspace(1) %p, i64 %z
  %x = load float, ptr addrspace(1) %pp
  %l = call float @llvm.log.f32(float %x)
  %qq = getelementptr float, ptr addrspace(1) %q, i64 %z
  store float %l, ptr addrspace(1) %qq
  ret void
}

declare float @llvm.log.f32(float)
declare float @llvm.pow.f32(float, float)
declare i32 @llvm.agx.thread.position.in.grid.x()
