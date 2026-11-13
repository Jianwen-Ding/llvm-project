; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-threadgroup-ranges=0 < %s | FileCheck %s --check-prefix=OFF

; A threadgroup of at most N threads ("agx-max-threads-per-threadgroup", a
; promise of the launch) has its positions below N and its SIMD-group index
; below N / 32: the masks below are then known to keep every bit. Without the
; attribute (or the option) the bounds are the hardware's (1024 threads, 32
; SIMD-groups), and the masks stay.

declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare i32 @llvm.agx.simdgroup.index.in.threadgroup()

; CHECK-LABEL: position:
; CHECK-NOT: and
; CHECK: stop
; OFF-LABEL: position:
; OFF: and
define agx_kernel void @position(ptr addrspace(1) %o) #0 {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %m = and i32 %t, 255
  store i32 %m, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: simdgroup:
; CHECK-NOT: and
; CHECK: stop
; OFF-LABEL: simdgroup:
; OFF: and
define agx_kernel void @simdgroup(ptr addrspace(1) %o) #1 {
  %s = call i32 @llvm.agx.simdgroup.index.in.threadgroup()
  %m = and i32 %s, 3
  store i32 %m, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: unknown:
; CHECK: and
; CHECK: stop
define agx_kernel void @unknown(ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %m = and i32 %t, 255
  store i32 %m, ptr addrspace(1) %o, align 4
  ret void
}

attributes #0 = { "agx-max-threads-per-threadgroup"="256" }
attributes #1 = { "agx-max-threads-per-threadgroup"="128" }
