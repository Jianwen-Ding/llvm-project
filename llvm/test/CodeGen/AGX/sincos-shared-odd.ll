; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sincos-shared-odd=0 < %s | FileCheck %s --check-prefix=OFF

; sincos (L15-E502): cos is sin a quadrant ahead, and q + 1 is odd where q is
; not, so cos's polynomial select uses sin's test of q with its operands
; swapped: one and of q's low bit, where (q + 1) & 1 was a second.

; CHECK-LABEL: sincos:
; CHECK: and {{r[0-9]+}}, [[Q:r[0-9]+]], 1
; CHECK-NOT: and {{r[0-9]+}}, {{r[0-9]+}}, 1{{$}}
; CHECK: device_store

; OFF-LABEL: sincos:
; OFF: and {{r[0-9]+}}, {{r[0-9]+}}, 1{{$}}
; OFF: and {{r[0-9]+}}, {{r[0-9]+}}, 1{{$}}
; OFF: device_store

declare i32 @llvm.agx.thread.position.in.grid.x()
declare { float, float } @llvm.sincos.f32(float)

define agx_kernel void @sincos(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %sc = call { float, float } @llvm.sincos.f32(float %x)
  %s = extractvalue { float, float } %sc, 0
  %c = extractvalue { float, float } %sc, 1
  store float %s, ptr addrspace(1) %a
  %b = getelementptr float, ptr addrspace(1) %q, i32 %i
  store float %c, ptr addrspace(1) %b
  ret void
}
