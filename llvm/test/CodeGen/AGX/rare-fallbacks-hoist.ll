; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rare-fallbacks-hoist=0 < %s | FileCheck %s --check-prefix=OFF

; L15-E505: what follows a branched-around fallback's selects (L15-E504) and
; needs neither them nor a write before it goes ahead of the branch: the
; rotation's loads of x start before the hardware's sin and cos are tested
; for. The load after the store of y0 (another address only at run time)
; stays after it.

; CHECK-LABEL: k:
; CHECK: device_load
; CHECK: device_load
; CHECK: device_load
; CHECK: if_fcmp
; CHECK-NEXT: jmp_exec_none
; CHECK: pop_exec
; CHECK-NOT: device_load
; CHECK: device_store
; CHECK: device_load
; CHECK: device_store

; OFF-LABEL: k:
; OFF: device_load
; OFF-NOT: device_load
; OFF: if_fcmp
; OFF: pop_exec
; OFF: device_load
; OFF: device_load

declare i32 @llvm.agx.thread.position.in.grid.x()
declare { float, float } @llvm.sincos.f32(float)
define agx_kernel void @k(ptr addrspace(1) %ang, ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %z) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr float, ptr addrspace(1) %ang, i32 %i
  %a = load float, ptr addrspace(1) %pa
  %sc = call { float, float } @llvm.sincos.f32(float %a)
  %s = extractvalue { float, float } %sc, 0
  %c = extractvalue { float, float } %sc, 1
  %j = shl i32 %i, 1
  %j1 = or i32 %j, 1
  %p0 = getelementptr float, ptr addrspace(1) %x, i32 %j
  %p1 = getelementptr float, ptr addrspace(1) %x, i32 %j1
  %x0 = load float, ptr addrspace(1) %p0
  %x1 = load float, ptr addrspace(1) %p1
  %m0 = fmul float %x0, %c
  %m1 = fmul float %x1, %s
  %y0 = fsub float %m0, %m1
  %q0 = getelementptr float, ptr addrspace(1) %y, i32 %j
  store float %y0, ptr addrspace(1) %q0
  ; a load after a store to the same buffer (y[j + z], z = 0 at run time):
  ; it stays after the store
  %q1 = getelementptr float, ptr addrspace(1) %y, i32 %j1
  %jz = add i32 %j, %z
  %qz = getelementptr float, ptr addrspace(1) %y, i32 %jz
  %back = load float, ptr addrspace(1) %qz
  %u = fmul float %x0, %s
  %v = fmul float %x1, %c
  %y1 = fadd float %u, %v
  %y1b = fadd float %y1, %back
  store float %y1b, ptr addrspace(1) %q1
  ret void
}
