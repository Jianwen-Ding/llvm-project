; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rare-fallbacks-unlikely=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=COLD
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rare-fallbacks-unlikely=0 < %s | FileCheck %s --check-prefix=SINS
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rare-fallbacks-unlikely=0 -agx-rare-hardware-branch=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rare-fallbacks-unlikely=0 -agx-rare-fallbacks-min=1 < %s | FileCheck %s --check-prefix=ONE
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rare-fallbacks-unlikely=0 -agx-rare-fallbacks-merge=0 < %s | FileCheck %s --check-prefix=APART

; sin and cos past 2^22 take the hardware's reduction (L15-E504): sincos's
; two and tan's run in a block of their own, skipped by a SIMD-group none of
; whose threads has such an argument. Without it, every thread computed them
; and a select chose. One alone (8 instructions) would run with an empty
; mask anyway: left as the select.

; CHECK-LABEL: sin_f32:
; CHECK-NOT: if_fcmp
; CHECK: sin_pt_1
; CHECK-NOT: if_fcmp
; CHECK: fcmpsel lt
; CHECK: device_store

; By default the branch to the rare block is weighted unlikely and
; AGXColdRegions moves the block after the kernel's end: the common path
; falls through a jmp_exec_any not taken (L15-E862).
; COLD-LABEL: sincos_f32:
; COLD: if_fcmp {{r[0-9]+l}}, nlt, {{r[0-9]+}}.abs, {{r[0-9]+}}, 1
; COLD-NEXT: jmp_exec_any [[R:.LBB[0-9_]+]]
; COLD: pop_exec
; COLD: stop
; COLD: [[R]]:
; COLD-COUNT-2: sin_pt_1
; COLD: jmp_exec_any
; CHECK-LABEL: sincos_f32:
; CHECK: if_fcmp [[E:r[0-9]+l]], nlt, {{r[0-9]+}}.abs, {{r[0-9]+}}, 1
; CHECK-NEXT: jmp_exec_none [[J:.LBB[0-9_]+]]
; CHECK-COUNT-2: sin_pt_1
; CHECK: [[J]]:
; CHECK-NEXT: pop_exec [[E]], 1
; CHECK-NOT: sin_pt_1
; CHECK: device_store

; CHECK-LABEL: tan_f32:
; CHECK: if_fcmp {{r[0-9]+l}}, nlt, {{r[0-9]+}}.abs, {{r[0-9]+}}, 1
; CHECK-NEXT: jmp_exec_none
; CHECK-COUNT-2: sin_pt_1
; CHECK: pop_exec
; CHECK: rcp
; CHECK: device_store

; OFF-LABEL: sincos_f32:
; OFF-NOT: jmp_exec_none
; OFF: sin_pt_1
; OFF-NOT: jmp_exec_none
; OFF: fcmpsel lt
; OFF: device_store

; CHECK-LABEL: two_sins:
; CHECK-NOT: if_
; CHECK: stop
; ONE-LABEL: two_sins:
; ONE: if_fcmp
; ONE: sin_pt_1
; ONE: pop_exec
; ONE: if_
; ONE: sin_pt_1
; ONE: pop_exec
; ONE: stop

; Four sins of four arguments in one block (an unrolled loop's): four
; conditions, one fallback each, too few apart; merged, one branch to a block
; of all four hardware sins where any argument needs one, each value there
; the select of its own condition (L15-E789).
; CHECK-LABEL: four_sins:
; CHECK-COUNT-4: fcmpsel lt, {{r[0-9]+}}, {{r[0-9]+}}.abs
; CHECK: if_icmp
; CHECK: jmp_exec_none [[J:.LBB[0-9_]+]]
; (the rare block's compares read abs(x) as a modifier, no and of the bits)
; CHECK-NOT: {{^}}	and
; CHECK: fcmpsel lt, {{r[0-9]+}}, {{r[0-9]+}}.abs
; CHECK-NOT: {{^}}	and
; CHECK: [[J]]:
; CHECK-NEXT: pop_exec
; CHECK-NOT: sin_pt_1
; CHECK: device_store
; SINS-LABEL: four_sins:
; SINS: jmp_exec_none
; SINS-COUNT-4: sin_pt_1
; SINS: pop_exec
; SINS-NOT: sin_pt_1
; SINS: device_store
; APART-LABEL: four_sins:
; APART-NOT: if_
; APART: device_store

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.sin.f32(float)
declare { float, float } @llvm.sincos.f32(float)
declare float @llvm.tan.f32(float)

define agx_kernel void @sin_f32(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %r = call float @llvm.sin.f32(float %x)
  store float %r, ptr addrspace(1) %a
  ret void
}

define agx_kernel void @sincos_f32(ptr addrspace(1) %p, ptr addrspace(1) %q) {
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

define agx_kernel void @tan_f32(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %r = call float @llvm.tan.f32(float %x)
  store float %r, ptr addrspace(1) %a
  ret void
}

; Two sins of two arguments in one block: two conditions, a group each. With
; one fallback a group enough, the first group's split of the block moves the
; second's selects to its tail (after the first store, the second load and
; sin stay behind the first's selects), where the second branches from (that
; split once asserted on the stale block).
define agx_kernel void @two_sins(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %r = call float @llvm.sin.f32(float %x)
  store float %r, ptr addrspace(1) %a
  %b = getelementptr float, ptr addrspace(1) %a, i32 1024
  %y = load float, ptr addrspace(1) %b
  %s = call float @llvm.sin.f32(float %y)
  store float %s, ptr addrspace(1) %b
  ret void
}

define agx_kernel void @four_sins(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr <4 x float>, ptr addrspace(1) %p, i32 %i
  %x = load <4 x float>, ptr addrspace(1) %a
  %r = call <4 x float> @llvm.sin.v4f32(<4 x float> %x)
  store <4 x float> %r, ptr addrspace(1) %a
  ret void
}

declare <4 x float> @llvm.sin.v4f32(<4 x float>)
