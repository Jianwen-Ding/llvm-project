; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-narrow-bounded-scaled-indices=0 < %s | FileCheck %s --check-prefix=OFF

; zext(i) << 3 as a float array's index, i below 2^17 by its upper_bound: the
; shift is 32-bit, under the access's own scale (lsl 2), not a 64-bit shift's
; halves (bfeil, bfi) and a 64-bit add.

; CHECK-LABEL: vec8:
; CHECK-NOT: bfeil
; CHECK: iadd [[I:r[0-9]+]], 0, {{r[0-9]+}}, lsl 3
; CHECK: device_load {{.*}}, [[I]], unsigned
; OFF-LABEL: vec8:
; OFF: bfeil
define agx_kernel void @vec8(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %i to i64
  %o = shl nuw nsw i64 %z, 3
  %a = getelementptr [4 x i8], ptr addrspace(1) %p, i64 %o
  %v = load <4 x float>, ptr addrspace(1) %a, align 4
  %b = getelementptr [4 x i8], ptr addrspace(1) %q, i64 %o
  store <4 x float> %v, ptr addrspace(1) %b, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()

attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="131072,0,0" }
