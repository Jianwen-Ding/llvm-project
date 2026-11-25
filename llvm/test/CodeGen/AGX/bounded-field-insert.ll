; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-bounded-field-insert=0 < %s | FileCheck %s --check-prefix=OFF

; i * 1024 + j with j's upper_bound 1024 (MLIR's, as the kernel's
; agx-upper-bounds attribute): one bfi of i into j, not a bfi and an add.

; CHECK-LABEL: rows:
; CHECK: get_sr [[I:r[0-9]+]], sr81
; CHECK: get_sr [[J:r[0-9]+]], sr80
; CHECK: bfi [[K:r[0-9]+]], [[J]], [[I]], 10, mask 0x3FFFFF
; CHECK-NOT: iadd
; CHECK: device_store {{.*}}, [[K]], unsigned
; OFF-LABEL: rows:
; OFF: bfi {{r[0-9]+}}, 0, {{r[0-9]+}}, 10
; OFF: iadd
define agx_kernel void @rows(ptr addrspace(1) %p) #0 {
  %i = call i32 @llvm.agx.thread.position.in.grid.y()
  %j = call i32 @llvm.agx.thread.position.in.grid.x()
  %s = shl i32 %i, 10
  %k = add i32 %s, %j
  %z = zext i32 %k to i64
  %g = getelementptr float, ptr addrspace(1) %p, i64 %z
  store float 1.0, ptr addrspace(1) %g
  ret void
}

; j's bound 2048 is above 2^10: the add stays.
; CHECK-LABEL: wide:
; CHECK: iadd
define agx_kernel void @wide(ptr addrspace(1) %p) #1 {
  %i = call i32 @llvm.agx.thread.position.in.grid.y()
  %j = call i32 @llvm.agx.thread.position.in.grid.x()
  %s = shl i32 %i, 10
  %k = add i32 %s, %j
  %z = zext i32 %k to i64
  %g = getelementptr float, ptr addrspace(1) %p, i64 %z
  store float 1.0, ptr addrspace(1) %g
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.grid.y()

attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="1024,1024,0" }
attributes #1 = { "agx-upper-bounds:air.thread_position_in_grid"="2048,1024,0" }
