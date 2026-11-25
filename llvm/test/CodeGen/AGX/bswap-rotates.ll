; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-bswap-rotates=0 < %s | FileCheck %s --check-prefix=OFF

; A 32-bit byte swap (L15-E556): [b3 b2 b1 b0] rotated by 8 either way has
; two of its bytes in place, masked and or'ed: two rotates, the shared
; unit's, where the expansion took three (two rotates and a shift). A 64-bit
; one is two of its words, swapped (twelve shifts and inserts before).

; CHECK-LABEL: b32:
; CHECK-COUNT-2: extr
; CHECK-NOT: bfeil
; CHECK: or
; CHECK: device_store
; OFF-LABEL: b32:
; OFF: bfeil
; CHECK-LABEL: b64:
; CHECK-COUNT-4: extr
; CHECK-NOT: bfi
; CHECK-NOT: bfeil
; CHECK: device_store
; OFF-LABEL: b64:
; OFF: bfi
define agx_kernel void @b32(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %i
  %x = load i32, ptr addrspace(1) %a
  %r = call i32 @llvm.bswap.i32(i32 %x)
  store i32 %r, ptr addrspace(1) %a
  ret void
}
define agx_kernel void @b64(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i64, ptr addrspace(1) %p, i32 %i
  %x = load i64, ptr addrspace(1) %a
  %r = call i64 @llvm.bswap.i64(i64 %x)
  store i64 %r, ptr addrspace(1) %a
  ret void
}
define agx_kernel void @b16(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i16, ptr addrspace(1) %p, i32 %i
  %x = load i16, ptr addrspace(1) %a
  %r = call i16 @llvm.bswap.i16(i16 %x)
  store i16 %r, ptr addrspace(1) %a
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare i16 @llvm.bswap.i16(i16)
