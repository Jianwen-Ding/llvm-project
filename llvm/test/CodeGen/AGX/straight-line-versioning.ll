; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-align-versioning < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-align-versioning -agx-straight-line-versioning-min=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=ASM

; A kernel of one block (its loops unrolled whole) with many half accesses at
; align 2 off its arguments (a row of vector<4xf16>: 16 elements here) runs a
; copy at the raised alignment where every base is 4-byte aligned (L15-E522):
; there the loads are 32-bit pairs and the store one of four halves.

; CHECK-LABEL: @rows(
; CHECK: align.check:
; CHECK: %{{.*}}.al4 = icmp eq i64
; CHECK: br i1 %{{.*}}, label %{{.*}}.aligned, label %entry
; CHECK: load <4 x half>, ptr addrspace(1) %{{.*}}, align 2
; CHECK: .aligned:
; CHECK: load <4 x half>, ptr addrspace(1) %{{.*}}, align 4
; CHECK: store <4 x half> %{{.*}}, ptr addrspace(1) %{{.*}}, align 4
; OFF-LABEL: @rows(
; OFF-NOT: align.check
; OFF: ret void
; ASM-LABEL: rows:
; ASM: device_load {{[01]}}, i32, xy
; ASM: device_store 0, i16, xyzw

define agx_kernel void @rows(ptr addrspace(1) %x, ptr addrspace(1) %y) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 7
  %i0 = zext i32 %row to i64
  %p0 = getelementptr half, ptr addrspace(1) %x, i64 %i0
  %v0 = load <4 x half>, ptr addrspace(1) %p0, align 2
  %i1 = add nuw nsw i64 %i0, 32
  %p1 = getelementptr half, ptr addrspace(1) %x, i64 %i1
  %v1 = load <4 x half>, ptr addrspace(1) %p1, align 2
  %i2 = add nuw nsw i64 %i0, 64
  %p2 = getelementptr half, ptr addrspace(1) %x, i64 %i2
  %v2 = load <4 x half>, ptr addrspace(1) %p2, align 2
  %s01 = fadd <4 x half> %v0, %v1
  %s = fadd <4 x half> %s01, %v2
  %q = getelementptr half, ptr addrspace(1) %y, i64 %i0
  store <4 x half> %s, ptr addrspace(1) %q, align 2
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
