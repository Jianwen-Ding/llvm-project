; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; A grid position is bounded by the kernel's known grid (agx-threads-per-grid)
; in a kernel without loops: a signed division of it by 4 is a shift. In a
; kernel with a loop it is not bounded (L15-E191: the folded offsets cost a
; 64-bit base each iteration there), and the division keeps its sign fix-up;
; nor without the attribute.

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: known:
; CHECK-NOT: asr
; CHECK: bfeil {{r[0-9]+}}, 0, {{r[0-9]+}}, 2
; CHECK: stop
define agx_kernel void @known(ptr addrspace(1) %o) #0 {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %d = sdiv i32 %i, 4
  store i32 %d, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: unknown:
; CHECK: asr
; CHECK: stop
define agx_kernel void @unknown(ptr addrspace(1) %o) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %d = sdiv i32 %i, 4
  store i32 %d, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: loop:
; CHECK: asr
; CHECK: stop
define agx_kernel void @loop(ptr addrspace(1) %o, i32 %n) #0 {
entry:
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %d = sdiv i32 %i, 4
  br label %l
l:
  %k = phi i32 [ 0, %entry ], [ %k1, %l ]
  %p = getelementptr i32, ptr addrspace(1) %o, i32 %k
  store i32 %d, ptr addrspace(1) %p, align 4
  %k1 = add i32 %k, 1
  %c = icmp ult i32 %k1, %n
  br i1 %c, label %l, label %x
x:
  ret void
}

attributes #0 = { "agx-threads-per-grid"="65536,1,1" }
