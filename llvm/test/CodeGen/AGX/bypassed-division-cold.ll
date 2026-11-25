; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-rare-fallbacks < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-rare-fallbacks -agx-wide-division-unlikely=false < %s | FileCheck %s --check-prefix=OFF

; A test that a 64-bit division's operands fit 32 bits, (a | b) > 2^32 - 1
; (CodeGenPrepare's slow-division bypass reused): the 64-bit side unlikely
; (the expansion's own weights here) and its block marked !agx.cold, which
; AGXColdRegions moves out of line where the structurizer drops the edge's
; probability (L15-E860).

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: br i1 %wide, label %slow, label %fast, !prof
; CHECK: slow:
; CHECK-NEXT: !agx.cold
; OFF-LABEL: define agx_kernel void @k(
; OFF-NOT: !agx.cold

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @k(ptr addrspace(1) %p, i64 %d) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %g = getelementptr i64, ptr addrspace(1) %p, i32 %t
  %x = load i64, ptr addrspace(1) %g
  %o = or i64 %x, %d
  %wide = icmp ugt i64 %o, 4294967295
  br i1 %wide, label %slow, label %fast
slow:
  %q64 = udiv i64 %x, %d
  br label %end
fast:
  %x32 = trunc i64 %x to i32
  %d32 = trunc i64 %d to i32
  %q32 = udiv i32 %x32, %d32
  %q = zext i32 %q32 to i64
  br label %end
end:
  %r = phi i64 [ %q64, %slow ], [ %q, %fast ]
  store i64 %r, ptr addrspace(1) %g
  ret void
}
