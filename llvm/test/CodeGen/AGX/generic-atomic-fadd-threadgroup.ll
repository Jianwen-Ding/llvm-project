; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
;
; A generic float atomic add that may name threadgroup memory: its threadgroup
; branch (made by agx-lower-generic-addresses, after the first atomic expansion)
; is a compare-and-swap loop: threadgroup memory has no float atomics.
;
; CHECK-LABEL: k:
; CHECK:         atomic sub,
; CHECK:         threadgroup_atomic cmpxchg
@tg = internal addrspace(3) global [4 x float] poison
declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @k(ptr addrspace(1) %d, i32 %sel) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %t = addrspacecast ptr addrspace(3) @tg to ptr
  %g = addrspacecast ptr addrspace(1) %d to ptr
  %c = icmp eq i32 %sel, 0
  %p = select i1 %c, ptr %t, ptr %g
  %r = atomicrmw fadd ptr %p, float 1.0 monotonic
  ret void
}
