; A 32-bit value widened to 64 bits in one block and used as 32 bits in
; others costs nothing: the widening is the value itself and a zero high
; half, which no instruction reads (agx-tv, alive2's AGX lifter, proves the
; code of both kernels).
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: narrow_use:
; CHECK:      get_sr r1, sr52
; CHECK-NOT:  iadd r{{[0-9]+}}_r{{[0-9]+}}, r1, 0
; CHECK:      iadd r2, r1, 7
; CHECK-NEXT: device_store 0, i32, x, r2, u0_u1, r1, unsigned
; CHECK-LABEL: wide_use:
; CHECK:      device_store 0, i32, xy,
target triple = "agx"
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.simdgroup.index.in.threadgroup()

; Widened in the entry block, used narrow in another: the high half is dead.
define agx_kernel void @narrow_use(ptr addrspace(1) %out, i32 %n) {
entry:
  %l = call i32 @llvm.agx.thread.index.in.simdgroup()
  %l64 = zext i32 %l to i64
  %c = icmp slt i32 %l, %n
  br i1 %c, label %body, label %done
body:
  %t = trunc i64 %l64 to i32
  %v = add i32 %t, 7
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %p
  br label %done
done:
  ret void
}

; Widened, then used as 64 bits: the zero high half is needed.
define agx_kernel void @wide_use(ptr addrspace(1) %out, i32 %a, i32 %n) {
entry:
  %w = zext i32 %a to i64
  %c = icmp slt i32 %a, %n
  br i1 %c, label %body, label %done
body:
  %s = shl i64 %w, 20
  store i64 %s, ptr addrspace(1) %out
  br label %done
done:
  ret void
}
