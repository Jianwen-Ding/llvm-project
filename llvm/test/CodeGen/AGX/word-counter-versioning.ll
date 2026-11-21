; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -o /dev/null -print-after=agx-word-counter-versioning < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -o /dev/null -print-after=agx-word-counter-versioning -agx-loop-word-counters=0 < %s 2>&1 | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; Two buffers walked by 64-bit byte counters stepped by a runtime stride d (the
; PTX front end's addresses, after strength reduction): a copy for a stride
; of whole words whose count of words over the loop fits in 32 bits, from the
; exit test c + s < n (no count known) without a division, with one 32-bit
; counter of words that both loads take as their index register (L15-E674).

; CHECK-LABEL: @k(
; CHECK: [[Q:%.*]] = ashr i64 %d, 2
; CHECK: and i64 %d, 3
; CHECK: icmp sgt i64 {{%.*}}, 0
; CHECK: [[NOT:%.*]] = xor i1 {{%.*}}, true
; CHECK-NEXT: br i1 [[NOT]], label %[[ORIG:.*]], label %[[FAST:.*]]
; CHECK: [[ORIG]]:
; CHECK: [[FAST]]:
; CHECK-NEXT: [[Q32:%.*]] = trunc i64 [[Q]] to i32
; CHECK: loop:
; CHECK-NEXT: [[W:%.*]] = phi i32 [ 0, %[[FAST]] ], [ [[WN:%.*]], %loop ]
; CHECK: [[WS:%.*]] = sext i32 [[W]] to i64
; CHECK: getelementptr i32, ptr addrspace(1) {{%.*}}, i64 [[WS]]
; CHECK: getelementptr i32, ptr addrspace(1) {{%.*}}, i64 [[WS]]
; CHECK: [[WN]] = add nsw i32 [[W]], [[Q32]]
; OFF-LABEL: @k(
; OFF-NOT: %words
; OFF: ret void

; ASM-LABEL: k:
; ASM: device_load {{[01]}}, i32, x, r{{[0-9]+}}, r{{[0-9]+_r[0-9]+}}, [[I:r[0-9]+]], signed
; ASM-NEXT: device_load {{[01]}}, i32, x, r{{[0-9]+}}, r{{[0-9]+_r[0-9]+}}, [[I]], signed

; A constant stride (a row's short run) is left as it was.
; CHECK-LABEL: @constant_stride(
; CHECK-NOT: ashr i64
; CHECK: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()
; out[t + 64 * c] = a[pa] + b[pb] for c = t, t + s, ... while c + s < n (rotated), pa/pb byte offsets stepped by d
define agx_kernel void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %n, i32 %s, i64 %d) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %ai = ptrtoint ptr addrspace(1) %a to i64
  %bi = ptrtoint ptr addrspace(1) %b to i64
  %t4 = shl i32 %t, 2
  %t64 = zext i32 %t4 to i64
  %pa0 = add i64 %ai, %t64
  %pb0 = add i64 %bi, %t64
  br label %loop
loop:
  %c = phi i32 [ 0, %entry ], [ %c.next, %loop ]
  %pa = phi i64 [ %pa0, %entry ], [ %pa.next, %loop ]
  %pb = phi i64 [ %pb0, %entry ], [ %pb.next, %loop ]
  %ap = inttoptr i64 %pa to ptr addrspace(1)
  %x = load i32, ptr addrspace(1) %ap, align 4
  %bp = inttoptr i64 %pb to ptr addrspace(1)
  %y = load i32, ptr addrspace(1) %bp, align 4
  %r = add i32 %x, %y
  %oi = mul i32 %c, 64
  %oj = add i32 %oi, %t
  %oz = zext i32 %oj to i64
  %op = getelementptr i32, ptr addrspace(1) %out, i64 %oz
  store i32 %r, ptr addrspace(1) %op
  %pa.next = add i64 %pa, %d
  %pb.next = add i64 %pb, %d
  %c.next = add i32 %c, %s
  %cmp = icmp slt i32 %c.next, %n
  br i1 %cmp, label %loop, label %done
done:
  ret void
}

define agx_kernel void @constant_stride(ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %n) {
entry:
  %ai = ptrtoint ptr addrspace(1) %a to i64
  br label %loop
loop:
  %c = phi i32 [ 0, %entry ], [ %c.next, %loop ]
  %pa = phi i64 [ %ai, %entry ], [ %pa.next, %loop ]
  %ap = inttoptr i64 %pa to ptr addrspace(1)
  %x = load i32, ptr addrspace(1) %ap, align 4
  %op = getelementptr i32, ptr addrspace(1) %out, i32 %c
  store i32 %x, ptr addrspace(1) %op
  %pa.next = add i64 %pa, 128
  %c.next = add i32 %c, 1
  %cmp = icmp slt i32 %c.next, %n
  br i1 %cmp, label %loop, label %done
done:
  ret void
}
