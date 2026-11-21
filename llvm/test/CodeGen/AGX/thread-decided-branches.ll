; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-thread-decided-branches=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; MLIR's find_first: an scf.while whose condition guards its load,
; in = j < 256; v = in ? x[j] : 0; go = in & (v <= t). The header's edge for
; j >= 256 decides the latch's branch: it goes to the exit, and the latch,
; entered past the guard only, tests v <= t alone (L15-E685).

; CHECK-LABEL: @find_first(
; CHECK:       [[IN:%.*]] = icmp samesign ult i64 [[J:%.*]], 256
; CHECK-NEXT:  br i1 [[IN]], label %[[LOAD:.*]], label %[[EXIT:.*]]
; CHECK:       [[GO:%.*]] = fcmp ole float
; CHECK-NOT:   and i1
; CHECK:       br i1 [[GO]], label %{{.*}}, label %[[EXIT]]
; OFF-LABEL: @find_first(
; OFF:       and i1
; ASM-LABEL: find_first:
; ASM:       if_icmp r0l, ult
; ASM:       device_load
; ASM:       fcmpsel lte
; ASM-NOT:   icmpsel
; ASM:       while_icmp
define agx_kernel void @find_first(ptr addrspace(1) %x, ptr addrspace(1) %o, float %t) {
entry:
  %tid = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %tid, 8
  %rowz = zext i32 %row to i64
  %base = getelementptr inbounds [4 x i8], ptr addrspace(1) %x, i64 %rowz
  br label %head

head:
  %j = phi i64 [ %jn, %latch ], [ 0, %entry ]
  %in = icmp samesign ult i64 %j, 256
  br i1 %in, label %load, label %latch

load:
  %p = getelementptr inbounds [4 x i8], ptr addrspace(1) %base, i64 %j
  %l = load float, ptr addrspace(1) %p, align 4
  br label %latch

latch:
  %v = phi float [ %l, %load ], [ 0.000000e+00, %head ]
  %le = fcmp ole float %v, %t
  %go = and i1 %in, %le
  %jn = add nuw nsw i64 %j, 1
  br i1 %go, label %head, label %exit

exit:
  %r = trunc nuw nsw i64 %j to i32
  %tz = zext i32 %tid to i64
  %q = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %o, i64 %tz
  store i32 %r, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
