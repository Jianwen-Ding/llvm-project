; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-narrow-bounded-after-counters=0 < %s | FileCheck %s --check-prefix=OFF

; A causal mask j <= row in a loop over j = lane + c (MLIR's 64-bit index):
; once AGXNarrowInductions has narrowed the counter c, lane + zext(c) is a
; tree of 32-bit values below 2^32, compared in 32 bits (L15-E426).

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare float @llvm.maxnum.f32(float, float)

; CHECK-LABEL: mask:
; CHECK: .LBB0_{{[0-9]+}}:
; CHECK-NOT: icmpsel ueq
; CHECK: icmpsel ugt, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: jmp_exec_any
; OFF-LABEL: mask:
; OFF: icmpsel ueq
define agx_kernel void @mask(ptr addrspace(1) %s, ptr addrspace(1) %o) #0 {
entry:
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %row = call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %lane64 = zext i32 %lane to i64
  %row64 = zext i32 %row to i64
  %base = shl i32 %row, 10
  %bl = add i32 %base, %lane
  %end = sub nuw nsw i64 1024, %lane64
  br label %l
l:
  %c = phi i64 [ 0, %entry ], [ %c1, %l ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %l ]
  %j = add nuw nsw i64 %c, %lane64
  %c32 = trunc nuw nsw i64 %c to i32
  %ix = add i32 %bl, %c32
  %ix64 = zext i32 %ix to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %s, i64 %ix64
  %v = load float, ptr addrspace(1) %p, align 4
  %out = icmp ugt i64 %j, %row64
  %vm = select i1 %out, float 0xFFF0000000000000, float %v
  %m1 = call float @llvm.maxnum.f32(float %m, float %vm)
  %c1 = add nuw nsw i64 %c, 32
  %go = icmp ult i64 %c1, %end
  br i1 %go, label %l, label %e
e:
  %q = getelementptr inbounds float, ptr addrspace(1) %o, i32 %bl
  store float %m1, ptr addrspace(1) %q, align 4
  ret void
}

attributes #0 = { "agx-upper-bounds:air.threadgroup_position_in_grid"="1024,0,0" }
