; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -o /dev/null -print-after=agx-wide-index-versioning < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -o /dev/null -print-after=agx-wide-index-versioning -agx-loop-wide-products=0 < %s 2>&1 | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; A loop's 64-bit products of a sign-extended value and a loop-invariant
; one sign-extended and scaled, (long)x * y * 4 and (long)x * y * 12: the
; loop runs as a copy where y * 4 and y * 12 fit in 32 bits, whose products
; are of two sign-extended 32-bit values (a 32 x 32 -> 64-bit multiply-add),
; and as it was otherwise (L15-E670).

; CHECK-LABEL: @k(
; CHECK: loop.preheader:
; CHECK-NEXT: [[A:%.*]] = sub i32 %y, -536870912
; CHECK-NEXT: [[FA:%.*]] = icmp ugt i32 [[A]], 1073741823
; CHECK-NEXT: [[B:%.*]] = sub i32 %y, -178956970
; CHECK-NEXT: [[FB:%.*]] = icmp ugt i32 [[B]], 357913940
; CHECK-NEXT: [[F:%.*]] = or i1 [[FA]], [[FB]]
; CHECK-NEXT: br i1 [[F]], label %[[ORIG:.*]], label %[[FAST:.*]]
; CHECK: [[ORIG]]:
; CHECK: mul i64 {{%.*}}, %s4
; CHECK: mul i64 %s12, {{%.*}}
; CHECK: [[FAST]]:
; CHECK-NEXT: [[Y4:%.*]] = shl nsw i32 %y, 2
; CHECK-NEXT: [[S4:%.*]] = sext i32 [[Y4]] to i64
; CHECK-NEXT: [[Y12:%.*]] = mul nsw i32 %y, 12
; CHECK-NEXT: [[S12:%.*]] = sext i32 [[Y12]] to i64
; CHECK: mul i64 %sx, [[S4]]
; CHECK: mul i64 [[S12]], %sx
; OFF-LABEL: @k(
; OFF-NOT: shl nsw i32 %y, 2
; OFF: ret void

; The fast copy's products and their sum: two signed 32 x 32 -> 64-bit
; multiply-adds, the second with the first as its addend.
; ASM-LABEL: k:
; ASM: imadd [[P:r[0-9]+_r[0-9]+]], r{{[0-9]+}}.sx, r{{[0-9]+}}.sx, 0
; ASM-NEXT: imadd r{{[0-9]+_r[0-9]+}}, r{{[0-9]+}}.sx, r{{[0-9]+}}.sx, [[P]]

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @k(ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %n, i32 %y) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %sy = sext i32 %y to i64
  %s4 = shl nsw i64 %sy, 2
  %s12 = mul nsw i64 %sy, 12
  %base = mul i32 %t, %n
  %pos = icmp sgt i32 %n, 0
  br i1 %pos, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx = add i32 %base, %i
  %ia = sext i32 %idx to i64
  %pa = getelementptr i32, ptr addrspace(1) %a, i64 %ia
  %x = load i32, ptr addrspace(1) %pa
  %sx = sext i32 %x to i64
  %p1 = mul i64 %sx, %s4
  %p2 = mul i64 %s12, %sx
  %r = add i64 %p1, %p2
  %po = getelementptr i64, ptr addrspace(1) %out, i64 %ia
  store i64 %r, ptr addrspace(1) %po
  %i.next = add nuw nsw i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %done

done:
  ret void
}

; A loop with a SIMD-group shuffle: versioned on a uniform y (an argument),
; not on one each thread loads (the group's lanes could take different
; copies).
; CHECK-LABEL: @shuffle_uniform(
; CHECK: shl nsw i32 %y, 2
; CHECK-LABEL: @shuffle_varying(
; CHECK-NOT: shl nsw i32 {{%.*}}, 2
; CHECK: ret void

declare i32 @llvm.agx.simd.shuffle.xor.i32(i32, i32) #0

define agx_kernel void @shuffle_uniform(ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %n, i32 %y) {
entry:
  %sy = sext i32 %y to i64
  %s4 = shl nsw i64 %sy, 2
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ia = zext i32 %i to i64
  %pa = getelementptr i32, ptr addrspace(1) %a, i64 %ia
  %x = load i32, ptr addrspace(1) %pa
  %v = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %x, i32 1)
  %sx = sext i32 %v to i64
  %p = mul i64 %sx, %s4
  %po = getelementptr i64, ptr addrspace(1) %out, i64 %ia
  store i64 %p, ptr addrspace(1) %po
  %i.next = add nuw nsw i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %done

done:
  ret void
}

define agx_kernel void @shuffle_varying(ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %py = getelementptr i32, ptr addrspace(1) %a, i32 %t
  %y = load i32, ptr addrspace(1) %py
  %sy = sext i32 %y to i64
  %s4 = shl nsw i64 %sy, 2
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ia = zext i32 %i to i64
  %pa = getelementptr i32, ptr addrspace(1) %a, i64 %ia
  %x = load i32, ptr addrspace(1) %pa
  %v = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %x, i32 1)
  %sx = sext i32 %v to i64
  %p = mul i64 %sx, %s4
  %po = getelementptr i64, ptr addrspace(1) %out, i64 %ia
  store i64 %p, ptr addrspace(1) %po
  %i.next = add nuw nsw i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %done

done:
  ret void
}

attributes #0 = { convergent nounwind }
