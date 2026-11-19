; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; A loop's running llvm.maximum used only by itself: an ordered compare-
; select, the last NaN seen and the signed maximum of the values' bits in
; the loop, maximum's NaN and +0 decided after it (L15-E444). @twice's
; running value has another use in the loop: it stays.

declare float @llvm.maximum.f32(float, float)

; CHECK-LABEL: @rowmax(
; CHECK:       loop:
; CHECK-NOT:   @llvm.maximum
; CHECK:       fcmp ogt float
; CHECK:       fcmp uno float
; CHECK:       icmp sgt i32
; CHECK:       done:
; CHECK:       icmp sge i32
; CHECK:       fcmp uno float
; CHECK:       ret void
define agx_kernel void @rowmax(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %mn, %loop ]
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %mn = call float @llvm.maximum.f32(float %m, float %v)
  %in = add nuw i32 %i, 1
  %c = icmp ult i32 %in, %n
  br i1 %c, label %loop, label %done
done:
  store float %mn, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: @twice(
; CHECK:       @llvm.maximum.f32
define agx_kernel void @twice(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %mn, %loop ]
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %mn = call float @llvm.maximum.f32(float %m, float %v)
  store float %mn, ptr addrspace(1) %p, align 4
  %in = add nuw i32 %i, 1
  %c = icmp ult i32 %in, %n
  br i1 %c, label %loop, label %done
done:
  store float %mn, ptr addrspace(1) %o, align 4
  ret void
}
