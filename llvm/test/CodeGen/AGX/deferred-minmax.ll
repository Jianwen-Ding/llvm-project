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

; An nsz maximum (no zero sign to keep) lowers to its compare-select and NaN
; select already: it stays (L15-E445).
; CHECK-LABEL: @nsz(
; CHECK:       call nsz float @llvm.maximum.f32
define agx_kernel void @nsz(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %mn, %loop ]
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %mn = call nsz float @llvm.maximum.f32(float %m, float %v)
  %in = add nuw i32 %i, 1
  %c = icmp ult i32 %in, %n
  br i1 %c, label %loop, label %done
done:
  store float %mn, ptr addrspace(1) %o, align 4
  ret void
}

; A running maximumnum from a number (never a NaN) whose old and new values
; also feed the loop's e^(m - m') (no zero's sign reaches it): those uses take
; the deferred value, the store after the loop the exact one (L15-E446).
declare float @llvm.maximumnum.f32(float, float)
declare float @llvm.exp2.f32(float)
; CHECK-LABEL: @online(
; CHECK:       loop:
; CHECK-NOT:   @llvm.maximumnum
; CHECK:       fcmp oge float
; CHECK:       icmp sgt i32
; CHECK:       fsub float %{{.*}}.m,
; CHECK:       ret void
define agx_kernel void @online(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %m = phi float [ -3.0e+38, %entry ], [ %mn, %loop ]
  %s = phi float [ 0.0, %entry ], [ %sn, %loop ]
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %mn = call float @llvm.maximumnum.f32(float %m, float %v)
  %d = fsub float %m, %mn
  %e = call float @llvm.exp2.f32(float %d)
  %sn = fmul float %s, %e
  %in = add nuw i32 %i, 1
  %c = icmp ult i32 %in, %n
  br i1 %c, label %loop, label %done
done:
  store float %mn, ptr addrspace(1) %o, align 4
  %o1 = getelementptr inbounds float, ptr addrspace(1) %o, i32 1
  store float %sn, ptr addrspace(1) %o1, align 4
  ret void
}
