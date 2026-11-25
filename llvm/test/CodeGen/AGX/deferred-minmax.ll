; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-nan-in-m=0 < %s | FileCheck %s --check-prefix=CARRY

; A loop's running llvm.maximum used only by itself: in the loop maximum
; without the order of zeros (nsz: one fcmpsel, which keeps a NaN: L15-E453)
; and the signed maximum of the values' bits, maximum's +0 decided after it
; (L15-E444; before E453 an ordered compare-select and the last NaN seen beside
; it, CARRY). @twice's running value has another use in the loop: it stays.

declare float @llvm.maximum.f32(float, float)

; CHECK-LABEL: @rowmax(
; CHECK:       loop:
; CHECK:       call nsz float @llvm.maximum.f32(float %m.m, float %v)
; CHECK-NOT:   fcmp uno
; CHECK:       icmp sgt i32
; CHECK:       done:
; CHECK:       icmp sge i32
; CHECK-NOT:   fcmp uno float
; CHECK:       ret void
; CARRY-LABEL: @rowmax(
; CARRY:       loop:
; CARRY-NOT:   @llvm.maximum
; CARRY:       fcmp ogt float
; CARRY:       fcmp uno float
; CARRY:       icmp sgt i32
; CARRY:       done:
; CARRY:       icmp sge i32
; CARRY:       fcmp uno float
; CARRY:       ret void
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

; An nsz maximum (no zero sign to keep) lowers to one fcmpsel already: it
; stays (L15-E445, E452).
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

; A vector's running maximum reduced after the loop: the results of its
; lanes, combined by maximum nsz of their m's and smax of their z's, one
; result instead of four and three maximums (L15-E454).
; CHECK-LABEL: @vec4(
; CHECK:       done:
; CHECK:       call nsz float @llvm.maximum.f32(float %m.i0.m.next, float %m.i1.m.next)
; CHECK:       call i32 @llvm.smax.i32
; CHECK:       call nsz float @llvm.maximum.f32
; CHECK:       call i32 @llvm.smax.i32
; CHECK:       call nsz float @llvm.maximum.f32
; CHECK:       call i32 @llvm.smax.i32
; CHECK-NOT:   @llvm.maximum.f32
; CHECK:       select i1 %{{.*}}, float 0.000000e+00, float
; CHECK:       ret void
define agx_kernel void @vec4(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %m = phi <4 x float> [ splat (float 0xFFF0000000000000), %entry ], [ %mn, %loop ]
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %x, i32 %i
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  %mn = call <4 x float> @llvm.maximum.v4f32(<4 x float> %m, <4 x float> %v)
  %in = add nuw i32 %i, 1
  %c = icmp ult i32 %in, %n
  br i1 %c, label %loop, label %done
done:
  %e0 = extractelement <4 x float> %mn, i64 0
  %e1 = extractelement <4 x float> %mn, i64 1
  %m01 = call float @llvm.maximum.f32(float %e0, float %e1)
  %e2 = extractelement <4 x float> %mn, i64 2
  %m012 = call float @llvm.maximum.f32(float %m01, float %e2)
  %e3 = extractelement <4 x float> %mn, i64 3
  %r = call float @llvm.maximum.f32(float %m012, float %e3)
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)

; A running maximum of extended halves from -inf: the deferred chain in half,
; its result extended once after the loop (L15-E462).
; CHECK-LABEL: @halves(
; CHECK:       loop:
; CHECK-NOT:   fpext
; CHECK:       call nsz half @llvm.maximum.f16
; CHECK:       icmp sgt i16
; CHECK:       done:
; CHECK:       fpext half
; CHECK:       ret void
define agx_kernel void @halves(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %mn, %loop ]
  %p = getelementptr inbounds half, ptr addrspace(1) %x, i32 %i
  %h = load half, ptr addrspace(1) %p, align 2
  %v = fpext half %h to float
  %mn = call float @llvm.maximum.f32(float %m, float %v)
  %in = add nuw i32 %i, 1
  %c = icmp ult i32 %in, %n
  br i1 %c, label %loop, label %done
done:
  store float %mn, ptr addrspace(1) %o, align 4
  ret void
}

; The same with maximum (MLIR's arith.maximumf): its NaN-carrying deferred
; value also feeds the zero-sign-free uses in the loop (L15-E463).
; CHECK-LABEL: @online_maximum(
; CHECK:       loop:
; CHECK:       %m.m.next = call nsz float @llvm.maximum.f32(float %m.m, float %v)
; CHECK:       fsub float %m.m, %m.m.next
; CHECK:       done:
; CHECK:       icmp sge i32
; CHECK:       ret void
define agx_kernel void @online_maximum(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %mn, %loop ]
  %s = phi float [ 0.0, %entry ], [ %sn, %loop ]
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %mn = call float @llvm.maximum.f32(float %m, float %v)
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
