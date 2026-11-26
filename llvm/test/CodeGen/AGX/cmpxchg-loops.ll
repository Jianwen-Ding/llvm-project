; Compare-exchange loops that apply an operation the target has natively are
; that atomicrmw (AGXCmpXchgLoops): G13 and G14 have the 32-bit integer
; operations, G14 also the float add on device memory. Other loops stay.
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-cmpxchg-loops -S %s | FileCheck %s --check-prefixes=CHECK,G14
; RUN: opt -mtriple=agx -mcpu=agx-g13g -passes=agx-cmpxchg-loops -S %s | FileCheck %s --check-prefixes=CHECK,G13
; RUN: opt -mtriple=agx -mcpu=agx-g14g -O3 -S %s | FileCheck %s --check-prefix=O3
; RUN: llc -mtriple=agx -mcpu=agx-g14g -O3 %s -o /dev/null

; CHECK-LABEL: @fadd(
; G14: %[[R:.*]] = atomicrmw volatile fadd ptr addrspace(1) %p, float %v syncscope("device") monotonic, align 4
; G14-NEXT: %[[I:.*]] = bitcast float %[[R]] to i32
; G14-NOT: cmpxchg
; G14: store i32 %[[I]], ptr addrspace(1) %o
; G13: cmpxchg
; O3-LABEL: @fadd(
; O3: atomicrmw volatile fadd
; O3-NOT: cmpxchg
define agx_kernel void @fadd(ptr addrspace(1) %p, float %v, ptr addrspace(1) %o) {
entry:
  %init = load i32, ptr addrspace(1) %p, align 4
  br label %loop
loop:
  %old = phi i32 [ %init, %entry ], [ %found, %loop ]
  %of = bitcast i32 %old to float
  %nf = fadd float %of, %v
  %ni = bitcast float %nf to i32
  %x = cmpxchg volatile ptr addrspace(1) %p, i32 %old, i32 %ni syncscope("device") monotonic monotonic, align 4
  %found = extractvalue { i32, i1 } %x, 0
  %ok = extractvalue { i32, i1 } %x, 1
  br i1 %ok, label %done, label %loop
done:
  store i32 %found, ptr addrspace(1) %o
  ret void
}

; An integer add whose exit compares the value found with the one expected.
; CHECK-LABEL: @add(
; O3-LABEL: @add(
; CHECK: %[[A:.*]] = atomicrmw add ptr addrspace(1) %p, i32 %v acq_rel, align 4
; CHECK-NOT: cmpxchg
; CHECK: store i32 %[[A]], ptr addrspace(1) %o
define agx_kernel void @add(ptr addrspace(1) %p, i32 %v, ptr addrspace(1) %o) {
entry:
  br label %loop
loop:
  %old = phi i32 [ 0, %entry ], [ %found, %loop ]
  %n = add i32 %v, %old
  %x = cmpxchg ptr addrspace(1) %p, i32 %old, i32 %n acq_rel acquire, align 4
  %found = extractvalue { i32, i1 } %x, 0
  %ok = icmp eq i32 %found, %old
  br i1 %ok, label %done, label %loop
done:
  %r = phi i32 [ %found, %loop ]
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @umax_xchg(
; CHECK: atomicrmw umax ptr addrspace(1) %p, i32 %v monotonic
; CHECK: atomicrmw xchg ptr addrspace(1) %q, i32 7 monotonic
; CHECK-NOT: cmpxchg
define agx_kernel void @umax_xchg(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %v) {
entry:
  br label %loop
loop:
  %old = phi i32 [ 0, %entry ], [ %found, %loop ]
  %n = call i32 @llvm.umax.i32(i32 %old, i32 %v)
  %x = cmpxchg ptr addrspace(1) %p, i32 %old, i32 %n monotonic monotonic, align 4
  %found = extractvalue { i32, i1 } %x, 0
  %ok = extractvalue { i32, i1 } %x, 1
  br i1 %ok, label %mid, label %loop
mid:
  br label %loop2
loop2:
  %old2 = phi i32 [ 0, %mid ], [ %found2, %loop2 ]
  %x2 = cmpxchg ptr addrspace(1) %q, i32 %old2, i32 7 monotonic monotonic, align 4
  %found2 = extractvalue { i32, i1 } %x2, 0
  %ok2 = extractvalue { i32, i1 } %x2, 1
  br i1 %ok2, label %done, label %loop2
done:
  ret void
}

; A signed max as a select of its compare's operands (CUDA and MSL code
; spell it so), with the operands of the select swapped: a min.
; CHECK-LABEL: @select_minmax(
; CHECK: atomicrmw max ptr addrspace(1) %p, i32 %v monotonic
; CHECK: atomicrmw umin ptr addrspace(1) %p, i32 %v monotonic
; CHECK-NOT: cmpxchg
define agx_kernel void @select_minmax(ptr addrspace(1) %p, i32 %v) {
entry:
  br label %loop
loop:
  %old = phi i32 [ 0, %entry ], [ %found, %loop ]
  %c = icmp sgt i32 %old, %v
  %n = select i1 %c, i32 %old, i32 %v
  %x = cmpxchg ptr addrspace(1) %p, i32 %old, i32 %n monotonic monotonic, align 4
  %found = extractvalue { i32, i1 } %x, 0
  %ok = extractvalue { i32, i1 } %x, 1
  br i1 %ok, label %mid, label %loop
mid:
  br label %loop2
loop2:
  %old2 = phi i32 [ 0, %mid ], [ %found2, %loop2 ]
  %c2 = icmp ugt i32 %old2, %v
  %n2 = select i1 %c2, i32 %v, i32 %old2
  %x2 = cmpxchg ptr addrspace(1) %p, i32 %old2, i32 %n2 monotonic monotonic, align 4
  %found2 = extractvalue { i32, i1 } %x2, 0
  %ok2 = extractvalue { i32, i1 } %x2, 1
  br i1 %ok2, label %done, label %loop2
done:
  ret void
}

; Not natively (a float max), a weak exchange judged by its value (it may fail
; spuriously with the value expected), a second access, a value made in the
; loop used after it, and a subtraction of the old value: the loops stay.
; CHECK-LABEL: @kept(
; CHECK-COUNT-5: cmpxchg
; CHECK-NOT: atomicrmw
define agx_kernel void @kept(ptr addrspace(1) %p, float %v, i32 %w, ptr addrspace(1) %o) {
entry:
  br label %fmax
fmax:
  %o1 = phi i32 [ 0, %entry ], [ %f1, %fmax ]
  %of1 = bitcast i32 %o1 to float
  %m1 = call float @llvm.maxnum.f32(float %of1, float %v)
  %n1 = bitcast float %m1 to i32
  %x1 = cmpxchg ptr addrspace(1) %p, i32 %o1, i32 %n1 monotonic monotonic, align 4
  %f1 = extractvalue { i32, i1 } %x1, 0
  %k1 = extractvalue { i32, i1 } %x1, 1
  br i1 %k1, label %weak.pre, label %fmax
weak.pre:
  br label %weak
weak:
  %o2 = phi i32 [ 0, %weak.pre ], [ %f2, %weak ]
  %n2 = add i32 %o2, %w
  %x2 = cmpxchg weak ptr addrspace(1) %p, i32 %o2, i32 %n2 monotonic monotonic, align 4
  %f2 = extractvalue { i32, i1 } %x2, 0
  %k2 = icmp eq i32 %f2, %o2
  br i1 %k2, label %two.pre, label %weak
two.pre:
  br label %two
two:
  %o3 = phi i32 [ 0, %two.pre ], [ %f3, %two ]
  store i32 %o3, ptr addrspace(1) %p, align 4
  %n3 = add i32 %o3, %w
  %x3 = cmpxchg ptr addrspace(1) %p, i32 %o3, i32 %n3 monotonic monotonic, align 4
  %f3 = extractvalue { i32, i1 } %x3, 0
  %k3 = extractvalue { i32, i1 } %x3, 1
  br i1 %k3, label %new.pre, label %two
new.pre:
  br label %new
new:
  %o4 = phi i32 [ 0, %new.pre ], [ %f4, %new ]
  %n4 = add i32 %o4, %w
  %x4 = cmpxchg ptr addrspace(1) %p, i32 %o4, i32 %n4 monotonic monotonic, align 4
  %f4 = extractvalue { i32, i1 } %x4, 0
  %k4 = extractvalue { i32, i1 } %x4, 1
  br i1 %k4, label %sub.pre, label %new
sub.pre:
  %use = add i32 %n4, 1
  br label %sub
sub:
  %o5 = phi i32 [ 0, %sub.pre ], [ %f5, %sub ]
  %n5 = sub i32 %w, %o5
  %x5 = cmpxchg ptr addrspace(1) %p, i32 %o5, i32 %n5 monotonic monotonic, align 4
  %f5 = extractvalue { i32, i1 } %x5, 0
  %k5 = extractvalue { i32, i1 } %x5, 1
  br i1 %k5, label %done, label %sub
done:
  store i32 %use, ptr addrspace(1) %o
  ret void
}

declare i32 @llvm.umax.i32(i32, i32)
declare float @llvm.maxnum.f32(float, float)
