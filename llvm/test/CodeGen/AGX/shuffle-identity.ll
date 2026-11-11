; RUN: llc -mtriple=agx -mcpu=agx-g14g -stop-after=agx-codegenprepare < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=agx -mcpu=agx-g14g < %s | FileCheck %s

; Shuffles that undo each other, through selects and element moves that
; depend only on the thread index, are the value they started from: each
; thread reads back its own bits. (The PTX front end's mma emulation moves D from the
; matrix unit's layout to the PTX fragments, and the next mma's C back.)

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.simd.shuffle.i32(i32, i32)

; lane ^ 5 twice.
; CHECK-LABEL: xor_twice:
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @xor_twice(ptr addrspace(1) %out, i32 %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %v = add i32 %x, %lane
  %i = xor i32 %lane, 5
  %a = call i32 @llvm.agx.simd.shuffle.i32(i32 %v, i32 %i)
  %b = call i32 @llvm.agx.simd.shuffle.i32(i32 %a, i32 %i)
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %lane
  store i32 %b, ptr addrspace(1) %p
  ret void
}

; Two values exchanged between even and odd threads in two registers, and
; back: each thread selects the register that holds its own value.
; IR-LABEL: @pairs(
; IR-NOT:     simd.shuffle
; IR:         store <2 x float> %acc
define agx_kernel void @pairs(ptr addrspace(1) %out, <2 x float> %acc) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %e0 = extractelement <2 x float> %acc, i64 0
  %e1 = extractelement <2 x float> %acc, i64 1
  %i0 = bitcast float %e0 to i32
  %i1 = bitcast float %e1 to i32
  ; thread t keeps element 1 - (t & 1) of thread t ^ 1 first
  %odd = and i32 %lane, 1
  %src = xor i32 %lane, 1
  %s0 = call i32 @llvm.agx.simd.shuffle.i32(i32 %i0, i32 %src)
  %s1 = call i32 @llvm.agx.simd.shuffle.i32(i32 %i1, i32 %src)
  %isodd = icmp ne i32 %odd, 0
  %m0 = select i1 %isodd, i32 %s1, i32 %s0
  %m1 = select i1 %isodd, i32 %s0, i32 %s1
  ; and back
  %t0 = call i32 @llvm.agx.simd.shuffle.i32(i32 %m0, i32 %src)
  %t1 = call i32 @llvm.agx.simd.shuffle.i32(i32 %m1, i32 %src)
  %r0 = select i1 %isodd, i32 %t0, i32 %t1
  %r1 = select i1 %isodd, i32 %t1, i32 %t0
  %f0 = bitcast i32 %r0 to float
  %f1 = bitcast i32 %r1 to float
  %v0 = insertelement <2 x float> poison, float %f0, i64 0
  %v1 = insertelement <2 x float> %v0, float %f1, i64 1
  %p = getelementptr <2 x float>, ptr addrspace(1) %out, i32 %lane
  store <2 x float> %v1, ptr addrspace(1) %p
  ret void
}

; Not an identity: lane ^ 5 then lane ^ 4 reads thread lane ^ 1.
; CHECK-LABEL: not_identity:
; CHECK:         simd_shuffle
; CHECK:         stop
define agx_kernel void @not_identity(ptr addrspace(1) %out, i32 %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %v = add i32 %x, %lane
  %i = xor i32 %lane, 5
  %j = xor i32 %lane, 4
  %a = call i32 @llvm.agx.simd.shuffle.i32(i32 %v, i32 %i)
  %b = call i32 @llvm.agx.simd.shuffle.i32(i32 %a, i32 %j)
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %lane
  store i32 %b, ptr addrspace(1) %p
  ret void
}
