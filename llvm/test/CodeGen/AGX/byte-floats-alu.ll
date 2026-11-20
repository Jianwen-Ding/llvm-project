; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-byte-floats-alu=0 < %s | FileCheck %s --check-prefix=OFF

; A signed byte's float in a loop (L15-E533): its 16-bit lane xor 0x6480 is
; the half 1152 + s, less 1152 exactly s, which the fma reads as its 16-bit
; source: an xor and an fadd16 on the ALU instead of convert s8_to_f on the
; shared unit. The half sum's byte likewise.

; CHECK-LABEL: k:
; CHECK-NOT: convert
; CHECK: mov_imm {{r[0-9]+[lh]?}}, 25728
; CHECK: xor
; CHECK: fadd16
; CHECK: fmadd32
; CHECK-NOT: convert
; CHECK: device_store
; OFF-LABEL: k:
; OFF: convert s8_to_f

declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @k(ptr addrspace(1) %a, ptr addrspace(1) %w, ptr addrspace(1) %o, ptr addrspace(1) %h) {
entry:
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %base = mul i32 %i, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc1, %loop ]
  %hacc = phi half [ 0.0, %entry ], [ %hacc1, %loop ]
  %ij = add i32 %base, %j
  %pa = getelementptr i8, ptr addrspace(1) %a, i32 %ij
  %b = load i8, ptr addrspace(1) %pa
  %pw = getelementptr float, ptr addrspace(1) %w, i32 %j
  %wv = load float, ptr addrspace(1) %pw
  %f = sitofp i8 %b to float
  %acc1 = call float @llvm.fma.f32(float %f, float %wv, float %acc)
  %g = sitofp i8 %b to half
  %hacc1 = fadd half %hacc, %g
  %j1 = add i32 %j, 1
  %c = icmp ult i32 %j1, 256
  br i1 %c, label %loop, label %exit
exit:
  %po = getelementptr float, ptr addrspace(1) %o, i32 %i
  store float %acc1, ptr addrspace(1) %po
  %ph = getelementptr half, ptr addrspace(1) %h, i32 %i
  store half %hacc1, ptr addrspace(1) %ph
  ret void
}
declare float @llvm.fma.f32(float, float, float)

; Alone (no loop to hoist its two constants into): the convert.
; CHECK-LABEL: alone:
; CHECK: convert s8_to_f
define agx_kernel void @alone(ptr addrspace(1) %a, ptr addrspace(1) %o) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr i8, ptr addrspace(1) %a, i32 %i
  %b = load i8, ptr addrspace(1) %pa
  %f = sitofp i8 %b to float
  %g = fmul float %f, 3.0
  %po = getelementptr float, ptr addrspace(1) %o, i32 %i
  store float %g, ptr addrspace(1) %po
  ret void
}

; A value below 1024 by its known bits plus a constant (q4 weights' nibble
; less 8, L15-E534): the half U | 0x6400 is 1024 + U, less 1024 - K it is
; U + K exactly.
; CHECK-LABEL: nibbles:
; CHECK-NOT: convert
; CHECK: or {{r[0-9]+}}l, {{r[0-9]+}}l, {{r[0-9]+}}l
; CHECK: fadd16
; CHECK: device_store
define agx_kernel void @nibbles(ptr addrspace(1) %a, ptr addrspace(1) %w, ptr addrspace(1) %o) {
entry:
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %base = mul i32 %i, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %a2, %loop ]
  %ij = add i32 %base, %j
  %pa = getelementptr i8, ptr addrspace(1) %a, i32 %ij
  %q = load i8, ptr addrspace(1) %pa
  %pw = getelementptr float, ptr addrspace(1) %w, i32 %j
  %wv = load float, ptr addrspace(1) %pw
  %qz = zext i8 %q to i32
  %lo = and i32 %qz, 15
  %lo8 = add nsw i32 %lo, -8
  %flo = sitofp i32 %lo8 to float
  %hi = lshr i32 %qz, 4
  %hi8 = add nsw i32 %hi, -8
  %fhi = sitofp i32 %hi8 to float
  %a1 = call float @llvm.fma.f32(float %flo, float %wv, float %acc)
  %a2 = call float @llvm.fma.f32(float %fhi, float %wv, float %a1)
  %j1 = add i32 %j, 1
  %c = icmp ult i32 %j1, 256
  br i1 %c, label %loop, label %exit
exit:
  %po = getelementptr float, ptr addrspace(1) %o, i32 %i
  store float %a2, ptr addrspace(1) %po
  ret void
}
