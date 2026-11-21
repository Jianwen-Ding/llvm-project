; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-pair-half-bitops=0 < %s | FileCheck %s --check-prefix=OFF

; Two bytes loaded into the halves of a register, each made a float by the
; xor with 0x6480 (L15-E533): one 32-bit xor with 0x64806480 for both
; (L15-E631), not one for each half.
; CHECK-LABEL: pair:
; CHECK: mov_imm {{r[0-9]+}}, 1686135936
; CHECK: xor {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
; CHECK-NOT: xor
; CHECK: device_store
; OFF-LABEL: pair:
; OFF: xor {{r[0-9]+}}l,
; OFF: xor {{r[0-9]+}}h,
define agx_kernel void @pair(ptr addrspace(1) %q, ptr addrspace(1) %x, ptr addrspace(1) %o) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %base = shl i32 %t, 8
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %a2, %loop ]
  %ij = add i32 %base, %j
  %pq = getelementptr i8, ptr addrspace(1) %q, i32 %ij
  %v = load <2 x i8>, ptr addrspace(1) %pq, align 2
  %b0 = extractelement <2 x i8> %v, i64 0
  %b1 = extractelement <2 x i8> %v, i64 1
  %f0 = sitofp i8 %b0 to float
  %f1 = sitofp i8 %b1 to float
  %px = getelementptr float, ptr addrspace(1) %x, i32 %j
  %x0 = load float, ptr addrspace(1) %px
  %px1 = getelementptr float, ptr addrspace(1) %px, i32 1
  %x1 = load float, ptr addrspace(1) %px1
  %a1 = call float @llvm.fma.f32(float %f0, float %x0, float %acc)
  %a2 = call float @llvm.fma.f32(float %f1, float %x1, float %a1)
  %j1 = add i32 %j, 2
  %c = icmp ult i32 %j1, 256
  br i1 %c, label %loop, label %exit
exit:
  %po = getelementptr float, ptr addrspace(1) %o, i32 %t
  store float %a2, ptr addrspace(1) %po
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.fma.f32(float, float, float)
