; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-stack-offset-fold=0 < %s | FileCheck %s --check-prefix=OFF

; A stack access at a frame index plus a constant (a private array's element)
; is an immediate offset, the object's offset plus the constant (b, at 32: b[5]
; at 52), instead of the frame address and an iadd per access (L15-E288).

; CHECK-LABEL: priv2:
; CHECK: stack_load 1, i32, x, {{r[0-9]+}}, 52
; OFF-LABEL: priv2:
; OFF: mov_imm [[B:r[0-9]+]], 32
; OFF: iadd [[R:r[0-9]+]], [[B]], 20
; OFF: stack_load 1, i32, x, {{r[0-9]+}}, [[R]]
define agx_kernel void @priv2(ptr addrspace(1) %in, ptr addrspace(1) %idx, ptr addrspace(1) %out) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr float, ptr addrspace(1) %in, i32 %t
  %v = load float, ptr addrspace(1) %pa
  %pi = getelementptr i32, ptr addrspace(1) %idx, i32 %t
  %j0 = load i32, ptr addrspace(1) %pi
  %j = and i32 %j0, 7
  %a = alloca [8 x float], align 4, addrspace(5)
  %b = alloca [8 x float], align 4, addrspace(5)
  br label %loop
loop:
  %k = phi i32 [ 0, %0 ], [ %k1, %loop ]
  %kf = sitofp i32 %k to float
  %m = fmul float %v, %kf
  %m2 = fadd float %m, 100.0
  %ga = getelementptr [8 x float], ptr addrspace(5) %a, i32 0, i32 %k
  store volatile float %m, ptr addrspace(5) %ga
  %gb = getelementptr [8 x float], ptr addrspace(5) %b, i32 0, i32 %k
  store volatile float %m2, ptr addrspace(5) %gb
  %k1 = add i32 %k, 1
  %c = icmp ult i32 %k1, 8
  br i1 %c, label %loop, label %done
done:
  %la = getelementptr [8 x float], ptr addrspace(5) %a, i32 0, i32 %j
  %ra = load float, ptr addrspace(5) %la
  %lb = getelementptr [8 x float], ptr addrspace(5) %b, i32 0, i32 %j
  %rb = load float, ptr addrspace(5) %lb
  %lb5 = getelementptr [8 x float], ptr addrspace(5) %b, i32 0, i32 5
  %rb5 = load volatile float, ptr addrspace(5) %lb5
  %s = fadd float %ra, %rb
  %s2 = fadd float %s, %rb5
  %po = getelementptr float, ptr addrspace(1) %out, i32 %t
  store float %s2, ptr addrspace(1) %po
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
