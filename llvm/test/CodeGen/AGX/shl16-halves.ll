; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-shl16-halves=0 < %s | FileCheck %s --check-prefix=OFF

; x << 16 in a loop (a bfloat's widening, L15-E559): x's low half moved
; under a zero half that the preheader sets once and the loop carries
; (AGXInvariantHalves), one ALU move a step instead of a bfi on the shared
; unit. Outside a loop the bfi stays (the zero would be a move of its own),
; and in a block of more than 4 such shifts (each carried value holds a
; register through the loop).

; CHECK-LABEL: loop:
; CHECK: Loop Header
; CHECK-NOT: bfi
; CHECK-NOT: bitop 0101, r{{[0-9]+}}l, r{{[0-9]+}}l, 0
; CHECK: bitop 0101, r{{[0-9]+}}h, r{{[0-9]+}}l, 0
; CHECK-NOT: bfi
; CHECK-NOT: bitop 0101, r{{[0-9]+}}l, r{{[0-9]+}}l, 0
; CHECK: jmp_exec_any
; OFF-LABEL: loop:
; OFF: Loop Header
; OFF: bfi r{{[0-9]+}}, 0, r{{[0-9]+}}, 16
; CHECK-LABEL: straight:
; CHECK: bfi r{{[0-9]+}}, 0, r{{[0-9]+}}, 16

define agx_kernel void @loop(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %n) {
entry:
  br label %l
l:
  %i = phi i32 [ 0, %entry ], [ %i1, %l ]
  %acc = phi float [ 0.0, %entry ], [ %acc1, %l ]
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %i
  %x = load i32, ptr addrspace(1) %a
  %s = shl i32 %x, 16
  %f = bitcast i32 %s to float
  %acc1 = fadd float %acc, %f
  %i1 = add i32 %i, 32
  %c = icmp ult i32 %i1, %n
  br i1 %c, label %l, label %exit
exit:
  store float %acc1, ptr addrspace(1) %q
  ret void
}

define agx_kernel void @straight(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i32, ptr addrspace(1) %p, i32 %i
  %x = load i32, ptr addrspace(1) %a
  %s = shl i32 %x, 16
  store i32 %s, ptr addrspace(1) %a
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
