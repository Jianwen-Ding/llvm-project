; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-select-bool16=0 < %s | FileCheck %s --check-prefix=OFF

; Lane conditions shared by several comparisons of i1 values (a bitonic
; network's direction bits): type legalization keeps them as i16, and each
; select compares the i16 equality directly instead of first widening it to
; 32 bits.

; CHECK-LABEL: shared_lane_bits:
; CHECK-NOT: bitop 0101, {{r[0-9]+}}, {{r[0-9]+[lh]}}, 0
; CHECK: xor {{r[0-9]+}}[[H:[lh]]], {{r[0-9]+[lh]}}, {{r[0-9]+[lh]}}
; CHECK: icmpsel ueq, {{r[0-9]+}}, {{r[0-9]+}}[[H]], 0,
; CHECK: stop
define agx_kernel void @shared_lane_bits(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %pa = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %a = load i32, ptr addrspace(1) %pa, align 4
  %qa = getelementptr i32, ptr addrspace(1) %q, i32 %t
  %b = load i32, ptr addrspace(1) %qa, align 4
  %b1 = and i32 %lane, 1
  %l1 = icmp eq i32 %b1, 0
  %b2 = and i32 %lane, 2
  %u1 = icmp eq i32 %b2, 0
  %b4 = and i32 %lane, 4
  %u2 = icmp eq i32 %b4, 0
  %m1 = icmp eq i1 %l1, %u1
  %m2 = icmp eq i1 %l1, %u2
  %m3 = icmp eq i1 %u1, %u2
  %lo = call i32 @llvm.umin.i32(i32 %a, i32 %b)
  %hi = call i32 @llvm.umax.i32(i32 %a, i32 %b)
  %v1 = select i1 %m1, i32 %lo, i32 %hi
  %w = add i32 %v1, %a
  %v2 = select i1 %m2, i32 %w, i32 %b
  %v3 = select i1 %m3, i32 %v2, i32 %v1
  store i32 %v3, ptr addrspace(1) %pa, align 4
  ret void
}
; A boolean made by selects between comparisons and 0 / 1 (an argmax's
; "greater, or equal with a lower index") has its high 16 bits known zero:
; the selects on it compare its low half with 0, reading two 32-bit
; registers instead of three (L15-E366).
; CHECK-LABEL: and_or_flag:
; CHECK: icmpsel ueq, {{r[0-9]+}}, [[F:r[0-9]+]]l, 0, {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: icmpsel ueq, {{r[0-9]+}}, [[F]]l, 0, {{r[0-9]+}}, {{r[0-9]+}}
; OFF-LABEL: and_or_flag:
; OFF: icmpsel ueq, {{r[0-9]+}}, [[F:r[0-9]+]], 0, {{r[0-9]+}}, {{r[0-9]+}}
; OFF: icmpsel ueq, {{r[0-9]+}}, [[F]], 0, {{r[0-9]+}}, {{r[0-9]+}}
define agx_kernel void @and_or_flag(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %a = load i32, ptr addrspace(1) %pa, align 4
  %qa = getelementptr i32, ptr addrspace(1) %q, i32 %t
  %b = load i32, ptr addrspace(1) %qa, align 4
  %fa = bitcast i32 %a to float
  %fb = bitcast i32 %b to float
  %lt = icmp slt i32 %a, %b
  %eq = fcmp oeq float %fa, %fb
  %gt = fcmp olt float %fa, %fb
  %tie = and i1 %eq, %lt
  %take = or i1 %gt, %tie
  %v = select i1 %take, i32 %a, i32 %b
  %i = select i1 %take, i32 %b, i32 %a
  store i32 %v, ptr addrspace(1) %pa, align 4
  store i32 %i, ptr addrspace(1) %qa, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
