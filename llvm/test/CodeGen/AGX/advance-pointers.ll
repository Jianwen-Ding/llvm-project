; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-loop-pointers-advance-first=0 < %s | FileCheck %s --check-prefix=OFF
;
; A column's 16 loads from one pointer stepped 16 rows an iteration
; (L15-E642): the step first, each load at its offset less the step, so that
; the 64-bit add does not write the register the loads before it read.

; CHECK-LABEL: {{^}}col:
; CHECK: Loop Header
; CHECK-NEXT: iadd [[P:r[0-9]+_r[0-9]+]], [[P]], {{[ru][0-9]+}}
; CHECK: device_load 0, i32, x, {{r[0-9]+}}, [[P]], -16384, signed
; CHECK: device_load 1, i32, x, {{r[0-9]+}}, [[P]], -1024, signed
; CHECK: jmp_exec_any
; OFF-LABEL: {{^}}col:
; OFF: Loop Header
; OFF-NEXT: device_load 0, i32, x, {{r[0-9]+}}, [[P:r[0-9]+_r[0-9]+]], 0, signed
; OFF: device_load 1, i32, x, {{r[0-9]+}}, [[P]], 15360, signed
; OFF: iadd [[P]], [[P]], {{[ru][0-9]+}}

define agx_kernel void @col(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %tz = zext i32 %t to i64
  %p0 = getelementptr float, ptr addrspace(1) %x, i64 %tz
  br label %loop
loop:
  %p = phi ptr addrspace(1) [ %p0, %entry ], [ %pn, %loop ]
  %i = phi i32 [ 0, %entry ], [ %in, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s16, %loop ]
  %a0 = load float, ptr addrspace(1) %p, align 4
  %q1 = getelementptr i8, ptr addrspace(1) %p, i64 4096
  %a1 = load float, ptr addrspace(1) %q1, align 4
  %q2 = getelementptr i8, ptr addrspace(1) %p, i64 8192
  %a2 = load float, ptr addrspace(1) %q2, align 4
  %q3 = getelementptr i8, ptr addrspace(1) %p, i64 12288
  %a3 = load float, ptr addrspace(1) %q3, align 4
  %q4 = getelementptr i8, ptr addrspace(1) %p, i64 16384
  %a4 = load float, ptr addrspace(1) %q4, align 4
  %q5 = getelementptr i8, ptr addrspace(1) %p, i64 20480
  %a5 = load float, ptr addrspace(1) %q5, align 4
  %q6 = getelementptr i8, ptr addrspace(1) %p, i64 24576
  %a6 = load float, ptr addrspace(1) %q6, align 4
  %q7 = getelementptr i8, ptr addrspace(1) %p, i64 28672
  %a7 = load float, ptr addrspace(1) %q7, align 4
  %q8 = getelementptr i8, ptr addrspace(1) %p, i64 32768
  %a8 = load float, ptr addrspace(1) %q8, align 4
  %q9 = getelementptr i8, ptr addrspace(1) %p, i64 36864
  %a9 = load float, ptr addrspace(1) %q9, align 4
  %q10 = getelementptr i8, ptr addrspace(1) %p, i64 40960
  %a10 = load float, ptr addrspace(1) %q10, align 4
  %q11 = getelementptr i8, ptr addrspace(1) %p, i64 45056
  %a11 = load float, ptr addrspace(1) %q11, align 4
  %q12 = getelementptr i8, ptr addrspace(1) %p, i64 49152
  %a12 = load float, ptr addrspace(1) %q12, align 4
  %q13 = getelementptr i8, ptr addrspace(1) %p, i64 53248
  %a13 = load float, ptr addrspace(1) %q13, align 4
  %q14 = getelementptr i8, ptr addrspace(1) %p, i64 57344
  %a14 = load float, ptr addrspace(1) %q14, align 4
  %q15 = getelementptr i8, ptr addrspace(1) %p, i64 61440
  %a15 = load float, ptr addrspace(1) %q15, align 4
  %s1 = fadd float %s, %a0
  %s2 = fadd float %s1, %a1
  %s3 = fadd float %s2, %a2
  %s4 = fadd float %s3, %a3
  %s5 = fadd float %s4, %a4
  %s6 = fadd float %s5, %a5
  %s7 = fadd float %s6, %a6
  %s8 = fadd float %s7, %a7
  %s9 = fadd float %s8, %a8
  %s10 = fadd float %s9, %a9
  %s11 = fadd float %s10, %a10
  %s12 = fadd float %s11, %a11
  %s13 = fadd float %s12, %a12
  %s14 = fadd float %s13, %a13
  %s15 = fadd float %s14, %a14
  %s16 = fadd float %s15, %a15
  %pn = getelementptr i8, ptr addrspace(1) %p, i64 65536
  %in = add i32 %i, 16
  %c = icmp ult i32 %in, %n
  br i1 %c, label %loop, label %exit
exit:
  %py = getelementptr float, ptr addrspace(1) %y, i64 %tz
  store float %s16, ptr addrspace(1) %py, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
