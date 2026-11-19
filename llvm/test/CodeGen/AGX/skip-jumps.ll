; RUN: llc -mtriple=agx -agx-fold-condition-chains=0 -agx-preisel-simplifycfg=0 -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -agx-fold-condition-chains=0 -agx-preisel-simplifycfg=0 -mcpu=agx-g14c -agx-max-unskipped-instrs=0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ALWAYS
; RUN: llc -mtriple=agx -agx-fold-condition-chains=0 -agx-preisel-simplifycfg=0 -mcpu=agx-g14c -agx-unskipped-loads=0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NOLOADS

; A short ALU-only body runs with an empty execution mask instead of being
; jumped over (B31), and so does one with a plain load (L15-E432); a body
; with a store, or any body with the threshold at 0, is jumped over.

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: {{^}}alu:
; CHECK:         if_icmp
; CHECK-NOT:     jmp_exec_none
; CHECK:         pop_exec
; ALWAYS-LABEL: {{^}}alu:
; ALWAYS:        if_icmp
; ALWAYS-NEXT:   jmp_exec_none
define agx_kernel void @alu(ptr addrspace(1) %out, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %t, %n
  br i1 %c, label %then, label %join
then:
  %x = mul i32 %t, %t
  br label %join
join:
  %r = phi i32 [ %x, %then ], [ %t, %entry ]
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: {{^}}memory:
; CHECK:         if_icmp
; CHECK-NEXT:    jmp_exec_none
define agx_kernel void @memory(ptr addrspace(1) %out, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %t, %n
  br i1 %c, label %then, label %join
then:
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %t, ptr addrspace(1) %p
  br label %join
join:
  ret void
}

; CHECK-LABEL: {{^}}guarded_load:
; CHECK:         if_icmp
; CHECK-NOT:     jmp_exec_none
; CHECK:         device_load
; CHECK:         pop_exec
; NOLOADS-LABEL: {{^}}guarded_load:
; NOLOADS:       if_icmp
; NOLOADS-NEXT:  jmp_exec_none
define agx_kernel void @guarded_load(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %t, %n
  br i1 %c, label %then, label %join
then:
  %pi = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %t
  %v = load i32, ptr addrspace(1) %pi, align 4
  br label %join
join:
  %r = phi i32 [ %v, %then ], [ 7, %entry ]
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %r, ptr addrspace(1) %po, align 4
  ret void
}
