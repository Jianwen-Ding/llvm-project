; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-loop-index-bases=0 < %s | FileCheck %s --check-prefix=OFF

; An unrolled column walk, x[i + 512 k] for k = 0..4 at i = tid + 2048 j:
; each load took a 32-bit addition for its index. With the grid known
; (tid < 512) and the trip count bounded, i + 512 k cannot wrap, so the loads
; share x + 4 i, the 512 k as immediate indices (L15-E250).

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: col:
; CHECK: Loop Header
; CHECK: iadd {{r[0-9]+}}_{{r[0-9]+}}, {{r[0-9]+}}_{{r[0-9]+}}, {{r[0-9]+}}, lsl 2
; CHECK-DAG: device_load {{[0-9]}}, i32, x, {{r[0-9]+}}, {{r[0-9]+}}_{{r[0-9]+}}, 512, signed
; CHECK-DAG: device_load {{[0-9]}}, i32, x, {{r[0-9]+}}, {{r[0-9]+}}_{{r[0-9]+}}, 1024, signed
; CHECK-DAG: device_load {{[0-9]}}, i32, x, {{r[0-9]+}}, {{r[0-9]+}}_{{r[0-9]+}}, 1536, signed
; CHECK: jmp_exec_any
; OFF-LABEL: col:
; OFF-NOT: , 512, signed
; OFF: stop
define agx_kernel void @col(ptr addrspace(1) %x, ptr addrspace(1) %o) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  br label %l
l:
  %i = phi i32 [ %t, %entry ], [ %i1, %l ]
  %j = phi i32 [ 0, %entry ], [ %j1, %l ]
  %s = phi float [ 0.0, %entry ], [ %s4, %l ]
  %z0 = zext i32 %i to i64
  %p0 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z0
  %v0 = load float, ptr addrspace(1) %p0, align 4
  %a1 = add i32 %i, 512
  %z1 = zext i32 %a1 to i64
  %p1 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z1
  %v1 = load float, ptr addrspace(1) %p1, align 4
  %a2 = add i32 %i, 1024
  %z2 = zext i32 %a2 to i64
  %p2 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z2
  %v2 = load float, ptr addrspace(1) %p2, align 4
  %a3 = add i32 %i, 1536
  %z3 = zext i32 %a3 to i64
  %p3 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z3
  %v3 = load float, ptr addrspace(1) %p3, align 4
  %a4 = add i32 %i, 2048
  %z4 = zext i32 %a4 to i64
  %p4 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z4
  %v4 = load float, ptr addrspace(1) %p4, align 4
  %s0 = fadd float %s, %v0
  %s1 = fadd float %s0, %v1
  %s2 = fadd float %s1, %v2
  %s3 = fadd float %s2, %v3
  %s4 = fadd float %s3, %v4
  %i1 = add i32 %i, 2560
  %j1 = add nuw nsw i32 %j, 1
  %c = icmp ult i32 %j1, 32
  br i1 %c, label %l, label %e
e:
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i32 %t
  store float %s4, ptr addrspace(1) %po, align 4
  ret void
}

attributes #0 = { "agx-threads-per-grid"="512,1,1" "agx-threadgroups-per-grid"="4,1,1" }
