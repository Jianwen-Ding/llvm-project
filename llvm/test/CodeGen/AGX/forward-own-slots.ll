; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-forward-own-slots=0 < %s | FileCheck %s --check-prefix=OFF

; A tree reduction's first step: sh[t] = v; barrier; if (t < 128)
; sh[t] = sh[t] + sh[t + 128]. Every write to sh is at the thread's own slot
; (t unique: a one-row threadgroup), so no other thread writes it and the
; load of sh[t] after the barrier is v (L15-E246). Without the shape (t may
; repeat in other rows), or with a store to another slot, it stays.

@sh = internal addrspace(3) global [256 x float] undef
@sh2 = internal addrspace(3) global [256 x float] undef
@sh3 = internal addrspace(3) global [256 x float] undef
@sh4 = internal addrspace(3) global [256 x float] undef

declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare void @llvm.agx.threadgroup.barrier()

; CHECK-LABEL: own:
; CHECK: threadgroup_barrier
; CHECK-NOT: threadgroup_load i32, x, {{r[0-9]+}}, [[A:r[0-9]+l]], 0
; CHECK: threadgroup_load i32, x, {{r[0-9]+}}, {{r[0-9]+l}}, 32
; CHECK-NOT: threadgroup_load i32, x, {{r[0-9]+}}, {{r[0-9]+l}}, 0
; CHECK: threadgroup_load i32, x, {{r[0-9]+}}, 0, 0
; CHECK: stop
; OFF-LABEL: own:
; OFF: threadgroup_load i32, x, {{r[0-9]+}}, {{r[0-9]+l}}, 0
define agx_kernel void @own(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pi = getelementptr float, ptr addrspace(1) %in, i32 %t
  %v = load float, ptr addrspace(1) %pi, align 4
  %ps = getelementptr [256 x float], ptr addrspace(3) @sh, i32 0, i32 %t
  store float %v, ptr addrspace(3) %ps, align 4
  call void @llvm.agx.threadgroup.barrier()
  %lt = icmp ult i32 %t, 128
  br i1 %lt, label %step, label %done
step:
  %x = load float, ptr addrspace(3) %ps, align 4
  %t2 = add i32 %t, 128
  %p2 = getelementptr [256 x float], ptr addrspace(3) @sh, i32 0, i32 %t2
  %y = load float, ptr addrspace(3) %p2, align 4
  %s = fadd float %x, %y
  store float %s, ptr addrspace(3) %ps, align 4
  br label %done
done:
  call void @llvm.agx.threadgroup.barrier()
  %z = icmp eq i32 %t, 0
  br i1 %z, label %w, label %end
w:
  %r = load float, ptr addrspace(3) @sh, align 4
  store float %r, ptr addrspace(1) %out, align 4
  br label %end
end:
  ret void
}

; No known shape: two rows may share t.
; CHECK-LABEL: noshape:
; CHECK: threadgroup_barrier
; CHECK: threadgroup_load i32, x, {{r[0-9]+}}, {{r[0-9]+l}}, 0
define agx_kernel void @noshape(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pi = getelementptr float, ptr addrspace(1) %in, i32 %t
  %v = load float, ptr addrspace(1) %pi, align 4
  %ps = getelementptr [256 x float], ptr addrspace(3) @sh2, i32 0, i32 %t
  store float %v, ptr addrspace(3) %ps, align 4
  call void @llvm.agx.threadgroup.barrier()
  %x = load float, ptr addrspace(3) %ps, align 4
  %f = fmul float %x, 2.0
  store float %f, ptr addrspace(1) %pi, align 4
  ret void
}

; A 16 x 16 threadgroup: t is not unique either.
; CHECK-LABEL: rows:
; CHECK: threadgroup_barrier
; CHECK: threadgroup_load i32, x, {{r[0-9]+}}, {{r[0-9]+l}}, 0
define agx_kernel void @rows(ptr addrspace(1) %in, ptr addrspace(1) %out) #1 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pi = getelementptr float, ptr addrspace(1) %in, i32 %t
  %v = load float, ptr addrspace(1) %pi, align 4
  %ps = getelementptr [256 x float], ptr addrspace(3) @sh3, i32 0, i32 %t
  store float %v, ptr addrspace(3) %ps, align 4
  call void @llvm.agx.threadgroup.barrier()
  %x = load float, ptr addrspace(3) %ps, align 4
  %f = fmul float %x, 2.0
  store float %f, ptr addrspace(1) %pi, align 4
  ret void
}

; A store to the next thread's slot too: not only the own slot is written.
; CHECK-LABEL: other:
; CHECK: threadgroup_barrier
; CHECK: threadgroup_load i32, x, {{r[0-9]+}}, {{r[0-9]+l}}, 0
define agx_kernel void @other(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pi = getelementptr float, ptr addrspace(1) %in, i32 %t
  %v = load float, ptr addrspace(1) %pi, align 4
  %ps = getelementptr [256 x float], ptr addrspace(3) @sh4, i32 0, i32 %t
  store float %v, ptr addrspace(3) %ps, align 4
  %t1 = add i32 %t, 1
  %pn = getelementptr [256 x float], ptr addrspace(3) @sh4, i32 0, i32 %t1
  store float %v, ptr addrspace(3) %pn, align 4
  call void @llvm.agx.threadgroup.barrier()
  %x = load float, ptr addrspace(3) %ps, align 4
  %f = fmul float %x, 2.0
  store float %f, ptr addrspace(1) %pi, align 4
  ret void
}

attributes #0 = { "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" }
attributes #1 = { "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="16,16,1" }
