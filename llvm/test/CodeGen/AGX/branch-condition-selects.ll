; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-branch-condition-selects=0 < %s | FileCheck %s --check-prefix=OFF

; A 2D bounds check, x >= w || y >= h: (x >= w ? h : y) >= h, one select
; whose compare is x >= w and the branch's compare, not a select of each
; compare and their or (L15-E738).
; CHECK-LABEL: bounds:
; CHECK: icmpsel {{.*}}
; CHECK-NEXT: if_icmp
; OFF-LABEL: bounds:
; OFF: icmpsel
; OFF: icmpsel
; OFF: if_icmp
define agx_kernel void @bounds(ptr addrspace(1) %o, i32 %w, i32 %h) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %y = call i32 @llvm.agx.thread.position.in.grid.y()
  %ox = icmp sge i32 %x, %w
  %oy = icmp sge i32 %y, %h
  %out = or i1 %ox, %oy
  br i1 %out, label %done, label %body
body:
  %i = mul i32 %y, %w
  %k = add i32 %i, %x
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %o, i64 %z
  store float 1.0, ptr addrspace(1) %p, align 4
  br label %done
done:
  ret void
}

; Both in range, x < w && y < h: (x < w ? y : h) < h.
; CHECK-LABEL: inside:
; CHECK: icmpsel {{.*}}
; CHECK-NEXT: if_icmp
define agx_kernel void @inside(ptr addrspace(1) %o, i32 %w, i32 %h) {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %y = call i32 @llvm.agx.thread.position.in.grid.y()
  %ix = icmp slt i32 %x, %w
  %iy = icmp slt i32 %y, %h
  %in = and i1 %ix, %iy
  br i1 %in, label %body, label %done
body:
  %i = mul i32 %y, %w
  %k = add i32 %i, %x
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %o, i64 %z
  store float 1.0, ptr addrspace(1) %p, align 4
  br label %done
done:
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.grid.y()
