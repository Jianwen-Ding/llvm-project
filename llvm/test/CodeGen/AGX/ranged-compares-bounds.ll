; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-ranged-compares-bounds=0 < %s | FileCheck %s --check-prefix=OFF
;
; A padded stencil's centre tap: (x | y) <u 1024 with x, y below MLIR's
; upper bounds of 1024 (agx-upper-bounds) holds: the or's known bits have
; nothing at or above bit 10. Its guard goes; the others' (x + 1, the
; or of x - 1) stay (L15-E491).

; CHECK-LABEL: @centre(
; CHECK: br i1 true, label %load, label %next
; CHECK: br i1 %in2, label %load2, label %done
; CHECK: ret void
; OFF-LABEL: @centre(
; OFF: br i1 %in, label %load, label %next
define agx_kernel void @centre(ptr addrspace(1) %p, ptr addrspace(1) %o) #0 {
entry:
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %y = call i32 @llvm.agx.thread.position.in.grid.y()
  %c = or i32 %x, %y
  %in = icmp ult i32 %c, 1024
  br i1 %in, label %load, label %next

load:
  %row = shl i32 %y, 10
  %i = add i32 %row, %x
  %g = getelementptr float, ptr addrspace(1) %p, i32 %i
  %v = load float, ptr addrspace(1) %g, align 4
  br label %next

next:
  %a = phi float [ %v, %load ], [ 0.0, %entry ]
  %x1 = add i32 %x, 1
  %r = or i32 %x1, %y
  %in2 = icmp ult i32 %r, 1024
  br i1 %in2, label %load2, label %done

load2:
  %row2 = shl i32 %y, 10
  %i2 = add i32 %row2, %x1
  %g2 = getelementptr float, ptr addrspace(1) %p, i32 %i2
  %v2 = load float, ptr addrspace(1) %g2, align 4
  br label %done

done:
  %b = phi float [ %v2, %load2 ], [ 0.0, %next ]
  %s = fadd float %a, %b
  %io = add i32 %x, %y
  %go = getelementptr float, ptr addrspace(1) %o, i32 %io
  store float %s, ptr addrspace(1) %go, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.grid.y()

attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="1024,1024,0" }
