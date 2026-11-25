; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-far-offset-indices=0 < %s | FileCheck %s --check-prefix=OFF

; Rows of a 2-D buffer at a thread's column, 65536 elements apart: past the
; 16-bit immediate index, so the base, uniform, and a 32-bit index register
; (t + 65536 k, which the thread bound keeps from wrapping) per row instead of
; a 64-bit address and a 64-bit add of each row's offset (L15-E291; alive2:
; tests/alive-tv/agx/far-offset-index).

; CHECK-LABEL: rows:
; CHECK-NOT: iadd {{r[0-9]+}}_{{r[0-9]+}}
; CHECK: device_store 0, i32, x, {{r[0-9]+}}, u2_u3, {{r[0-9]+}}, unsigned
; CHECK: device_store 0, i32, x, {{r[0-9]+}}, u2_u3, {{r[0-9]+}}, unsigned
; CHECK: device_store 0, i32, x, {{r[0-9]+}}, u2_u3, {{r[0-9]+}}, unsigned
; OFF-LABEL: rows:
; OFF: iadd {{r[0-9]+}}_{{r[0-9]+}}
define agx_kernel void @rows(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %t to i64
  %pi = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %in, i64 %z
  %v = load i32, ptr addrspace(1) %pi
  %p0 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %out, i64 %z
  store i32 %v, ptr addrspace(1) %p0
  %a1 = add i32 %t, 65536
  %z1 = zext i32 %a1 to i64
  %p1 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %out, i64 %z1
  %v1 = add i32 %v, 1
  store i32 %v1, ptr addrspace(1) %p1
  %a2 = add i32 %t, 131072
  %z2 = zext i32 %a2 to i64
  %p2 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %out, i64 %z2
  %v2 = add i32 %v, 2
  store i32 %v2, ptr addrspace(1) %p2
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
attributes #0 = { "agx-threads-per-grid"="65536,1,1" "agx-threads-per-threadgroup"="256,1,1" }
