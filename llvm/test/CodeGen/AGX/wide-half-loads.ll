; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-wide-half-loads=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-wide-half-loads-divergent=0 < %s | FileCheck %s --check-prefix=ALL
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-wide-byte-vectors=0 < %s | FileCheck %s --check-prefix=NOBYTES

; A 4-byte aligned <8 x half> of a thread's own row, in a block that stores
; nothing to device memory (a row sum), is one 16-byte load (L15-E585), not
; an 8-byte one of the i16 format and a second through a pointer of its own.
; CHECK-LABEL: rowsum:
; CHECK: device_load 0, i32, xyzw
; CHECK-NOT: device_load
; CHECK: device_store
; OFF-LABEL: rowsum:
; OFF: device_load 0, i16, xyzw
; OFF: device_load 1, i32, xy,
; OFF: device_store

; A vector every lane reads alike (a matrix-vector product's x) in the
; block: both loads as two 8-byte halves, unless all are asked for.
; CHECK-LABEL: rowdot:
; CHECK-NOT: i32, xyzw
; CHECK: device_store
; ALL-LABEL: rowdot:
; ALL: i32, xyzw
; ALL: i32, xyzw

; A row of halves stored as halves (L15-E590): one 16-byte load, and one
; 16-byte store of words (L15-E591).
; CHECK-LABEL: hcopy:
; CHECK: device_load 0, i32, xyzw
; CHECK-NOT: device_load
; CHECK: device_store 0, i32, xyzw
; CHECK-NOT: device_store
; CHECK: stop
; OFF-LABEL: hcopy:
; OFF: device_store 0, i32, xyzw

; A thread's row of 16 bytes likewise (L15-E596): one 16-byte access of
; words, not four of the byte format.
; CHECK-LABEL: bytesum:
; CHECK: device_load 0, i32, xyzw
; CHECK-NOT: device_load
; CHECK: device_store
; NOBYTES-LABEL: bytesum:
; NOBYTES: i8, xyzw

; A block that stores floats: two 8-byte halves.
; CHECK-LABEL: widen:
; CHECK-NOT: device_load {{.*}}, i32, xyzw
; CHECK: stop

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @rowsum(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %a = phi float [ 0.0, %entry ], [ %a1, %loop ]
  %k = add i32 %row, %j
  %z = zext i32 %k to i64
  %p = getelementptr half, ptr addrspace(1) %x, i64 %z
  %v = load <8 x half>, ptr addrspace(1) %p, align 4
  %e = fpext <8 x half> %v to <8 x float>
  %a1 = call float @llvm.vector.reduce.fadd.v8f32(float %a, <8 x float> %e)
  %j1 = add i32 %j, 8
  %c = icmp ult i32 %j1, %n
  br i1 %c, label %loop, label %out
out:
  %tz = zext i32 %t to i64
  %q = getelementptr float, ptr addrspace(1) %y, i64 %tz
  store float %a1, ptr addrspace(1) %q, align 4
  ret void
}

define agx_kernel void @rowdot(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %a = phi float [ 0.0, %entry ], [ %a1, %loop ]
  %k = add i32 %row, %j
  %z = zext i32 %k to i64
  %p = getelementptr half, ptr addrspace(1) %x, i64 %z
  %v = load <8 x half>, ptr addrspace(1) %p, align 4
  %jz = zext i32 %j to i64
  %pw = getelementptr half, ptr addrspace(1) %w, i64 %jz
  %u = load <8 x half>, ptr addrspace(1) %pw, align 4
  %m = fmul <8 x half> %v, %u
  %e = fpext <8 x half> %m to <8 x float>
  %a1 = call float @llvm.vector.reduce.fadd.v8f32(float %a, <8 x float> %e)
  %j1 = add i32 %j, 8
  %c = icmp ult i32 %j1, %n
  br i1 %c, label %loop, label %out
out:
  %tz = zext i32 %t to i64
  %q = getelementptr float, ptr addrspace(1) %y, i64 %tz
  store float %a1, ptr addrspace(1) %q, align 4
  ret void
}

define agx_kernel void @hcopy(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %k = add i32 %row, %j
  %z = zext i32 %k to i64
  %p = getelementptr half, ptr addrspace(1) %x, i64 %z
  %v = load <8 x half>, ptr addrspace(1) %p, align 4
  %m = fmul <8 x half> %v, %v
  %q = getelementptr half, ptr addrspace(1) %y, i64 %z
  store <8 x half> %m, ptr addrspace(1) %q, align 4
  %j1 = add i32 %j, 8
  %c = icmp ult i32 %j1, %n
  br i1 %c, label %loop, label %out
out:
  ret void
}

define agx_kernel void @bytesum(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %a = phi i32 [ 0, %entry ], [ %a1, %loop ]
  %k = add i32 %row, %j
  %z = zext i32 %k to i64
  %p = getelementptr i8, ptr addrspace(1) %x, i64 %z
  %v = load <16 x i8>, ptr addrspace(1) %p, align 4
  %e = sext <16 x i8> %v to <16 x i32>
  %s = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %e)
  %a1 = add i32 %a, %s
  %j1 = add i32 %j, 16
  %c = icmp ult i32 %j1, %n
  br i1 %c, label %loop, label %out
out:
  %tz = zext i32 %t to i64
  %q = getelementptr i32, ptr addrspace(1) %y, i64 %tz
  store i32 %a1, ptr addrspace(1) %q, align 4
  ret void
}

define agx_kernel void @widen(ptr addrspace(1) %x, ptr addrspace(1) %y) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 3
  %z = zext i32 %row to i64
  %p = getelementptr half, ptr addrspace(1) %x, i64 %z
  %v = load <8 x half>, ptr addrspace(1) %p, align 4
  %e = fpext <8 x half> %v to <8 x float>
  %q = getelementptr float, ptr addrspace(1) %y, i64 %z
  store <8 x float> %e, ptr addrspace(1) %q, align 4
  ret void
}

declare float @llvm.vector.reduce.fadd.v8f32(float, <8 x float>)
declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)
