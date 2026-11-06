; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -stop-after=agx-lower-generic-addresses < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-generic-device-provenance=false -stop-after=agx-lower-generic-addresses < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s -o /dev/null

; A generic pointer read from constant memory can only hold a device address:
; the host wrote it, and the host knows no tagged address. In a function that
; must otherwise test the tags of the module's private pointers, an access
; through it (after integer offsets, an element of a vector load, a loop phi)
; is a device access without a test; one through a pointer argument, which
; may be the caller's private memory, still tests.

@table = addrspace(4) externally_initialized global [4 x i64] zeroinitializer, align 16

; CHECK-LABEL: define internal void @use(
; CHECK: %[[A:.*]] = inttoptr i64 %{{.*}} to ptr addrspace(1)
; CHECK-NEXT: load i32, ptr addrspace(1) %[[A]]
; CHECK: loop:
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK-NOT: lshr
; The argument only ever receives the kernel's private pointer: a stack store.
; CHECK: store i32 %{{.*}}, ptr addrspace(5)
; A 64-bit offset read from constant memory added to a pointer argument (PTX
; keeps pointers in integers: either could be the pointer) still tests.
; CHECK: lshr i32 %{{.*}}, 30
; CHECK: store i32 %{{.*}}, ptr addrspace(5)
; OFF-LABEL: define internal void @use(
; OFF: lshr i32 %{{.*}}, 30
; OFF: lshr i32 %{{.*}}, 30

define agx_kernel void @k(ptr addrspace(1) %out, i32 %i) {
  %buf = alloca i32, align 4, addrspace(5)
  %g = addrspacecast ptr addrspace(5) %buf to ptr
  call void @use(ptr %g, i32 %i)
  %r = load i32, ptr addrspace(5) %buf
  store i32 %r, ptr addrspace(1) %out
  ret void
}

define internal void @use(ptr %p, i32 %i) noinline {
entry:
  %pair = load <2 x i64>, ptr addrspace(4) @table, align 16
  %base = extractelement <2 x i64> %pair, i64 0
  %ext = sext i32 %i to i64
  %off = shl nsw i64 %ext, 2
  %addr = add i64 %base, %off
  %q = inttoptr i64 %addr to ptr
  %v = load i32, ptr %q, align 4
  %nodes = extractelement <2 x i64> %pair, i64 1
  br label %loop
loop:
  %n = phi i64 [ %nodes, %entry ], [ %next, %loop ]
  %np = inttoptr i64 %n to ptr
  %f = load <4 x float>, ptr %np, align 16
  %next = add i64 %n, 16
  %c = fcmp olt float 0.0, 1.0
  br i1 %c, label %loop, label %exit
exit:
  store i32 %v, ptr %p, align 4
  %poff = load i64, ptr addrspace(4) getelementptr (i8, ptr addrspace(4) @table, i64 16), align 8
  %pi = ptrtoint ptr %p to i64
  %pa = add i64 %pi, %poff
  %pp = inttoptr i64 %pa to ptr
  store i32 %v, ptr %pp, align 4
  ret void
}
