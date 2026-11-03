; agx-lower-air compiles air-emulate-f64's compare-exchange loop for a double
; precision atomic addition back into the atomicrmw it stands for.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -S %t.air.ll | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -agx-air-native-f64-arith=0 -S %t.air.ll \
; RUN:   | FileCheck %s --check-prefix=OFF

; CHECK-LABEL: define agx_kernel void @k(
; CHECK:         atomicrmw fadd ptr addrspace(1) %p, double %{{.*}} syncscope("agent-one-as") monotonic, align 8
; CHECK-NOT:     cmpxchg
; CHECK:         ret void
; OFF:           cmpxchg
define void @k(ptr addrspace(1) %p, i64 %x, ptr addrspace(1) %out) "air-kernel" {
entry:
  %init = load volatile i64, ptr addrspace(1) %p, align 8
  br label %loop
loop:
  %old = phi i64 [ %init, %entry ], [ %loaded, %loop ]
  %new = call i64 @__air_f64_add(i64 %old, i64 %x)
  %cx = cmpxchg ptr addrspace(1) %p, i64 %old, i64 %new syncscope("agent-one-as") monotonic monotonic, align 8
  %loaded = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %done, label %loop
done:
  store i64 %loaded, ptr addrspace(1) %out, align 8
  ret void
}

define internal i64 @__air_f64_add(i64 %a, i64 %b) { ret i64 %a }
