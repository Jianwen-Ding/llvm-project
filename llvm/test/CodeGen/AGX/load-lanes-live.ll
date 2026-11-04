; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-keep-load-lanes=false < %s | FileCheck %s --check-prefix=OFF

; A load writes its destination when the memory answers. With subregister
; liveness, the unused high word of the first load's destination would be
; free for the second load, which would then wait for the first: every lane
; of a load's destination stays allocated until its first use.

; CHECK-LABEL: k:
; CHECK: device_load 0, i32, xy, [[P:r[0-9]+]]_r{{[0-9]+}},
; CHECK-NEXT: device_load {{[01]}}, i16, x,
; CHECK-NEXT: wait
; OFF-LABEL: k:
; OFF: device_load 0, i32, xy,
; OFF-NEXT: wait 0
; OFF-NEXT: device_load
define agx_kernel void @k(ptr addrspace(4) nofree readonly byref(i64) captures(none) %"182", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"183", ptr addrspace(4) nofree readnone byref(i64) captures(none) %"184", ptr addrspace(4) nofree readonly byref(i64) captures(none) %"185") #0 {
entry:
  %0 = load i64, ptr addrspace(4) %"182", align 8
  %1 = load i64, ptr addrspace(4) %"183", align 8
  %2 = load i64, ptr addrspace(4) %"185", align 8
  %e.i = tail call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %3 = zext i32 %e.i to i64
  %"202" = shl nuw nsw i64 %3, 3
  %"204" = add i64 %0, %"202"
  %"207" = add i64 %1, %"202"
  %"213" = add i64 %2, %"202"
  %"255" = inttoptr i64 %"204" to ptr addrspace(1)
  %4 = load i64, ptr addrspace(1) %"255", align 8
  %"256" = inttoptr i64 %"207" to ptr addrspace(1)
  %5 = load i64, ptr addrspace(1) %"256", align 8
  %"258" = trunc i64 %4 to i32
  %"177" = bitcast i32 %"258" to <2 x i16>
  %"240" = extractelement <2 x i16> %"177", i64 0
  %"243" = trunc i64 %5 to i16
  %"263" = mul i16 %"240", %"243"
  %"179" = insertelement <2 x i16> %"177", i16 %"263", i64 0
  %"266" = bitcast <2 x i16> %"179" to i32
  %"251" = zext i32 %"266" to i64
  %"268" = inttoptr i64 %"213" to ptr addrspace(1)
  store i64 %"251", ptr addrspace(1) %"268", align 8
  ret void
}

declare noundef i32 @llvm.agx.thread.position.in.threadgroup.x() #1

attributes #0 = { "denormal-fp-math"="preserve-sign" }
