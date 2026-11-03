; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-measured-call-frames -agx-measured-stack-bytes=4096 < %s | FileCheck %s
; RUN: not --crash llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-measured-call-frames -agx-measured-stack-bytes=64 < %s 2>&1 | FileCheck %s --check-prefix=SMALL
; A reservation smaller than all addressable stack: frames that fit use it, and
; frames that do not are refused, which the driver answers by compiling with all of it.
; CHECK: stack size: 4144 bytes
; SMALL: experimental AGX call frames exceed 64 bytes

target triple = "agx-unknown-unknown"

define agx_kernel void @kernel(ptr addrspace(1) %p) {
  %r = call i32 @f(i32 1)
  store volatile i32 %r, ptr addrspace(1) %p
  ret void
}

define internal i32 @f(i32 %x) #0 {
  %a = alloca [64 x i32], addrspace(5)
  %g = getelementptr [64 x i32], ptr addrspace(5) %a, i32 0, i32 %x
  store volatile i32 %x, ptr addrspace(5) %g
  %r = load volatile i32, ptr addrspace(5) %g
  ret i32 %r
}

attributes #0 = { noinline }
