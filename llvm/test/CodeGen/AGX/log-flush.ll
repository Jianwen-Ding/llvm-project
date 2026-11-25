; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; log's expansion scales no denormal operand by 2^23 where f32 flushes:
; the compare and the multiply of that
; path flush it anyway, and x == 0 gives -inf (bit-identical on the
; simulator's functional model over every exponent's edges, denormals and
; random words).

; (2^23 itself is the base of the exponent's bfeil since L15-E614: no
; multiply of x.)
; CHECK-LABEL: lg:
; CHECK-NOT: fmul32 {{.*}}u2
; CHECK: stop
define agx_kernel void @lg(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.log.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.log.f32(float)
