; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; log's expansion scales no denormal operand by 2^23 where f32 flushes:
; the compare and the multiply of that
; path flush it anyway, and x == 0 gives -inf (bit-identical on the
; simulator's functional model over every exponent's edges, denormals and
; random words).

; CHECK-LABEL: lg:
; CHECK-NOT: mov_imm {{r[0-9]+}}, 1258291200
; CHECK: stop
define agx_kernel void @lg(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.log.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.log.f32(float)
