; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 -O3 < %s | FileCheck %s --check-prefix=IEEE

; log's expansion scales a denormal operand by 2^23 only where f32 keeps
; denormals: with flushing arithmetic the compare and the multiply of that
; path flush it anyway, and x == 0 gives -inf (bit-identical on the
; simulator's functional model over every exponent's edges, denormals and
; random words).

; CHECK-LABEL: lg:
; CHECK-NOT: mov_imm {{r[0-9]+}}, 1258291200
; CHECK: stop
; IEEE-LABEL: lg:
; IEEE: mov_imm {{r[0-9]+}}, 1258291200
define agx_kernel void @lg(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.log.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.log.f32(float)
