; Reproducer for ptx-tv finding 7 (documentation), fourth part:
; NVPTXUsage.md documents
;   @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.override.addr.dim.1d.i32(...)
;   @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.override.addr.dim.stride.[2-5]d.i32(...)
; but these intrinsics are not overloaded (they support only a 32-bit
; multicast mask, IntrinsicsNVVM.td) and have no .i32 suffix: the documented
; name is not an intrinsic and the backend rejects it, while the name
; without the suffix lowers to the override::global_dim form.
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_107f -mattr=+ptx94 | FileCheck %s
; RUN: sed -e 's/dim\.1d(/dim.1d.i32(/g' %s | not llc -mtriple=nvptx64 -mcpu=sm_107f -mattr=+ptx94 -o /dev/null 2>&1 | FileCheck %s --check-prefix=DOC
; CHECK: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.override::global_address.override::global_dim
; DOC: unknown intrinsic 'llvm.nvvm.cp.async.bulk.tensor.g2s.tile.override.addr.dim.1d.i32'

declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.override.addr.dim.1d(ptr addrspace(7), ptr addrspace(3), ptr, ptr addrspace(1), i16, i32, i32, i64, i1, i1, i32, i32)
define void @f(ptr addrspace(7) %d, ptr addrspace(3) %b, ptr %t, ptr addrspace(1) %o, i16 %s, i32 %x, i32 %m, i64 %c) {
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.override.addr.dim.1d(ptr addrspace(7) %d, ptr addrspace(3) %b, ptr %t, ptr addrspace(1) %o, i16 %s, i32 %x, i32 %m, i64 %c, i1 0, i1 0, i32 0, i32 0)
  ret void
}
