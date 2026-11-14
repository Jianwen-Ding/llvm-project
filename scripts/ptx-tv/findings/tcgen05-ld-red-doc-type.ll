; Reproducer for ptx-tv finding 7 (documentation), second part:
; NVPTXUsage.md documents
;   declare <n x i32> @llvm.nvvm.tcgen05.ld.red.32x32b.<num>.i32(ptr addrspace(6) %tmem_addr, i32 %redOp)
;   declare <n x i32> @llvm.nvvm.tcgen05.ld.red.32x32b.<num>.f32(ptr addrspace(6) %tmem_addr, i32 %redOp, i1 %abs, i1 %nan)
; but IntrinsicsNVVM.td (NVVM_TCGEN05_LD_RED) returns {<n x T>, T} (the
; loaded values and the reduction result), with T = float for .f32.  The
; defined form below is accepted; the documented one is rejected.
; RUN: llvm-as < %s -o /dev/null
; RUN: sed -e 's/{<2 x i32>, i32}/<2 x i32>/g' %s | not llvm-as -o /dev/null 2>&1 | FileCheck %s --check-prefix=DOC
; DOC: intrinsic return type expected literal non-packed struct

define {<2 x i32>, i32} @red(ptr addrspace(6) %t) {
  %r = call {<2 x i32>, i32} @llvm.nvvm.tcgen05.ld.red.32x32b.x2.i32(ptr addrspace(6) %t, i32 0)
  ret {<2 x i32>, i32} %r
}
declare {<2 x i32>, i32} @llvm.nvvm.tcgen05.ld.red.32x32b.x2.i32(ptr addrspace(6), i32)
