; Reproducer for ptx-tv finding 7 (documentation): NVPTXUsage.md documents
;   declare void @llvm.nvvm.tcgen05.st.16x32bx2.<num>(ptr addrspace(6) %tmem_addr,
;                                                     <n x i32> %args, i64 %offset, i1 %unpack)
; but IntrinsicsNVVM.td (NVVM_TCGEN05_ST) defines the offset before the data:
;   (ptr addrspace(6), i64, <n x i32>, i1).  This file uses the defined order
; and is accepted; the documented order is rejected by the IR verifier (see
; the second RUN line, which feeds the documented form).
; RUN: llvm-as < %s -o /dev/null
; RUN: sed -e 's/i64 2, <1 x i32> %%v,/<1 x i32> %%v, i64 2,/' -e 's/i64, <1 x i32>, i1)/<1 x i32>, i64, i1)/' %s \
; RUN:   | not llvm-as -o /dev/null 2>&1 | FileCheck %s --check-prefix=DOC
; DOC: intrinsic argument 1 type expected i64

define void @st(ptr addrspace(6) %t, <1 x i32> %v) {
  call void @llvm.nvvm.tcgen05.st.16x32bx2.x1(ptr addrspace(6) %t, i64 2, <1 x i32> %v, i1 0)
  ret void
}
declare void @llvm.nvvm.tcgen05.st.16x32bx2.x1(ptr addrspace(6), i64, <1 x i32>, i1)
