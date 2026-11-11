; RUN: llc -mtriple=agx -mcpu=agx-g14g -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A trap in the else of a nested if. AGXCodeGenPrepare makes its unreachable a
; branch to the single return; the pre-ISel SimplifyCFG ends the block with
; unreachable again (llvm.trap is noreturn), which left the CFG unstructured
; ("failed to annotate the control flow graph") until AGXUnifyExits runs after
; it (PTX-formalization tc-*, fabric*).

declare void @llvm.trap()

; CHECK-LABEL: nested:
; CHECK:         trap
; CHECK:         stop
define agx_kernel void @nested(i1 %a, i1 %b, ptr addrspace(1) %out) {
  br i1 %a, label %inner, label %ret

inner:
  br i1 %b, label %ok, label %fail

ok:
  store i32 1, ptr addrspace(1) %out, align 4
  br label %ret

ret:
  ret void

fail:
  tail call void @llvm.trap()
  unreachable
}
