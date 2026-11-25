// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 | FileCheck %s
// RUN: mlir-opt %s --convert-gpu-to-air="use-bare-ptr-memref-call-conv=1 erf-reciprocal-division=0" | FileCheck %s --check-prefix=OFF

// f32 erf's rational approximation P / Q divided by the hardware's
// reciprocal y0 of Q, one Newton step y = y0 + y0 (1 - Q y0), q0 = P y and
// the residual correction q0 + y (P - Q q0): the same correctly rounded
// quotient as the general division for every f32 input (L15-E499).
gpu.module @erf_division {
  // CHECK-LABEL: llvm.func @k(
  // CHECK: %[[Y0:.*]] = llvm.fdiv %[[ONE:.*]], %[[Q:.*]] fastmath<afn> : f32
  // CHECK: %[[NQ:.*]] = llvm.fneg %[[Q]] : f32
  // CHECK: %[[E:.*]] = llvm.intr.fma(%[[NQ]], %[[Y0]], %[[ONE]])
  // CHECK: %[[Y:.*]] = llvm.intr.fma(%[[E]], %[[Y0]], %[[Y0]])
  // CHECK: %[[Q0:.*]] = llvm.fmul %[[P:.*]], %[[Y]] : f32
  // CHECK: %[[R:.*]] = llvm.intr.fma(%[[NQ]], %[[Q0]], %[[P]])
  // CHECK: llvm.intr.fma(%[[R]], %[[Y]], %[[Q0]])
  // CHECK-NOT: llvm.fdiv
  // OFF-LABEL: llvm.func @k(
  // OFF: llvm.fdiv %{{.*}}, %{{.*}} : f32
  gpu.func @k(%a: memref<8xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %a[%c0] : memref<8xf32, 1>
    %u = math.erf %v : f32
    memref.store %u, %a[%c0] : memref<8xf32, 1>
    gpu.return
  }
}
