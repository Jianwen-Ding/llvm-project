set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GPUAIR)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GPUAIR)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GPUAIR)
add_mlir_dialect_tablegen_target(MLIRGPUAIRPassIncGen)

add_mlir_doc(Passes GPUAIRPasses ./ -gen-pass-doc)
