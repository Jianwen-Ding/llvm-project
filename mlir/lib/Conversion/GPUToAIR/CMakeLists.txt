add_mlir_conversion_library(MLIRGPUToAIRTransforms
  LowerGpuOpsToAIROps.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAIRDialect
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRComplexToStandard
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathTransforms
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRPass
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSCFTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  )
