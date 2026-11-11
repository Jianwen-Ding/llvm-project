add_llvm_component_group(AGX)

set(LLVM_TARGET_DEFINITIONS AGX.td)

llvm_tablegen(AGXGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(AGXGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(AGXGenCallingConv.inc -gen-callingconv)
llvm_tablegen(AGXGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(AGXGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(AGXGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(AGXGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(AGXGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(AGXGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(AGXGenSearchableTables.inc -gen-searchable-tables)
llvm_tablegen(AGXGenSubtargetInfo.inc -gen-subtarget)

# The AIR target's mapping between LLVM intrinsics and AIR functions, which
# agx-lower-air inverts. Generated here from the same file, so that the AGX
# target does not depend on the AIR target being built.
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../AIR/AIRIntrinsicMappings.td)
llvm_tablegen(AGXGenAIRFunctions.inc -gen-searchable-tables)
set(LLVM_TARGET_DEFINITIONS AGX.td)

add_public_tablegen_target(AGXCommonTableGen)

add_llvm_target(AGXCodeGen
  AGXAnnotateControlFlow.cpp
  AGXAsmPrinter.cpp
  AGXCodeGenPrepare.cpp
  AGXMathExpansion.cpp
  AGXMemCluster.cpp
  AGXUniformConstants.cpp
  AGXLowerRecursion.cpp
  AGXFoldOperands.cpp
  AGXHoistReloads.cpp
  AGXLoadLanesLive.cpp
  AGXHoistFromElseFlow.cpp
  AGXSinkRemat.cpp
  AGXFrameLowering.cpp
  AGXHoistGuardedLoads.cpp
  AGXNarrowInductions.cpp
  AGXSpecialRegisterRanges.cpp
  AGXVectorAccesses.cpp
  AGXISelDAGToDAG.cpp
  AGXISelLowering.cpp
  AGXInsertWaits.cpp
  AGXInstrInfo.cpp
  AGXLowerControlFlow.cpp
  AGXCheckedAccesses.cpp
  AGXLowerAIR.cpp
  AGXAliasAnalysis.cpp
  AGXLowerGenericAddresses.cpp
  AGXMachineFunctionInfo.cpp
  AGXMCInstLower.cpp
  AGXPromoteKernelArgs.cpp
  AGXRegisterInfo.cpp
  AGXSelectionDAGInfo.cpp
  AGXExpandWideAtomics.cpp
  AGXSinkAddresses.cpp
  AGXThreadgroupAddresses.cpp
  AGXSoftF64.cpp
  AGXStrictFP.cpp
  AGXSubtarget.cpp
  AGXTargetMachine.cpp
  AGXTargetTransformInfo.cpp

  LINK_COMPONENTS
  AGXDesc
  AGXInfo
  AGXUtils
  Analysis
  AsmParser
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IPO
  InstCombine
  Linker
  MC
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  AGX
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
