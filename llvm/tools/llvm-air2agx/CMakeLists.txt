set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  BitWriter
  CodeGen
  Core
  IRReader
  MC
  Object
  Passes
  Support
  Target
  TargetParser
  )

add_llvm_tool(llvm-air2agx
  llvm-air2agx.cpp
  MetalLib.cpp

  DEPENDS
  intrinsics_gen
  )
