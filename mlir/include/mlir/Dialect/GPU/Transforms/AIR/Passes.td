//===-- Passes.td - GPU AIR pass definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_AIR_PASSES
#define MLIR_DIALECT_GPU_TRANSFORMS_AIR_PASSES

include "mlir/Pass/PassBase.td"

def GpuAIRAttachTarget: Pass<"air-attach-target", ""> {
  let summary = "Attaches an AIR target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and
    attaches an AIR target (Apple GPUs) if the module matches the name
    specified by the `module` argument. Without a chip the target is a
    `.metallib`; with an AGX chip, an AGX object for that chip.

    Example:
    ```
    // mlir-opt --air-attach-target="module=agx.* chip=agx-g14c" in.mlir
    gpu.module @agx_module [#air.target<chip = "agx-g14c">] {...}
    ```
  }];
  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"triple", "triple", "std::string",
           /*default=*/ "\"air64_v28-apple-macosx26.0.0\"",
           "AIR target triple.">,
    Option<"chip", "chip", "std::string",
           /*default=*/"\"\"",
           "Target chip: empty for a .metallib, or an AGX processor.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"\"",
           "Target features (AGX chips).">,
    Option<"optLevel", "O", "unsigned",
           /*default=*/"3",
           "Optimization level.">,
  ];
}

#endif // MLIR_DIALECT_GPU_TRANSFORMS_AIR_PASSES
