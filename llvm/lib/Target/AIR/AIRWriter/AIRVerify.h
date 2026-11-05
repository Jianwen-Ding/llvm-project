//===- AIRVerify.h - Conformance checks for AIR modules ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Whether a module is well-formed AIR, without a device: see
// llvm/docs/AIR/AIRTarget.rst, "Checking AIR".
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_AIRVERIFY_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_AIRVERIFY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include <string>

namespace llvm {
class Module;
namespace metal {

enum class AIRForm {
  /// The AIR target's portable passes' output, for any AIR consumer: it may
  /// keep what a consumer lowers itself (wide integers, hints).
  Portable,
  /// What Metal is given.
  Metal,
};

/// A description of each way M is not AIR of this form. Without FullVerifier,
/// of LLVM's verifier only what passes after llc's own run of it can break (the
/// grouping of phis), for the cost of every compilation.
SmallVector<std::string, 4> verifyAIR(const Module &M, AIRForm Form,
                                      bool FullVerifier = true);

/// Whether Bytes reads back as a module that LLVM's verifier accepts. Empty if
/// so, else what is wrong.
std::string verifyAIRBitcode(ArrayRef<uint8_t> Bytes);

} // namespace metal

/// air-verify, air-verify-metal: report the violations as errors.
class AIRVerifyPass : public RequiredPassInfoMixin<AIRVerifyPass> {
  metal::AIRForm Form;

public:
  explicit AIRVerifyPass(metal::AIRForm Form) : Form(Form) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif
