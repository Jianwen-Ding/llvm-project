//===-- AGXCallingConv.td - AGX calling conventions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX kernels receive their arguments in uniform registers. The layout is
// computed by AGXTargetLowering::LowerFormalArguments (natural alignment in
// 16-bit units, see AGXUsage.rst), not by a CCState, so only the callee saved
// register list is described here. The default path inlines calls; the
// experimental leaf-call path uses a small register window and no callee-saved
// registers.
//
//===----------------------------------------------------------------------===//

def CSR_AGX_NoRegs : CalleeSavedRegs<(add)>;
// Called functions keep the top of the register file (-agx-callee-saved, 48 by
// default), so that what is live across a call need not go to memory unless the
// callee uses them. Blender Cycles' Open Data scenes at 0 / 16 / 48 / 64 / 80:
// classroom 176 / 191 / 197 / 196 / 193 samples per minute.
def CSR_AGX_Top8 : CalleeSavedRegs<(sequence "R%u", 120, 127)>;
def CSR_AGX_Top16 : CalleeSavedRegs<(sequence "R%u", 112, 127)>;
def CSR_AGX_Top32 : CalleeSavedRegs<(sequence "R%u", 96, 127)>;
def CSR_AGX_Top48 : CalleeSavedRegs<(sequence "R%u", 80, 127)>;
def CSR_AGX_Top64 : CalleeSavedRegs<(sequence "R%u", 64, 127)>;
def CSR_AGX_Top80 : CalleeSavedRegs<(sequence "R%u", 48, 127)>;
