//===- MetalLib.h - Reading the AIR bitcode of a .metallib ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The reading side of llvm/docs/AIR/AIRLibFormat.rst: the entries of a
// .metallib container and the wrapped AIR bitcode each one names. Nothing is
// copied: every entry's bitcode is a reference into the container's buffer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AIR2AGX_METALLIB_H
#define LLVM_TOOLS_LLVM_AIR2AGX_METALLIB_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <array>
#include <cstdint>

namespace llvm {
namespace metallib {

/// One function of a .metallib.
struct Entry {
  /// Its name (the NAME tag).
  StringRef Name;
  /// Its function type (TYPE: 0 vertex, 1 fragment, 2 kernel, ...).
  uint8_t Type = 0xff;
  /// The bitcode it names: wrapped (0x0B17C0DE) or raw, as LLVM's bitcode
  /// reader takes either. Several entries may name the same bitcode.
  MemoryBufferRef Bitcode;
  /// The SHA-256 the container gives for Bitcode (HASH), if it has one.
  std::array<uint8_t, 32> Hash{};
  bool HasHash = false;
  /// AIR and language versions (VERS), if given.
  uint16_t AIRMajor = 0, AIRMinor = 0, LangMajor = 0, LangMinor = 0;
};

/// Whether Buffer starts with the .metallib magic ("MTLB").
bool isMetalLib(StringRef Buffer);

/// The entries of the .metallib in Buffer, which must outlive them. Every
/// offset and size is bounds-checked; a malformed container is an error.
Expected<SmallVector<Entry, 4>> readEntries(MemoryBufferRef Buffer);

/// Whether Entry's bitcode hashes to the SHA-256 its HASH tag gives (true for
/// an entry without one).
bool hashMatches(const Entry &E);

} // namespace metallib
} // namespace llvm

#endif
