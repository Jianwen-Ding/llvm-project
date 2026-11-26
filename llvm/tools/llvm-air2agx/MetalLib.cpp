//===- MetalLib.cpp - Reading the AIR bitcode of a .metallib --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The layout (llvm/docs/AIR/AIRLibFormat.rst, the writer in
// llvm/lib/Target/AIR/AIRWriter/AIRLibWriter.cpp): an 88-byte "MTLB" header
// with the offsets and sizes of four sections; section 0 a u32 entry count and
// one block per entry (a u32 block size counting itself, then tags of a
// 4-letter name, a u16 length and the payload, up to "ENDT"); section 3 the
// wrapped bitcode. An entry's bitcode is MDSZ bytes at OFFT[2] in section 3.
//
//===----------------------------------------------------------------------===//

#include "MetalLib.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SHA256.h"

using namespace llvm;
using namespace llvm::metallib;
using namespace llvm::support;

static constexpr size_t HeaderSize = 88;

bool metallib::isMetalLib(StringRef Buffer) {
  return Buffer.size() >= 4 && Buffer.starts_with("MTLB");
}

static Error malformed(const Twine &Msg) {
  return createStringError(inconvertibleErrorCode(),
                           "malformed .metallib: " + Msg);
}

namespace {
/// A bounds-checked little-endian cursor over the container.
class Reader {
  StringRef Data;

public:
  explicit Reader(StringRef Data) : Data(Data) {}
  bool has(uint64_t Off, uint64_t Size) const {
    return Off <= Data.size() && Size <= Data.size() - Off;
  }
  uint16_t u16(uint64_t Off) const {
    return endian::read16le(Data.data() + Off);
  }
  uint32_t u32(uint64_t Off) const {
    return endian::read32le(Data.data() + Off);
  }
  uint64_t u64(uint64_t Off) const {
    return endian::read64le(Data.data() + Off);
  }
  StringRef bytes(uint64_t Off, uint64_t Size) const {
    return Data.substr(Off, Size);
  }
};
} // namespace

Expected<SmallVector<Entry, 4>>
metallib::readEntries(MemoryBufferRef Buffer) {
  StringRef Data = Buffer.getBuffer();
  Reader R(Data);
  if (!isMetalLib(Data))
    return malformed("no MTLB magic");
  if (!R.has(0, HeaderSize))
    return malformed("truncated header");

  struct Section {
    uint64_t Off, Size;
  } Sec[4];
  for (unsigned I = 0; I < 4; ++I) {
    Sec[I] = {R.u64(28 + 16 * I), R.u64(36 + 16 * I)};
    if (!R.has(Sec[I].Off, Sec[I].Size))
      return malformed("section " + Twine(I) + " lies outside the file");
  }

  uint64_t P = Sec[0].Off;
  if (!R.has(P, 4))
    return malformed("no entry count");
  uint32_t Count = R.u32(P);
  P += 4;
  // A block holds at least its size, a NAME and an ENDT.
  if (Count > Data.size() / 16)
    return malformed("entry count " + Twine(Count) + " is too large");

  SmallVector<Entry, 4> Entries;
  for (uint32_t I = 0; I < Count; ++I) {
    if (!R.has(P, 4))
      return malformed("entry " + Twine(I) + " is truncated");
    uint32_t BlockSize = R.u32(P);
    if (BlockSize < 4 || !R.has(P, BlockSize))
      return malformed("entry " + Twine(I) + " has a bad size");
    uint64_t End = P + BlockSize, T = P + 4;
    Entry E;
    bool HasOffsets = false, HasSize = false;
    uint64_t BitcodeOff = 0, BitcodeSize = 0;
    while (T + 4 <= End) {
      StringRef Tag = R.bytes(T, 4);
      if (Tag == "ENDT")
        break;
      if (!R.has(T + 4, 2))
        return malformed("tag " + Tag + " is truncated");
      uint16_t Len = R.u16(T + 4);
      uint64_t Payload = T + 6;
      if (Payload + Len > End)
        return malformed("tag " + Tag + " runs past its entry");
      if (Tag == "NAME") {
        E.Name = R.bytes(Payload, Len);
        // NUL-terminated.
        E.Name = E.Name.take_until([](char C) { return C == '\0'; });
      } else if (Tag == "TYPE" && Len >= 1) {
        E.Type = Data[Payload];
      } else if (Tag == "HASH" && Len == 32) {
        memcpy(E.Hash.data(), Data.data() + Payload, 32);
        E.HasHash = true;
      } else if (Tag == "MDSZ" && Len >= 8) {
        BitcodeSize = R.u64(Payload);
        HasSize = true;
      } else if (Tag == "OFFT" && Len >= 24) {
        BitcodeOff = R.u64(Payload + 16);
        HasOffsets = true;
      } else if (Tag == "VERS" && Len >= 8) {
        E.AIRMajor = R.u16(Payload);
        E.AIRMinor = R.u16(Payload + 2);
        E.LangMajor = R.u16(Payload + 4);
        E.LangMinor = R.u16(Payload + 6);
      }
      T = Payload + Len;
    }
    // Without its offsets an entry's bitcode is the whole of section 3.
    if (!HasOffsets)
      BitcodeOff = 0;
    if (!HasSize)
      BitcodeSize = Sec[3].Size - std::min(BitcodeOff, Sec[3].Size);
    if (BitcodeOff > Sec[3].Size || BitcodeSize > Sec[3].Size - BitcodeOff ||
        BitcodeSize == 0)
      return malformed("the bitcode of entry " + Twine(I) + " (" + E.Name +
                       ") lies outside section 3");
    E.Bitcode = MemoryBufferRef(R.bytes(Sec[3].Off + BitcodeOff, BitcodeSize),
                                Buffer.getBufferIdentifier());
    Entries.push_back(E);
    P = End;
  }
  return Entries;
}

bool metallib::hashMatches(const Entry &E) {
  if (!E.HasHash)
    return true;
  StringRef B = E.Bitcode.getBuffer();
  std::array<uint8_t, 32> H = SHA256::hash(
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(B.data()), B.size()));
  return H == E.Hash;
}
