//===- llvm-air2agx.cpp - Compile AIR (.air, .metallib) to AGX ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compiles AIR, Apple's LLVM IR for its GPUs, to AGX machine code in one
// process: a .metallib container (every AIR module its entries name), a .air
// file (bitcode, bare or behind the 0x0B17C0DE wrapper) or LLVM IR text goes
// through agx-lower-air and the optimization pipeline of -O (with the AGX
// target's own passes), then the AGX code generator. The output is the AGX
// backend's ELF object (llvm/docs/AGXUsage.rst), its assembly, the optimized
// AGX IR, or each kernel's raw instructions (what the AGX Performance
// Simulator's agx_cost and agx_dispatch read).
//
// Each AIR module, for each processor of -mcpu, is one job; jobs run in
// parallel (-j), each in its own LLVMContext and TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "MetalLib.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include <chrono>
#include <mutex>

using namespace llvm;

static cl::OptionCategory Cat("llvm-air2agx options");

static cl::list<std::string> Inputs(cl::Positional, cl::OneOrMore,
                                    cl::desc("<.metallib | .air | .ll>..."),
                                    cl::cat(Cat));

static cl::opt<std::string>
    Output("o", cl::desc("Output file (one job), or the prefix of each "
                         "kernel's file with --emit=code"),
           cl::value_desc("file"), cl::cat(Cat));

static cl::opt<std::string>
    OutputDir("output-dir",
              cl::desc("Directory for outputs named after their inputs "
                       "(default: the current directory)"),
              cl::value_desc("dir"), cl::cat(Cat));

static cl::list<std::string>
    CPUs("mcpu", cl::CommaSeparated,
         cl::desc("AGX processors to compile for (comma separated; one "
                  "output each when there are several)"),
         cl::value_desc("cpu"), cl::cat(Cat));

static cl::opt<std::string> Features("mattr", cl::desc("AGX target features"),
                                     cl::value_desc("a1,+a2,-a3"),
                                     cl::cat(Cat));

static cl::opt<std::string>
    TripleName("mtriple", cl::init("agx"),
               cl::desc("AGX triple to compile for (default: agx)"),
               cl::cat(Cat));

static cl::opt<char>
    OptLevel("O", cl::Prefix, cl::init('3'),
             cl::desc("Optimization level: -O0, -O1, -O2 or -O3 (default)"),
             cl::cat(Cat));

enum class Emit { Obj, Asm, LLVM, BC, Code, AGX };
static cl::opt<Emit> EmitKind(
    "emit", cl::init(Emit::Obj), cl::desc("What to write"),
    cl::values(clEnumValN(Emit::Obj, "obj", "the AGX ELF object (default)"),
               clEnumValN(Emit::Asm, "asm", "AGX assembly"),
               clEnumValN(Emit::LLVM, "llvm",
                          "the optimized AGX IR, as text (before code "
                          "generation)"),
               clEnumValN(Emit::BC, "bc", "the optimized AGX IR, as bitcode"),
               clEnumValN(Emit::AGX, "agx",
                          "pure AGX machine code: the instructions alone, "
                          "with no container (the default for -o *.agx)"),
               clEnumValN(Emit::Code, "code",
                          "each kernel's raw instructions, one .bin per "
                          "kernel (agx_cost / agx_dispatch input)")),
    cl::cat(Cat));

static cl::opt<std::string>
    KernelName("kernel",
               cl::desc("With --emit=agx, only this kernel's instructions "
                        "(from offset 0, as agx_cost and agx_run take them)"),
               cl::value_desc("name"), cl::cat(Cat));

static cl::opt<std::string>
    Threads("j", cl::desc("Jobs to run at once (default: one per core)"),
            cl::value_desc("N"), cl::cat(Cat));

static cl::list<std::string>
    EntryFilter("entry", cl::CommaSeparated,
                cl::desc("Only the .metallib entries of these names"),
                cl::value_desc("name"), cl::cat(Cat));

static cl::opt<bool> List("list",
                          cl::desc("List each input's AIR modules and exit"),
                          cl::cat(Cat));

static cl::opt<bool> VerifyHash(
    "verify-hash", cl::init(true),
    cl::desc("Check each .metallib entry's SHA-256 against its bitcode"),
    cl::cat(Cat));

static cl::opt<bool> NoVerify("disable-verify",
                              cl::desc("Do not verify the modules"),
                              cl::cat(Cat));

static cl::opt<bool> TimeJobs("time",
                              cl::desc("Report each job's read, optimization "
                                       "and code generation times"),
                              cl::cat(Cat));

namespace {

/// One AIR module of an input.
struct Unit {
  std::string Input;
  /// The .metallib entry it comes from (empty for a .air / .ll input).
  std::string Entry;
  /// Its bitcode or text, a reference into Buffers.
  MemoryBufferRef Data;
};

struct Job {
  const Unit *U;
  std::string CPU;
  std::string OutPath;
  // Results.
  std::string Log;
  bool Failed = false;
  double ReadMs = 0, OptMs = 0, CodegenMs = 0;
  unsigned Kernels = 0;
  uint64_t CodeBytes = 0;
};

/// Diagnostics of one job, collected to be printed in job order.
struct CollectingHandler : DiagnosticHandler {
  Job &J;
  explicit CollectingHandler(Job &J) : J(J) {}
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    if (DI.getSeverity() == DS_Remark || DI.getSeverity() == DS_Note)
      return true;
    raw_string_ostream OS(J.Log);
    OS << J.U->Input;
    if (!J.U->Entry.empty())
      OS << "(" << J.U->Entry << ")";
    OS << ": "
       << (DI.getSeverity() == DS_Error ? "error: " : "warning: ");
    DiagnosticPrinterRawOStream DP(OS);
    DI.print(DP);
    OS << "\n";
    if (DI.getSeverity() == DS_Error)
      J.Failed = true;
    return true;
  }
};

} // namespace

static double msSince(std::chrono::steady_clock::time_point T) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - T)
      .count();
}

static OptimizationLevel optimizationLevel() {
  switch (OptLevel) {
  case '0':
    return OptimizationLevel::O0;
  case '1':
    return OptimizationLevel::O1;
  case '2':
    return OptimizationLevel::O2;
  default:
    return OptimizationLevel::O3;
  }
}

static CodeGenOptLevel codeGenOptLevel() {
  switch (OptLevel) {
  case '0':
    return CodeGenOptLevel::None;
  case '1':
    return CodeGenOptLevel::Less;
  case '3':
    return CodeGenOptLevel::Aggressive;
  default:
    return CodeGenOptLevel::Default;
  }
}

static StringRef extension() {
  switch (EmitKind) {
  case Emit::Obj:
    return ".o";
  case Emit::Asm:
    return ".s";
  case Emit::LLVM:
    return ".ll";
  case Emit::BC:
    return ".bc";
  case Emit::Code:
    return "";
  case Emit::AGX:
    return ".agx";
  }
  llvm_unreachable("emit kind");
}

namespace {
/// The code of an object: its .text and each kernel's range in it.
struct ObjectCode {
  std::unique_ptr<object::ObjectFile> Obj;
  StringRef Text;
  struct Kernel {
    StringRef Name;
    uint64_t Addr, Size;
  };
  SmallVector<Kernel, 8> Kernels;
};
} // namespace

/// The instructions of the object the code generator wrote: .text, which must
/// need no relocation (pure code runs where it is loaded), and each kernel's
/// symbol's bytes in it (to the next symbol where the symbol gives no size).
static Expected<ObjectCode> objectCode(StringRef Object, StringRef Path) {
  ObjectCode C;
  auto Obj = object::ObjectFile::createObjectFile(MemoryBufferRef(Object, Path));
  if (!Obj)
    return Obj.takeError();
  C.Obj = std::move(*Obj);
  std::optional<object::SectionRef> Text;
  for (const object::SectionRef &S : C.Obj->sections())
    if (S.isText()) {
      Text = S;
      break;
    }
  if (!Text)
    return createStringError(inconvertibleErrorCode(), "no .text section");
  for (const object::SectionRef &S : C.Obj->sections()) {
    Expected<object::section_iterator> Target = S.getRelocatedSection();
    if (!Target) {
      consumeError(Target.takeError());
      continue;
    }
    if (*Target == *Text && !S.relocations().empty())
      return createStringError(
          inconvertibleErrorCode(),
          "the code needs relocations, which pure AGX code cannot carry: "
          "write the ELF object");
  }
  Expected<StringRef> Contents = Text->getContents();
  if (!Contents)
    return Contents.takeError();
  C.Text = *Contents;
  for (const object::SymbolRef &S : C.Obj->symbols()) {
    Expected<object::SymbolRef::Type> Ty = S.getType();
    Expected<object::section_iterator> Sec = S.getSection();
    Expected<StringRef> Name = S.getName();
    Expected<uint64_t> Addr = S.getValue();
    if (!Ty || !Sec || !Name || !Addr) {
      consumeError(Ty.takeError());
      consumeError(Sec.takeError());
      consumeError(Name.takeError());
      consumeError(Addr.takeError());
      continue;
    }
    if (*Ty != object::SymbolRef::ST_Function || *Sec != *Text)
      continue;
    C.Kernels.push_back({*Name, *Addr, object::ELFSymbolRef(S).getSize()});
  }
  llvm::sort(C.Kernels, [](const ObjectCode::Kernel &A,
                           const ObjectCode::Kernel &B) {
    return A.Addr < B.Addr;
  });
  for (size_t I = 0; I < C.Kernels.size(); ++I) {
    ObjectCode::Kernel &K = C.Kernels[I];
    uint64_t End =
        I + 1 < C.Kernels.size() ? C.Kernels[I + 1].Addr : C.Text.size();
    if (K.Addr > End)
      return createStringError(inconvertibleErrorCode(),
                               "kernel " + K.Name + " lies outside .text");
    if (!K.Size || K.Addr + K.Size > End)
      K.Size = End - K.Addr;
  }
  return std::move(C);
}

static Error writeBytes(StringRef Path, StringRef Bytes) {
  std::error_code EC;
  ToolOutputFile Out(Path, EC, sys::fs::OF_None);
  if (EC)
    return createStringError(EC, Path + ": " + EC.message());
  Out.os() << Bytes;
  Out.keep();
  return Error::success();
}

/// --emit=code: each kernel's instructions, PREFIX.<kernel>.bin.
static Error writeKernelCode(Job &J, StringRef Object) {
  Expected<ObjectCode> C = objectCode(Object, J.OutPath);
  if (!C)
    return C.takeError();
  for (const ObjectCode::Kernel &K : C->Kernels) {
    if (Error E = writeBytes((J.OutPath + "." + K.Name + ".bin").str(),
                             C->Text.substr(K.Addr, K.Size)))
      return E;
    ++J.Kernels;
    J.CodeBytes += K.Size;
  }
  return Error::success();
}

/// --emit=agx: the module's instructions alone, each kernel at its offset in
/// .text; with --kernel, that kernel's instructions, from offset 0.
static Error writePureAGX(Job &J, StringRef Object) {
  Expected<ObjectCode> C = objectCode(Object, J.OutPath);
  if (!C)
    return C.takeError();
  StringRef Bytes = C->Text;
  J.Kernels = C->Kernels.size();
  if (!KernelName.empty()) {
    auto It = find_if(C->Kernels, [](const ObjectCode::Kernel &K) {
      return K.Name == KernelName;
    });
    if (It == C->Kernels.end())
      return createStringError(inconvertibleErrorCode(),
                               "no kernel " + KernelName);
    Bytes = Bytes.substr(It->Addr, It->Size);
    J.Kernels = 1;
  }
  J.CodeBytes = Bytes.size();
  return writeBytes(J.OutPath, Bytes);
}

static void runJob(Job &J) {
  auto Start = std::chrono::steady_clock::now();
  LLVMContext Ctx;
  Ctx.setDiagnosticHandler(std::make_unique<CollectingHandler>(J));
  auto Fail = [&](const Twine &Msg) {
    raw_string_ostream OS(J.Log);
    OS << J.U->Input;
    if (!J.U->Entry.empty())
      OS << "(" << J.U->Entry << ")";
    OS << ": error: " << Msg << "\n";
    J.Failed = true;
  };

  // The module keeps the data layout it has: AIR's, until agx-lower-air gives
  // it AGX's (with AGX's, the reader would put AIR's allocas, which bitcode
  // writes without their address space, in AGX's private space).
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIR(J.U->Data, Err, Ctx);
  if (!M) {
    std::string Msg;
    raw_string_ostream OS(Msg);
    Err.print("", OS, /*ShowColors=*/false);
    Fail(StringRef(Msg).trim());
    return;
  }
  if (!NoVerify) {
    std::string Msg;
    raw_string_ostream OS(Msg);
    if (verifyModule(*M, &OS)) {
      Fail("the input module is not valid:\n" + Msg);
      return;
    }
  }
  J.ReadMs = msSince(Start);

  Triple TT(Triple::normalize(TripleName));
  std::string LookupError;
  const Target *T = TargetRegistry::lookupTarget(TT, LookupError);
  if (!T) {
    Fail(LookupError);
    return;
  }
  TargetOptions Options;
  // As llc: assembly with the backend's comments (each kernel's registers,
  // uniforms, threadgroup memory and stack).
  Options.MCOptions.AsmVerbose = true;
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      TT, J.CPU, Features, Options, std::nullopt, std::nullopt,
      codeGenOptLevel()));
  if (!TM) {
    Fail("could not create the AGX target machine");
    return;
  }

  bool IsAIR = M->getNamedMetadata("air.kernel") != nullptr;
  M->setTargetTriple(TT);
  if (!IsAIR)
    M->setDataLayout(TM->createDataLayout());
  // The processor and features of every function that does not name its own.
  for (Function &F : *M) {
    if (F.isDeclaration())
      continue;
    if (!J.CPU.empty() && !F.hasFnAttribute("target-cpu"))
      F.addFnAttr("target-cpu", J.CPU);
    if (!Features.empty()) {
      StringRef Old = F.getFnAttribute("target-features").getValueAsString();
      F.addFnAttr("target-features",
                  Old.empty() ? Features : (Old + "," + Features).str());
    }
  }

  // agx-lower-air (at the pipeline's start, the AGX target machine adds it),
  // then the optimization pipeline.
  auto OptStart = std::chrono::steady_clock::now();
  {
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PipelineTuningOptions PTO;
    PassBuilder PB(Ctx, TM.get(), PTO);
    TargetLibraryInfoImpl TLII(TT);
    FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });
    FAM.registerPass([&] { return PB.buildDefaultAAPipeline(); });
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    OptimizationLevel L = optimizationLevel();
    ModulePassManager MPM = L == OptimizationLevel::O0
                                ? PB.buildO0DefaultPipeline(L)
                                : PB.buildPerModuleDefaultPipeline(L);
    if (!NoVerify)
      MPM.addPass(VerifierPass());
    MPM.run(*M, MAM);
  }
  J.OptMs = msSince(OptStart);
  if (J.Failed)
    return;
  if (M->getNamedMetadata("air.kernel")) {
    Fail("agx-lower-air did not compile the AIR kernels");
    return;
  }

  auto CGStart = std::chrono::steady_clock::now();
  SmallString<0> Buf;
  raw_svector_ostream BufOS(Buf);
  if (EmitKind == Emit::LLVM) {
    M->print(BufOS, nullptr);
  } else if (EmitKind == Emit::BC) {
    WriteBitcodeToFile(*M, BufOS);
  } else {
    legacy::PassManager CG;
    TargetLibraryInfoImpl TLII(TT);
    CG.add(new TargetLibraryInfoWrapperPass(TLII));
    CodeGenFileType FT = EmitKind == Emit::Asm ? CodeGenFileType::AssemblyFile
                                               : CodeGenFileType::ObjectFile;
    if (TM->addPassesToEmitFile(CG, BufOS, nullptr, FT)) {
      Fail("the AGX target cannot emit this file type");
      return;
    }
    CG.run(*M);
  }
  J.CodegenMs = msSince(CGStart);
  if (J.Failed)
    return;

  if (EmitKind == Emit::Code || EmitKind == Emit::AGX) {
    if (Error E = EmitKind == Emit::Code ? writeKernelCode(J, Buf)
                                         : writePureAGX(J, Buf))
      Fail(toString(std::move(E)));
    return;
  }
  for (Function &F : *M)
    if (F.getCallingConv() == CallingConv::AGX_KERNEL && !F.isDeclaration())
      ++J.Kernels;
  std::error_code EC;
  ToolOutputFile Out(J.OutPath, EC,
                     EmitKind == Emit::Asm || EmitKind == Emit::LLVM
                         ? sys::fs::OF_Text
                         : sys::fs::OF_None);
  if (EC) {
    Fail(J.OutPath + ": " + EC.message());
    return;
  }
  Out.os() << Buf;
  Out.keep();
}

/// The file stem of an input, without the extensions of the AIR forms.
static std::string stem(StringRef Path) {
  StringRef S = sys::path::filename(Path);
  for (StringRef Ext : {".metallib", ".air", ".bc", ".ll"})
    if (S.ends_with(Ext) && S.size() > Ext.size())
      return S.drop_back(Ext.size()).str();
  return S.str();
}

/// Entry names as file name parts.
static std::string sanitize(StringRef Name) {
  std::string R;
  for (char C : Name)
    R += isAlnum(C) || C == '_' || C == '-' || C == '.' ? C : '_';
  return R;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
  cl::HideUnrelatedOptions(Cat);
  cl::ParseCommandLineOptions(
      argc, argv,
      "Compile AIR (.metallib, .air, .ll) to AGX machine code\n\n"
      "  The AIR goes through agx-lower-air and -O's pipeline with the AGX\n"
      "  target's passes, then the AGX code generator, in one process; every\n"
      "  AIR module x processor is a job, and jobs run in parallel.\n");
  // Without --emit, -o's extension says what to write (k.agx: pure AGX code).
  if (!EmitKind.getNumOccurrences())
    EmitKind = StringSwitch<Emit>(sys::path::extension(Output))
                   .Case(".agx", Emit::AGX)
                   .Case(".s", Emit::Asm)
                   .Case(".ll", Emit::LLVM)
                   .Case(".bc", Emit::BC)
                   .Default(Emit::Obj);
  if (!KernelName.empty() && EmitKind != Emit::AGX) {
    WithColor::error(errs(), "llvm-air2agx") << "--kernel needs --emit=agx\n";
    return 1;
  }

  std::vector<std::unique_ptr<MemoryBuffer>> Buffers;
  std::vector<Unit> Units;
  StringSet<> Wanted;
  for (const std::string &E : EntryFilter)
    Wanted.insert(E);
  bool Failed = false;
  for (const std::string &In : Inputs) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> B = MemoryBuffer::getFileOrSTDIN(
        In, /*IsText=*/false, /*RequiresNullTerminator=*/false);
    if (!B) {
      WithColor::error(errs(), "llvm-air2agx")
          << In << ": " << B.getError().message() << "\n";
      Failed = true;
      continue;
    }
    MemoryBufferRef Ref = (*B)->getMemBufferRef();
    Buffers.push_back(std::move(*B));
    if (!metallib::isMetalLib(Ref.getBuffer())) {
      if (List)
        outs() << In << ": AIR module (" << Ref.getBufferSize()
               << " bytes)\n";
      Units.push_back({In, "", Ref});
      continue;
    }
    Expected<SmallVector<metallib::Entry, 4>> Entries =
        metallib::readEntries(Ref);
    if (!Entries) {
      WithColor::error(errs(), "llvm-air2agx")
          << In << ": " << toString(Entries.takeError()) << "\n";
      Failed = true;
      continue;
    }
    // Entries that name the same bitcode are one module.
    DenseMap<const char *, size_t> Seen;
    for (const metallib::Entry &E : *Entries) {
      bool HashOK = metallib::hashMatches(E);
      if (List)
        outs() << In << ": entry " << E.Name << ", type "
               << unsigned(E.Type) << ", AIR " << E.AIRMajor << "."
               << E.AIRMinor << ", language " << E.LangMajor << "."
               << E.LangMinor << ", " << E.Bitcode.getBufferSize()
               << " bytes of bitcode"
               << (E.HasHash ? (HashOK ? ", hash ok" : ", HASH MISMATCH")
                             : "")
               << "\n";
      if (!Wanted.empty() && !Wanted.count(E.Name))
        continue;
      if (VerifyHash && !HashOK) {
        WithColor::error(errs(), "llvm-air2agx")
            << In << ": the bitcode of entry " << E.Name
            << " does not match its SHA-256\n";
        Failed = true;
        continue;
      }
      if (!Seen.try_emplace(E.Bitcode.getBufferStart(), Units.size()).second)
        continue;
      Units.push_back({In, E.Name.str(), E.Bitcode});
    }
  }
  if (List)
    return Failed;
  if (Failed)
    return 1;

  std::vector<std::string> CPUList(CPUs.begin(), CPUs.end());
  if (CPUList.empty())
    CPUList.push_back("");
  std::vector<Job> Jobs;
  bool OneJob = Units.size() * CPUList.size() == 1;
  if (!Output.empty() && !OneJob && EmitKind != Emit::Code) {
    WithColor::error(errs(), "llvm-air2agx")
        << "-o names one output, but there are "
        << Units.size() * CPUList.size()
        << " (AIR modules x processors): use --output-dir\n";
    return 1;
  }
  // A metallib whose entries all name one module gives its file the input's
  // name; otherwise each entry adds its own.
  StringMap<unsigned> UnitsPerInput;
  for (const Unit &U : Units)
    ++UnitsPerInput[U.Input];
  // Inputs of one name (a.metallib and a.air) number the later ones' outputs.
  StringMap<unsigned> StemUses;
  StringMap<std::string> StemOf;
  for (const Unit &U : Units)
    if (!StemOf.count(U.Input)) {
      std::string S = stem(U.Input);
      unsigned N = StemUses[S]++;
      StemOf[U.Input] = N ? S + "." + std::to_string(N) : S;
    }
  for (const Unit &U : Units)
    for (const std::string &CPU : CPUList) {
      Job J;
      J.U = &U;
      J.CPU = CPU;
      if (!Output.empty() && OneJob) {
        J.OutPath = Output;
      } else {
        std::string Name = Output.empty() ? StemOf[U.Input] : Output;
        if (!U.Entry.empty() && UnitsPerInput[U.Input] > 1)
          Name += "." + sanitize(U.Entry);
        if (CPUList.size() > 1)
          Name += "." + sanitize(CPU.empty() ? "generic" : CPU);
        Name += extension();
        SmallString<128> P(Output.empty() ? StringRef(OutputDir) : "");
        sys::path::append(P, Name);
        J.OutPath = std::string(P);
      }
      Jobs.push_back(std::move(J));
    }

  if (!OutputDir.empty())
    if (std::error_code EC = sys::fs::create_directories(OutputDir)) {
      WithColor::error(errs(), "llvm-air2agx")
          << OutputDir << ": " << EC.message() << "\n";
      return 1;
    }

  auto Start = std::chrono::steady_clock::now();
  ThreadPoolStrategy S = heavyweight_hardware_concurrency(Threads);
  if (Jobs.size() == 1 || S.compute_thread_count() <= 1) {
    for (Job &J : Jobs)
      runJob(J);
  } else {
    DefaultThreadPool Pool(S);
    for (Job &J : Jobs)
      Pool.async([&J] { runJob(J); });
    Pool.wait();
  }
  double TotalMs = msSince(Start);

  for (const Job &J : Jobs) {
    errs() << J.Log;
    Failed |= J.Failed;
    if (TimeJobs)
      errs() << J.OutPath << ": " << (J.CPU.empty() ? "generic" : J.CPU)
             << ", " << J.Kernels << " kernel(s): read "
             << format("%.1f", J.ReadMs) << " ms, opt "
             << format("%.1f", J.OptMs) << " ms, codegen "
             << format("%.1f", J.CodegenMs) << " ms\n";
  }
  if (TimeJobs)
    errs() << Jobs.size() << " job(s) in " << format("%.1f", TotalMs)
           << " ms\n";
  return Failed;
}
