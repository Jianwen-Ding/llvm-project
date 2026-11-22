//===-- AGXTargetMachine.cpp - Define TargetMachine for AGX ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGX code generation pipeline. Calls are inlined by default; the opt-in
// leaf-call path keeps eligible callees. The control flow graph is
// structurized before instruction selection.
// Unlike AMDGPU, AGX has no scalar unit and no scalar branches: every
// conditional branch, uniform or not, is lowered to execution mask
// operations on the per-thread nesting counter r0l.
//
//===----------------------------------------------------------------------===//

#include "AGXTargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include "AGX.h"
#include "AGXAliasAnalysis.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXTargetTransformInfo.h"
#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"

using namespace llvm;

#define DEBUG_TYPE "agx-target-machine"

static cl::opt<bool> AGXPruneBeforeInline(
    "agx-prune-before-inline", cl::init(true), cl::Hidden,
    cl::desc("Prune unreachable AGX device functions before forced inlining"));

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAGXTarget() {
  RegisterTargetMachine<AGXTargetMachine> X(getTheAGXTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeAGXAlwaysInlinePass(PR);
  initializeAGXLowerAIRLegacyPass(PR);
  initializeAGXAAWrapperPassPass(PR);
  initializeAGXExternalAAWrapperPass(PR);
  initializeAGXPromoteKernelArgsPass(PR);
  initializeAGXCheckedAccessesPass(PR);
  initializeAGXCodeGenPreparePass(PR);
  initializeAGXStrictFPPass(PR);
  initializeAGXLowerFPEnvPass(PR);
  initializeAGXSoftF64Pass(PR);
  initializeAGXExpandWideAtomicsPass(PR);
  initializeAGXAnnotateControlFlowPass(PR);
  initializeAGXSinkAddressesPass(PR);
  initializeAGXRareFallbacksPass(PR);
  initializeAGXThreadgroupAddressesPass(PR);
  initializeAGXHoistGuardedLoadsPass(PR);
  initializeAGXAliasVersioningPass(PR);
  initializeAGXWideIndexVersioningPass(PR);
  initializeAGXAlignmentVersioningPass(PR);
  initializeAGXNarrowInductionsPass(PR);
  initializeAGXSplitWideAccumulatorsPass(PR);
  initializeAGXWordCounterVersioningPass(PR);
  initializeAGXWordIndicesPass(PR);
  initializeAGXSpecialRegisterRangesPassPass(PR);
  initializeAGXVectorAccessesPass(PR);
  initializeAGXLowerGenericAddressesPass(PR);
  initializeAGXDAGToDAGISelLegacyPass(PR);
  initializeAGXFoldOperandsPass(PR);
  initializeAGXLoadLanesPass(PR);
  initializeAGXInvariantHalvesPassPass(PR);
  initializeAGXPairedHalfBitopsPassPass(PR);
  initializeAGXAdvancePointersPassPass(PR);
  initializeAGXHoistReloadsPassPass(PR);
  initializeAGXUniformConstantsPassPass(PR);
  initializeAGXHoistFromElseFlowPass(PR);
  initializeAGXSinkRematPass(PR);
  initializeAGXLowerControlFlowPass(PR);
  initializeAGXInsertWaitsPass(PR);
  initializeAGXColdRegionsPass(PR);
  initializeAGXAsmPrinterPass(PR);
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::PIC_);
}

AGXTargetMachine::AGXTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT, CPU.empty() ? "generic" : CPU, FS,
                               Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small),
                               OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  setRequiresStructuredCFG(true);
  initAsmInfo();
}

AGXTargetMachine::~AGXTargetMachine() = default;

const AGXSubtarget *
AGXTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");
  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;
  // The module flag "agx-strict" selects Mesa's instruction forms for every
  // function of the module.
  if (const auto *Flag = mdconst::extract_or_null<ConstantInt>(
          F.getParent()->getModuleFlag("agx-strict")))
    if (!Flag->isZero())
      FS = FS.empty() ? "+agx-strict" : FS + ",+agx-strict";
  if (const auto *Flag = mdconst::extract_or_null<ConstantInt>(
          F.getParent()->getModuleFlag("agx-checked")))
    if (!Flag->isZero())
      FS = FS.empty() ? "+agx-checked" : FS + ",+agx-checked";

  auto &I = SubtargetMap[CPU + FS];
  if (!I)
    I = std::make_unique<AGXSubtarget>(TargetTriple, CPU, FS, *this);
  return I.get();
}

TargetTransformInfo
AGXTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<AGXTTIImpl>(this, F));
}

MachineFunctionInfo *AGXTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return AGXMachineFunctionInfo::create<AGXMachineFunctionInfo>(Allocator, F,
                                                                STI);
}

// Stack slots are private memory: a memory operand the DAG rebuilds from a frame
// index (merged or split stores of an alloca) has the stack's pseudo source value,
// whose address space must be the alloca's for the private_load / private_store
// patterns to match (it was 0: "Cannot select: store (s64) into %stack.0" on two
// zero <2 x float> stores into one alloca, llama.cpp's mul_mat_vec_q_moe).
unsigned AGXTargetMachine::getAddressSpaceForPseudoSourceKind(unsigned Kind) const {
  switch (Kind) {
  case PseudoSourceValue::Stack:
  case PseudoSourceValue::FixedStack:
    return AGX::PRIVATE_ADDRESS;
  }
  return CodeGenTargetMachineImpl::getAddressSpaceForPseudoSourceKind(Kind);
}

bool AGXTargetMachine::isNoopAddrSpaceCast(const DataLayout &DL, unsigned SrcAS,
                                           unsigned DestAS) const {
  // Generic, device and constant pointers all address device memory.
  auto IsDevice = [](unsigned AS) {
    return AS == AGX::GENERIC_ADDRESS || AS == AGX::DEVICE_ADDRESS ||
           AS == AGX::CONSTANT_ADDRESS;
  };
  return IsDevice(SrcAS) && IsDevice(DestAS);
}

namespace {

class AGXPassConfig : public TargetPassConfig {
public:
  AGXPassConfig(AGXTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    // There is no stack unwinding or shrink wrapping.
    disablePass(&StackMapLivenessID);
    disablePass(&FuncletLayoutID);
    disablePass(&PatchableFunctionID);
    disablePass(&ShrinkWrapID);
    disablePass(&PostRAMachineSinkingID);
    // Keep the block order of the structurizer: execution mask code falls
    // through from each construct into the next, as the vendor and Mesa
    // compilers lay it out.
    disablePass(&MachineBlockPlacementID);
    substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
  }

  AGXTargetMachine &getAGXTargetMachine() const {
    return getTM<AGXTargetMachine>();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  void addMachineSSAOptimization() override;
  void addPostRegAlloc() override;
  void addPreEmitPass() override;
};

} // namespace

void AGXTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  // AIR input becomes AGX before anything optimizes it.
  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(AGXLowerAIRPass(*this));
      });
  // Unrolled straight-line runs of device accesses versioned on their
  // buffers not overlapping, after the last unrolling (L15-E528).
  PB.registerOptimizerLastEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase) {
        if (Level != OptimizationLevel::O0)
          MPM.addPass(
              createModuleToFunctionPassAdaptor(AGXAliasVersioningPass()));
      });
  // The exps below another loop's maximum marked before the unroller
  // (L15-E731).
  PB.registerVectorizerStartEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel Level) {
        if (Level != OptimizationLevel::O0)
          FPM.addPass(AGXMarkExpsBelowMaxPass());
      });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager &FPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name != "agx-mark-exps-below-max")
          return false;
        FPM.addPass(AGXMarkExpsBelowMaxPass());
        return true;
      });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager &FPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name != "agx-alias-version")
          return false;
        FPM.addPass(AGXAliasVersioningPass());
        return true;
      });
  PB.registerPipelineParsingCallback(
      [this](StringRef Name, ModulePassManager &MPM,
             ArrayRef<PassBuilder::PipelineElement>) {
        if (Name != "agx-lower-air")
          return false;
        MPM.addPass(AGXLowerAIRPass(*this));
        return true;
      });
  PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &FAM) {
    FAM.registerPass([] { return AGXAA(); });
  });
  PB.registerParseAACallback([](StringRef Name, AAManager &AAM) {
    if (Name != "agx-aa")
      return false;
    AAM.registerFunctionAnalysis<AGXAA>();
    return true;
  });
}

void AGXTargetMachine::registerEarlyDefaultAliasAnalyses(AAManager &AAM) {
  AAM.registerFunctionAnalysis<AGXAA>();
}

static cl::opt<bool> AGXSchedKeepOccupancy(
    "agx-sched-keep-occupancy", cl::init(true), cl::Hidden,
    cl::desc("Keep a region's original order where scheduling it would lower "
             "the threads its register pressure lets reside"));

static cl::opt<bool> AGXSchedKeepOccupancyNeed(
    "agx-sched-keep-occupancy-need", cl::init(true), cl::Hidden,
    cl::desc("agx-sched-keep-occupancy not for a region without stores, of "
             "fewer than agx-sched-core-occupancy-per-load instructions a "
             "device load, whose new order keeps a third of the core's "
             "SIMD-groups and (with 16 halves of margin) the kernel's "
             "threadgroups"));

static cl::opt<bool> AGXSchedCoreOccupancy(
    "agx-sched-core-occupancy", cl::init(true), cl::Hidden,
    cl::desc("Also keep the original order of a region with fewer than 4 "
             "device loads where its schedule would cut the SIMD-groups its "
             "registers let reside on a core by agx-sched-core-occupancy-cut "
             "or more"));

static cl::opt<unsigned> AGXSchedCoreOccupancyPerLoad(
    "agx-sched-core-occupancy-per-load", cl::init(16), cl::Hidden,
    cl::desc("agx-sched-core-occupancy also for a region of more device "
             "loads with at least this many instructions for each (0: "
             "never)"));

static cl::opt<unsigned> AGXSchedCoreOccupancyCut(
    "agx-sched-core-occupancy-cut", cl::init(10), cl::Hidden,
    cl::desc("agx-sched-core-occupancy's cut of the core's SIMD-groups, in "
             "tenths: 10, any (1.8 times until L15-E761, 1.2 until "
             "L15-E799)"));

static cl::opt<unsigned> AGXSchedCoreShortInstrs(
    "agx-sched-core-occupancy-short", cl::init(40), cl::Hidden,
    cl::desc("Regions of fewer instructions take "
             "-agx-sched-core-occupancy-short-cut, not "
             "-agx-sched-core-occupancy-cut"));

static cl::opt<unsigned> AGXSchedCoreShortCut(
    "agx-sched-core-occupancy-short-cut", cl::init(12), cl::Hidden,
    cl::desc("agx-sched-core-occupancy's cut, in tenths, for regions of "
             "fewer than -agx-sched-core-occupancy-short instructions"));

static cl::opt<unsigned> AGXSchedLoadsFirstCut(
    "agx-sched-loads-first-cut", cl::init(18), cl::Hidden,
    cl::desc("agx-sched-original-order-loads-first's order dropped where it "
             "cuts the core's SIMD-groups this much, in tenths"));

static cl::opt<unsigned> AGXSchedNeedCut(
    "agx-sched-keep-occupancy-need-cut", cl::init(30), cl::Hidden,
    cl::desc("agx-sched-keep-occupancy-need's cut of the core's SIMD-groups "
             "past which the original order stays, in tenths"));

static cl::opt<unsigned> AGXSchedCoreManyCut(
    "agx-sched-core-occupancy-many-cut", cl::init(20), cl::Hidden,
    cl::desc("agx-sched-core-occupancy also for the other regions of word "
             "loads and at most agx-sched-core-occupancy-many-per-load "
             "instructions a load, at this cut of the core's SIMD-groups in "
             "tenths (0: never)"));

static cl::opt<unsigned> AGXSchedCoreManyPerLoad(
    "agx-sched-core-occupancy-many-per-load", cl::init(3), cl::Hidden,
    cl::desc("agx-sched-core-occupancy-many-cut only for regions of at most "
             "this many instructions a device load"));

static cl::opt<unsigned> AGXSchedCoreStoreCut(
    "agx-sched-core-occupancy-store-cut", cl::init(12), cl::Hidden,
    cl::desc("agx-sched-core-occupancy-many-cut for regions with a device "
             "store for every two device loads or more, counting the stores "
             "with the loads, at this cut in tenths (0: never)"));

static cl::opt<bool> AGXSchedKnownGrid(
    "agx-sched-known-grid", cl::init(true), cl::Hidden,
    cl::desc("Keep a schedule whose registers still let every SIMD-group of "
             "a known grid reside on one core"));

static cl::opt<bool> AGXOriginalOrderLoadsFirst(
    "agx-sched-original-order-loads-first", cl::init(true), cl::Hidden,
    cl::desc("A region kept in its original order, without barriers or "
             "threadgroup accesses, issues a ready device load first where "
             "its registers still let the kernel's threadgroups in"));

static cl::opt<unsigned> AGXSchedLoadsFirstMinGroups(
    "agx-sched-loads-first-min-groups", cl::init(28), cl::Hidden,
    cl::desc("agx-sched-original-order-loads-first's order dropped where it "
             "leaves the core fewer SIMD-groups than this (and than before)"));

static cl::opt<unsigned> AGXSchedLoadsFirstLimit(
    "agx-sched-loads-first-limit", cl::init(12), cl::Hidden,
    cl::desc("Where agx-sched-original-order-loads-first's order is dropped, "
             "that order again with at most this many device loads ahead of "
             "their first reader (0: off)"));

namespace {
/// Picks the region's instructions in their order before the first
/// scheduling (top-down; that order is a topological one of the DAG).
class AGXOriginalOrderStrategy : public MachineSchedStrategy {
  const DenseMap<const MachineInstr *, unsigned> &Index;
  SmallVector<SUnit *, 64> Ready;

public:
  AGXOriginalOrderStrategy(const DenseMap<const MachineInstr *, unsigned> &I)
      : Index(I) {}
  void initialize(ScheduleDAGMI *) override {
    Ready.clear();
    Ahead.clear();
  }
  /// A ready device load first (L15-E498).
  bool LoadsFirst = false;
  /// With LoadsFirst, at most this many device loads ahead of their first
  /// reader (0: any).
  unsigned Limit = 0;
  SmallPtrSet<const SUnit *, 16> Ahead;
  SUnit *pickNode(bool &IsTopNode) override {
    IsTopNode = true;
    if (Ready.empty())
      return nullptr;
    bool Room = !Limit || Ahead.size() < Limit;
    auto Key = [&](SUnit *X) {
      bool Load = LoadsFirst && Room && isAGXDeviceLoad(*X->getInstr());
      return std::make_pair(!Load, Index.lookup(X->getInstr()));
    };
    auto Best = llvm::min_element(
        Ready, [&](SUnit *A, SUnit *B) { return Key(A) < Key(B); });
    SUnit *SU = *Best;
    Ready.erase(Best);
    return SU;
  }
  void schedNode(SUnit *SU, bool) override {
    if (!Limit)
      return;
    for (const SDep &D : SU->Preds)
      if (D.getKind() == SDep::Data)
        Ahead.erase(D.getSUnit());
    if (isAGXDeviceLoad(*SU->getInstr()))
      Ahead.insert(SU);
  }
  void releaseTopNode(SUnit *SU) override { Ready.push_back(SU); }
  void releaseBottomNode(SUnit *) override {}
};

/// The generic scheduler, but a region whose new order needs more registers
/// than its original one, enough to cost resident threads (the hardware
/// allocates registers per thread: AGXSubtarget::maxThreadsForHalves), is
/// scheduled again in its original order, as AMDGPU's scheduler reverts a
/// region that lowered occupancy. The generic heuristics only see register
/// pressure against the whole register file, not against the occupancy
/// steps below it: on large straight-line blocks (emulated tensor-core and
/// matrix operations) their latency ordering kept dozens more values live.
static cl::opt<bool> AGXSchedCoreOccupancyFinal(
    "agx-sched-core-occupancy-final", cl::init(true), cl::Hidden,
    cl::desc("A region agx-sched-core-occupancy put back in its original "
             "order is scheduled freely again at the function's end where "
             "its free order's SIMD-groups are no fewer than the function's "
             "most pressed block leaves"));

/// SIMD-groups a core holds for a register count in halves (the launch's
/// unit of 8, AGX Performance Simulator dispatch.rs).
static unsigned coreGroupsForHalves(unsigned H) {
  H = alignTo(std::max(16u, H), 8);
  return std::min(96u, 2 * (3328 / (2 * H)));
}

class AGXScheduleDAGMILive : public ScheduleDAGMILive {
  int GPRSet = -1;
  /// Regions put back in their original order only for the core's
  /// SIMD-groups, with the halves of their free order.
  struct Reverted {
    MachineBasicBlock *MBB;
    MachineInstr *First;
    MachineBasicBlock::iterator End;
    unsigned NumInstrs;
    unsigned FreeHalves;
  };
  SmallVector<Reverted, 4> Reverts;
  bool ForceFree = false;

  /// The most GPR16 pressure of any block of the function as it is now.
  unsigned functionHalves() {
    unsigned H = 0;
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.empty())
        continue;
      IntervalPressure IP;
      RegPressureTracker T(IP);
      T.init(&MF, RegClassInfo, LIS, &MBB, MBB.end(), ShouldTrackLaneMasks,
             /*TrackUntiedDefs=*/false);
      while (T.getPos() != MBB.begin())
        T.recede();
      T.closeRegion();
      H = std::max(H, IP.MaxSetPressure[GPRSet]);
    }
    return H;
  }
  /// The tracker of regionHalves, kept across regions: a new one's set of
  /// live registers is allocated and zeroed for all the function's virtual
  /// registers, twice per region (quadratic in large functions); init resets
  /// a kept one.
  IntervalPressure P;
  RegPressureTracker RPT{P};

  /// The peak of the 16-bit GPR halves pressure set over the region, in its
  /// current order (live-through values included).
  unsigned regionHalves() {
    if (GPRSet < 0 || RegionBegin == RegionEnd)
      return 0;
    RPT.init(&MF, RegClassInfo, LIS, BB, RegionEnd, ShouldTrackLaneMasks,
             /*TrackUntiedDefs=*/false);
    while (RPT.getPos() != RegionBegin)
      RPT.recede();
    RPT.closeRegion();
    return P.MaxSetPressure[GPRSet];
  }

public:
  AGXScheduleDAGMILive(MachineSchedContext *C)
      : ScheduleDAGMILive(C, std::make_unique<GenericScheduler>(C)) {
    for (unsigned I = 0, E = TRI->getNumRegPressureSets(); I < E; ++I)
      if (StringRef(TRI->getRegPressureSetName(I)) == "GPR16")
        GPRSet = I;
  }

  void schedule() override {
    if (!AGXSchedKeepOccupancy || GPRSet < 0 || ForceFree) {
      ScheduleDAGMILive::schedule();
      return;
    }
    DenseMap<const MachineInstr *, unsigned> Index;
    for (MachineInstr &MI : make_range(RegionBegin, RegionEnd))
      Index[&MI] = Index.size();
    unsigned Before = regionHalves();
    ScheduleDAGMILive::schedule();
    unsigned After = regionHalves();
    LLVM_DEBUG(dbgs() << "agx-occupancy region " << BB->getName() << " "
                      << Index.size() << " instrs: halves " << Before << " -> "
                      << After << "\n");
    // The original order where the new one lowers the threads a threadgroup
    // may have, or, for a region of few device loads (whose clustered loads
    // E187 and the like rely on), cuts the SIMD-groups resident on a
    // core (min(96, 2 floor(3328 / (2 halves))), AGX Performance Simulator
    // dispatch, registers allocated by 8 halves) by 1.2 times or more: a tiled
    // matrix product's threadgroup loads, all hoisted ahead of their fmas,
    // doubled its registers for a latency the resident groups hide anyway
    // (L15-E239; 1.8 times until a row loop of transcendentals, 48 -> 74
    // halves and 68 -> 40 groups, was seen slower than its original order:
    // row_exp, row_sigmoid, cond_compute, sincos_rows -5%, L15-E761).
    auto CoreGroups = [](unsigned H) {
      H = alignTo(std::max(16u, H), 8);
      return std::min(96u, 2 * (3328 / (2 * H)));
    };
    unsigned DeviceLoads = 0;
    for (MachineInstr &MI : make_range(RegionBegin, RegionEnd))
      DeviceLoads += isAGXDeviceLoad(MI);
    bool Threads = AGXSubtarget::maxThreadsForHalves(After) <
                   AGXSubtarget::maxThreadsForHalves(Before);
    // (or of more loads where each has 16 instructions or more to hide its
    // latency behind: the resident groups' own then hide it, an online
    // softmax's exponentials, a byte swap's shifts, not a matrix product's
    // loads ahead of its fmas, L15-E681)
    bool FewLoads =
        DeviceLoads < 4 ||
        (AGXSchedCoreOccupancyPerLoad &&
         Index.size() >= AGXSchedCoreOccupancyPerLoad * DeviceLoads);
    // Fewer threads than before matter only where the threadgroups no longer
    // fit, or the core's groups must hide the latency: a loop of light work
    // a load without stores whose loads the new order issues ahead of their
    // uses, the original one each load waited for at its first use (a
    // matrix product's B rows, gemm_reg4 -4.2%, gemv_q8f -11.0% at 34 -> 18
    // and 58 -> 26 groups; with stores, a row normalization's 58 -> 26 lost
    // 43%; bf16_dot_rows' 58 -> 12, 18%, L15-E753).
    if (Threads && AGXSchedKeepOccupancyNeed) {
      unsigned Need = MF.getFunction().getFnAttributeAsParsedInteger(
          "agx-max-threads-per-threadgroup", 1024);
      if (Need == 0 || Need > 1024)
        Need = 1024;
      bool Stores = false;
      for (MachineInstr &MI : make_range(RegionBegin, RegionEnd))
        Stores |= MI.mayStore();
      Threads = Stores || FewLoads ||
                AGXSchedNeedCut * CoreGroups(After) <
                    10 * CoreGroups(Before) ||
                AGXSubtarget::maxThreadsForHalves(After + 16) < Need;
    }
    // (a short region at E761's 1.2 times: swiglu's and permute4d's of 28
    // instructions, 96 -> 82 groups, and permute4d's of 33, 68 -> 58, hide
    // more of their loads' latency in the free order than the groups it
    // costs, +4.3% / +2.0% at 1.0; the longer softmax, log and sincos rows'
    // gain at 1.0, L15-E806, E807)
    unsigned CoreCut = Index.size() < AGXSchedCoreShortInstrs
                           ? AGXSchedCoreShortCut
                           : AGXSchedCoreOccupancyCut;
    bool Core = AGXSchedCoreOccupancy && FewLoads &&
                CoreCut * CoreGroups(After) < 10 * CoreGroups(Before);
    // (and a region of little but word loads, at a cut of 2 times: a naive
    // matrix product's A and B columns, all issued ahead of their fmas by the
    // free order, 96 -> 40 groups; in order, 55 groups hide them,
    // matmul_naive -15.1%, L15-E764)
    auto WordLoads = [&] {
      for (MachineInstr &MI : make_range(RegionBegin, RegionEnd))
        if (isAGXDeviceLoad(MI))
          for (const MachineMemOperand *MMO : MI.memoperands())
            if (!MMO->getSize().hasValue() ||
                MMO->getSize().getValue().getKnownMinValue() < 4)
              return false;
      return true;
    };
    // (and a stream of word loads and stores, at E761's cut: a column's
    // values loaded, scaled and stored, the free order's loads all ahead,
    // 96 -> 68 groups; in order col_softmax_norm -10.2%; with fewer stores,
    // matvec8's eight to 32 loads, +1.1%, L15-E765)
    unsigned DeviceStores = 0;
    for (MachineInstr &MI : make_range(RegionBegin, RegionEnd))
      DeviceStores += MI.mayStore() && !MI.mayLoad() &&
                      any_of(MI.memoperands(), [](const MachineMemOperand *M) {
                        return M->getAddrSpace() == AGX::DEVICE_ADDRESS;
                      });
    Core |= AGXSchedCoreOccupancy && !FewLoads && AGXSchedCoreStoreCut &&
            DeviceStores && 2 * DeviceStores >= DeviceLoads &&
            Index.size() <=
                AGXSchedCoreManyPerLoad * (DeviceLoads + DeviceStores) &&
            AGXSchedCoreStoreCut * CoreGroups(After) <
                10 * CoreGroups(Before) &&
            WordLoads();
    Core |= AGXSchedCoreOccupancy && !FewLoads && AGXSchedCoreManyCut &&
            Index.size() <= AGXSchedCoreManyPerLoad * DeviceLoads &&
            AGXSchedCoreManyCut * CoreGroups(After) <
                10 * CoreGroups(Before) &&
            WordLoads();
    // A known grid whose SIMD-groups all reside on one core with the new
    // registers (and whose threadgroups still fit) loses no occupancy
    // anywhere: the new order stays, its loads' latencies hidden behind each
    // other (L15-E319).
    LLVM_DEBUG(dbgs() << "agx-occupancy decide: loads " << DeviceLoads
                      << " instrs " << Index.size() << " groups "
                      << CoreGroups(Before) << " -> " << CoreGroups(After)
                      << " threads " << Threads << " core " << Core << "\n");
    if (AGXSchedKnownGrid && (Threads || Core)) {
      const Function &Fn = MF.getFunction();
      unsigned TG =
          Fn.getFnAttributeAsParsedInteger("agx-max-threads-per-threadgroup", 0);
      uint64_t GridThreads = 0;
      if (Attribute A = Fn.getFnAttribute("agx-threads-per-grid");
          A.isStringAttribute()) {
        GridThreads = 1;
        SmallVector<StringRef, 3> Dims;
        A.getValueAsString().split(Dims, ',');
        for (StringRef D : Dims) {
          uint64_t V = 0;
          if (D.trim().getAsInteger(10, V) || V == 0) {
            GridThreads = 0;
            break;
          }
          GridThreads *= V;
        }
      }
      if (TG && GridThreads &&
          AGXSubtarget::maxThreadsForHalves(After) >= TG &&
          divideCeil(GridThreads, TG) * divideCeil(TG, 32) <=
              CoreGroups(After))
        Threads = Core = false;
    }
    if (After <= Before || (!Threads && !Core))
      return;
    // the original order again, through the scheduler's own moves; in a
    // region without barriers or threadgroup accesses its ready device loads
    // first, unless that order lowers the threads below what the kernel's
    // threadgroups have (its known maximum, else 1024; with 16 halves of
    // margin for register allocation) or cuts the core's SIMD-groups as
    // above (L15-E498: attention decode's loads ahead of its per-key work,
    // whose threadgroups of 32 the registers for them still allow)
    bool LoadsFirst = false;
    if (AGXOriginalOrderLoadsFirst && DeviceLoads) {
      LoadsFirst = true;
      for (MachineInstr &MI : make_range(RegionBegin, RegionEnd)) {
        for (const MachineMemOperand *MMO : MI.memoperands())
          LoadsFirst &= MMO->getAddrSpace() != AGX::THREADGROUP_ADDRESS;
        LoadsFirst &= MI.getOpcode() != AGX::THREADGROUP_BARRIER &&
                      MI.getOpcode() != AGX::MEMORY_BARRIER;
      }
    }
    auto Reschedule = [&](bool First, unsigned Limit = 0) {
      auto Strategy = std::make_unique<AGXOriginalOrderStrategy>(Index);
      Strategy->LoadsFirst = First;
      Strategy->Limit = Limit;
      std::unique_ptr<MachineSchedStrategy> Original = std::move(Strategy);
      // (without the mutations, whose clustering edges need not hold in the
      // original order)
      std::vector<std::unique_ptr<ScheduleDAGMutation>> Kept;
      std::swap(Mutations, Kept);
      std::swap(SchedImpl, Original);
      ScheduleDAGMILive::schedule();
      std::swap(SchedImpl, Original);
      std::swap(Mutations, Kept);
    };
    Reschedule(LoadsFirst);
    if (LoadsFirst) {
      unsigned H = regionHalves();
      unsigned Need = MF.getFunction().getFnAttributeAsParsedInteger(
          "agx-max-threads-per-threadgroup", 1024);
      if (Need == 0 || Need > 1024)
        Need = 1024;
      // (or leaves the core fewer than 28 groups, fewer than before: too few
      // to hide the latency the loads ahead saved: attention decode's 40 ->
      // 26, whose loads ahead gained under E498 and lost once its per-key
      // work shrank, L15-E530; the online softmaxes keep 32 to 46)
      LLVM_DEBUG(dbgs() << "agx-occupancy loads first: halves " << H
                        << " groups " << CoreGroups(H) << " vs "
                        << CoreGroups(Before) << "\n");
      auto Bad = [&](unsigned H) {
        return H > Before &&
               (AGXSubtarget::maxThreadsForHalves(H + 16) < Need ||
                AGXSchedLoadsFirstCut * CoreGroups(H) <
                    10 * CoreGroups(Before) ||
                (CoreGroups(H) < AGXSchedLoadsFirstMinGroups &&
                 CoreGroups(H) < CoreGroups(Before)));
      };
      // (and dropped, the same order with at most 12 loads ahead of their
      // readers, where that one keeps the groups: attention decode's keys
      // and values partly ahead, attn_decode_f16 -2.1%, softmax_thread
      // -0.5%; 4 ahead, bf16_dot_rows +13%, L15-E754)
      if (Bad(H)) {
        bool Done = false;
        if (AGXSchedLoadsFirstLimit) {
          Reschedule(true, AGXSchedLoadsFirstLimit);
          unsigned H2 = regionHalves();
          LLVM_DEBUG(dbgs() << "agx-occupancy loads first, limited: halves "
                            << H2 << " groups " << CoreGroups(H2) << "\n");
          Done = !Bad(H2);
        }
        if (!Done)
          Reschedule(false);
      }
    }
    LLVM_DEBUG(dbgs() << "agx-occupancy reverted: halves " << regionHalves()
                      << "\n");
    if (AGXSchedCoreOccupancyFinal && Core && !Threads && !LoadsFirst &&
        RegionBegin != RegionEnd)
      Reverts.push_back({BB, &*RegionBegin, RegionEnd, NumRegionInstrs, After});
  }

  /// The regions reverted for the core's SIMD-groups alone whose free order
  /// keeps as many as the function's most pressed block now leaves (another
  /// region sets the count: where_rows' first region, 96 -> 82 groups beside
  /// one of 46) are scheduled freely again: the groups do not change and
  /// the free order hides more latency (L15-E804).
  void finalizeSchedule() override {
    if (!Reverts.empty()) {
      unsigned Final = functionHalves();
      for (Reverted &R : Reverts) {
        if (coreGroupsForHalves(R.FreeHalves) < coreGroupsForHalves(Final))
          continue;
        LLVM_DEBUG(dbgs() << "agx-occupancy final: free again, halves "
                          << R.FreeHalves << " beside " << Final << "\n");
        startBlock(R.MBB);
        enterRegion(R.MBB, R.First->getIterator(), R.End, R.NumInstrs);
        ForceFree = true;
        schedule();
        ForceFree = false;
        exitRegion();
        finishBlock();
        Final = std::max(Final, R.FreeHalves);
      }
      Reverts.clear();
    }
    ScheduleDAGMILive::finalizeSchedule();
  }
};
} // namespace

ScheduleDAGInstrs *
AGXTargetMachine::createMachineScheduler(MachineSchedContext *C) const {
  ScheduleDAGMILive *DAG = new AGXScheduleDAGMILive(C);
  DAG->addMutation(createAGXMemClusterMutation());
  return DAG;
}

ScheduleDAGInstrs *
AGXTargetMachine::createPostMachineScheduler(MachineSchedContext *C) const {
  return createAGXPostRAScheduler(C);
}

TargetPassConfig *AGXTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AGXPassConfig(*this, PM);
}

static cl::opt<unsigned> AGXStraightLineSRMaxInstructions(
    "agx-straight-line-sr-max-instructions", cl::init(200000), cl::Hidden,
    cl::desc("Straight-line strength reduction and n-ary reassociation only "
             "for functions of at most this many instructions (0: all)"));

static cl::opt<bool> AGXEarlySRRanges(
    "agx-early-sr-ranges", cl::init(true), cl::Hidden,
    cl::desc("Special register ranges during straight-line strength "
             "reduction (taken off before loop strength reduction)"));

static cl::opt<bool> AGXNonNegativeSignSplats(
    "agx-nonnegative-sign-splats", cl::init(true), cl::Hidden,
    cl::desc("x >> (bits - 1) of an x known not negative from the special "
             "register ranges as 0"));

namespace {
/// Straight-line strength reduction, n-ary reassociation and the CSE after
/// them, on functions of at most AGXStraightLineSRMaxInstructions
/// instructions. Their scalar evolution of a kernel with millions of
/// instructions (Blender Cycles' shade_shadow, everything inlined) took
/// gigabytes, for address arithmetic sharing meant for unrolled tiles.
static cl::opt<bool> AGXMergedIndexChains(
    "agx-merged-index-chains", cl::init(true), cl::Hidden,
    cl::desc("a device access at a chain of geps whose indices sum below "
             "2^32 as one 32-bit index"));

static cl::opt<bool> AGXFarOffsetIndices(
    "agx-far-offset-indices", cl::init(true), cl::Hidden,
    cl::desc("a device address's constant offset past the immediate index as "
             "a 32-bit index add"));

class AGXStraightLineSRPass : public FunctionPass {
  TargetMachine *TM;
  unsigned Level;

public:
  static char ID;
  AGXStraightLineSRPass(TargetMachine *TM, unsigned Level)
      : FunctionPass(ID), TM(TM), Level(Level) {}
  StringRef getPassName() const override {
    return "AGX straight-line strength reduction (size limited)";
  }
  bool runOnFunction(Function &F) override {
    if (skipFunction(F) || (AGXStraightLineSRMaxInstructions &&
                            F.getInstructionCount() >
                                AGXStraightLineSRMaxInstructions))
      return false;
    PassBuilder PB(F.getContext(), TM);
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    FunctionPassManager FPM;
    FPM.addPass(StraightLineStrengthReducePass());
    FPM.addPass(EarlyCSEPass());
    FPM.addPass(NaryReassociatePass());
    if (Level > 1)
      FPM.addPass(GVNPass());
    else
      FPM.addPass(EarlyCSEPass());
    // The thread indices' bounds let these passes see an index plus a
    // constant (zext(tid + 256)) as the index and a constant offset, which
    // memory accesses take as an immediate. They are taken off again: with
    // them, loop strength reduction makes 64-bit offsets (L14-E1); the
    // special register ranges pass gives them back after it.
    SmallVector<WeakVH, 8> Ranged;
    if (AGXEarlySRRanges)
      AGX::addSpecialRegisterRanges(F, &Ranged);
    bool Changed = !FPM.run(F, FAM).areAllPreserved();
    // With the bounds, an index that starts at a thread position and steps
    // up without signed overflow is known not negative: the sign splat of
    // a signed division or remainder by a power of two (x >> 31, the
    // correction) is 0, so they select as shifts and masks.
    if (AGXEarlySRRanges && AGXNonNegativeSignSplats && !Ranged.empty()) {
      using namespace PatternMatch;
      SimplifyQuery SQ(F.getDataLayout());
      for (Instruction &I : make_early_inc_range(instructions(F))) {
        Value *X;
        const APInt *C;
        if (match(&I, m_AShr(m_Value(X), m_APInt(C))) &&
            *C == I.getType()->getScalarSizeInBits() - 1 &&
            isKnownNonNegative(X, SQ.getWithInstruction(&I))) {
          I.replaceAllUsesWith(Constant::getNullValue(I.getType()));
          I.eraseFromParent();
          Changed = true;
        }
      }
    }
    if (AGXFarOffsetIndices)
      Changed |= farOffsetIndices(F);
    if (AGXMergedIndexChains)
      Changed |= mergedIndexChains(F);
    for (WeakVH &V : Ranged)
      if (auto *CI = dyn_cast_or_null<CallBase>(V))
        CI->removeRetAttr(Attribute::Range);
    return Changed;
  }

  /// A device load or store at a chain of geps gep T, (gep T, B, x1), x2 ...
  /// (and constant byte offsets a multiple of sizeof(T)), its indices known
  /// non-negative and their sum below 2^32 by the thread bounds (a memref
  /// subview's offset, then the element's index, as expand-strided-metadata
  /// lowers it): gep T, B, zext(the 32-bit sum), so that the access takes the
  /// uniform base and an index register instead of 64-bit adds (L15-E303).
  static bool mergedIndexChains(Function &F) {
    using namespace PatternMatch;
    const DataLayout &DL = F.getDataLayout();
    bool Changed = false;
    for (Instruction &I : instructions(F)) {
      Value *Ptr = getLoadStorePointerOperand(&I);
      if (!Ptr || Ptr->getType()->getPointerAddressSpace() != 1)
        continue;
      // (index, scale in elements of T) along the chain
      SmallVector<std::pair<Value *, uint64_t>, 4> Terms;
      Type *T = nullptr;
      uint64_t S = 0;
      int64_t ConstElems = 0;
      Value *P = Ptr;
      unsigned Geps = 0;
      bool Ok = true;
      while (auto *G = dyn_cast<GetElementPtrInst>(P)) {
        if (G->getNumIndices() != 1) {
          Ok = false;
          break;
        }
        Value *X = G->getOperand(1);
        TypeSize GS = DL.getTypeAllocSize(G->getSourceElementType());
        if (GS.isScalable()) {
          Ok = false;
          break;
        }
        if (auto *C = dyn_cast<ConstantInt>(X)) {
          // a constant offset in bytes (or elements of another size)
          int64_t Bytes = C->getSExtValue() * int64_t(GS.getFixedValue());
          if (!T) {
            // (the element type: from the next non-constant gep)
            ConstElems += Bytes; // bytes until T is known
          } else {
            if (Bytes % int64_t(S)) {
              Ok = false;
              break;
            }
            ConstElems += Bytes / int64_t(S);
          }
        } else {
          if (!T) {
            T = G->getSourceElementType();
            S = GS.getFixedValue();
            if (!S || ConstElems % int64_t(S)) {
              Ok = false;
              break;
            }
            ConstElems /= int64_t(S);
          } else if (GS.getFixedValue() != S) {
            Ok = false;
            break;
          }
          Terms.push_back({X, 1});
        }
        ++Geps;
        P = G->getPointerOperand();
      }
      if (!Ok || !T || Geps < 2 || Terms.size() < 2 || ConstElems < 0)
        continue;
      // every term non-negative, their sum below 2^32
      SimplifyQuery SQ(DL, &I);
      uint64_t Max = uint64_t(ConstElems);
      for (auto &[X, Sc] : Terms) {
        // (known bits: through the shifts and additions of the bounded
        // thread positions)
        KnownBits K = computeKnownBits(X, SQ);
        APInt KMax = K.getMaxValue();
        if (KMax.ugt(UINT32_MAX))
          Ok = false;
        else
          Max += KMax.getZExtValue();
        if (!Ok || Max > UINT32_MAX)
          break;
      }
      if (!Ok || Max > UINT32_MAX)
        continue;
      IRBuilder<> B(&I);
      // The low 32 bits of a term, through the operations whose low bits
      // are those of their operands' (exact modulo 2^32), to a zext's i32.
      std::function<Value *(Value *, unsigned)> Low = [&](Value *X,
                                                         unsigned D) -> Value * {
        Value *Y;
        if (match(X, m_ZExt(m_Value(Y))) && Y->getType()->isIntegerTy(32))
          return Y;
        if (auto *C = dyn_cast<ConstantInt>(X))
          return B.getInt32(uint32_t(C->getZExtValue()));
        auto *BO = dyn_cast<BinaryOperator>(X);
        if (D < 6 && BO &&
            (BO->getOpcode() == Instruction::Add ||
             BO->getOpcode() == Instruction::Sub ||
             BO->getOpcode() == Instruction::Mul ||
             BO->getOpcode() == Instruction::Shl ||
             BO->getOpcode() == Instruction::And ||
             BO->getOpcode() == Instruction::Or ||
             BO->getOpcode() == Instruction::Xor) &&
            (BO->getOpcode() != Instruction::Shl ||
             (isa<ConstantInt>(BO->getOperand(1)) &&
              cast<ConstantInt>(BO->getOperand(1))->getZExtValue() < 32)))
          return B.CreateBinOp(BO->getOpcode(), Low(BO->getOperand(0), D + 1),
                               Low(BO->getOperand(1), D + 1));
        return B.CreateTrunc(X, B.getInt32Ty());
      };
      Value *Sum = nullptr;
      for (auto &[X, Sc] : Terms) {
        Value *X32 = Low(X, 0);
        Sum = Sum ? B.CreateAdd(Sum, X32, "", /*HasNUW=*/true) : X32;
      }
      if (ConstElems)
        Sum = B.CreateAdd(Sum, B.getInt32(uint32_t(ConstElems)), "", true);
      Value *NP = B.CreateGEP(T, P, B.CreateZExt(Sum, B.getInt64Ty()));
      I.setOperand(isa<LoadInst>(I) ? 0 : 1, NP);
      Changed = true;
    }
    return Changed;
  }

  /// gep i8 (gep T, B, zext(t)), C, whose C / sizeof(T) is past the device
  /// accesses' signed 16-bit immediate index (the rows of a 2-D buffer at a
  /// thread's column, which the passes above made the neighbour's address
  /// plus C): gep T, B, zext(t + C / sizeof(T)), the 32-bit add not wrapping
  /// by t's range (with the thread bounds still on), so that the access
  /// takes B and the index register instead of a 64-bit address and add per
  /// row (L15-E291).
  static bool farOffsetIndices(Function &F) {
    using namespace PatternMatch;
    const DataLayout &DL = F.getDataLayout();
    bool Changed = false;
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      auto *G = dyn_cast<GetElementPtrInst>(&I);
      if (!G || G->getNumIndices() != 1 ||
          !G->getSourceElementType()->isIntegerTy(8))
        continue;
      auto *CI = dyn_cast<ConstantInt>(G->getOperand(1));
      auto *P = dyn_cast<GetElementPtrInst>(G->getPointerOperand());
      Value *T;
      if (!CI || !P || P->getNumIndices() != 1 ||
          !match(P->getOperand(1), m_ZExt(m_Value(T))) ||
          !T->getType()->isIntegerTy(32) ||
          G->getPointerAddressSpace() != 1)
        continue;
      TypeSize Size = DL.getTypeAllocSize(P->getSourceElementType());
      if (Size.isScalable() || Size.getFixedValue() == 0)
        continue;
      int64_t S = Size.getFixedValue(), C = CI->getSExtValue();
      if (C % S || isInt<16>(C / S))
        continue;
      int64_t K = C / S;
      ConstantRange R = computeConstantRange(T, /*ForSigned=*/false,
                                             SimplifyQuery(DL, G));
      if (R.isFullSet() ||
          (K > 0 && R.getUnsignedMax().getZExtValue() + uint64_t(K) >
                        uint64_t(UINT32_MAX)) ||
          (K < 0 && R.getUnsignedMin().getZExtValue() < uint64_t(-K)))
        continue;
      IRBuilder<> B(G);
      Value *Idx = B.CreateAdd(T, B.getInt32(uint32_t(K)), "",
                               /*HasNUW=*/K > 0, /*HasNSW=*/false);
      Value *NG = B.CreateGEP(P->getSourceElementType(),
                              P->getPointerOperand(),
                              B.CreateZExt(Idx, P->getOperand(1)->getType()),
                              "", G->getNoWrapFlags() & P->getNoWrapFlags());
      NG->takeName(G);
      G->replaceAllUsesWith(NG);
      G->eraseFromParent();
      Changed = true;
    }
    return Changed;
  }
};
char AGXStraightLineSRPass::ID = 0;
} // namespace

static cl::opt<unsigned> AGXStraightLineSR(
    "agx-straight-line-sr", cl::init(1), cl::Hidden,
    cl::desc("Straight-line strength reduction and n-ary reassociation of "
             "address arithmetic before vectorization (1: then EarlyCSE, "
             "2: then GVN)"));

void AGXPassConfig::addIRPasses() {
  // AIR input becomes AGX kernels first (a no-op without !air.kernel).
  addPass(createAGXLowerAIRPass(&getAGXTargetMachine()));
  // Disjoint address spaces do not alias (for every IR and machine pass that
  // asks alias analysis, the scheduler's memory dependences among them).
  addPass(createAGXAAWrapperPass());
  addPass(createAGXExternalAAWrapperPass());
  // Inline by default; the experimental leaf-call option leaves eligible
  // functions outlined.
  addPass(createAGXAlwaysInlinePass());
  // Internalization can leave entire unused call chains alive: the inliner
  // otherwise expands their bodies before discovering they have no roots.
  if (AGXPruneBeforeInline)
    addPass(createGlobalDCEPass());
  addPass(createAlwaysInlinerLegacyPass());
  // Inlining non-callable wrappers can leave outlined leaves with no users.
  // Removing them before per-function codegen is material for large PTX
  // modules, where even dead device functions may be thousands of blocks.
  if (agxExperimentalLeafCallsEnabled()) {
    addPass(createGlobalDCEPass());
    // Called functions reach the global data buffer through a parameter.
    addPass(createAGXGlobalDataArgsPass());
  }

  // 64- and 128-bit atomics take locks; the others are hardware atomics.
  addPass(createAGXExpandWideAtomicsPass(&getAGXTargetMachine()));
  addPass(createAtomicExpandLegacyPass());
  if (getOptLevel() != CodeGenOptLevel::None) {
    addPass(createInferAddressSpacesPass());
    addPass(createSROAPass());
  }
  // Checked mode: check every access, adding the hidden arguments the
  // runtime fills (before generic pointers and kernel arguments change form).
  addPass(createAGXCheckedAccessesPass(&getAGXTargetMachine()));
  // Generic pointers that may name threadgroup memory become tagged integers.
  addPass(createAGXLowerGenericAddressesPass());
  // Atomics these made threadgroup ones (a generic float add's threadgroup
  // branch, an address InferAddressSpaces resolved), which the first
  // expansion saw as generic or device ones: threadgroup memory has no float
  // atomics (Cycles with a __shared__ BVH stack: AtomicLoadFAdd on
  // addrspace 3 could not be selected).
  addPass(createAtomicExpandLegacyPass());
  // By-reference kernel arguments become uniform register values.
  addPass(createAGXPromoteKernelArgsPass());
  // Neighbouring 32-bit accesses become one vector access (after checked
  // mode, which checks the accesses as the source wrote them); threadgroup
  // and private addresses computed in integers become base + constant first.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // Address arithmetic shared between straight-line computations (the
    // unrolled tiles of agx_libs' GEMMs): each address from a neighbour's
    // plus a difference, and sums reassociated so common parts are computed
    // once, as NVPTX does (straight-line strength reduction and n-ary
    // reassociation).
    if (AGXStraightLineSR)
      addPass(new AGXStraightLineSRPass(TM, AGXStraightLineSR));
    // Loads that all-pairs loops skip for one element issue together.
    addPass(createAGXHoistGuardedLoadsPass());
    addPass(createAGXThreadgroupAddressesPass());
    // Loops that store to one buffer and load from others: a copy for
    // access ranges that do not overlap, whose loads need not wait for the
    // stores before them.
    addPass(createAGXAliasVersioningPass());
    // Loops over buffers only known aligned to 1- or 2-byte elements: a
    // copy for 4-byte aligned bases, whose accesses the vectorizer merges.
    addPass(createAGXAlignmentVersioningPass());
    // Loops with 64-bit products of a loop-invariant value scaled by a
    // constant: a copy for a value that fits 32 bits scaled, whose products
    // are 32 x 32-bit multiply-adds (L15-E670).
    addPass(createAGXWideIndexVersioningPass());
    addPass(createAGXVectorAccessesPass(/*BeforeVectorizer=*/true));
    addPass(createLoadStoreVectorizerPass());
    addPass(createAGXVectorAccessesPass(/*BeforeVectorizer=*/false));
  }
  // Vector arithmetic is scalar code (as in instruction selection), which the
  // software floating point lowerings below then see.
  addPass(createScalarizerPass());
  // The floating point mode register and constrained floating point
  // intrinsics (directed rounding in software).
  addPass(createAGXLowerFPEnvPass());
  addPass(createAGXSoftF64Pass());
  addPass(createAGXCodeGenPreparePass());
  // Software IEEE binary32 arithmetic and canonical NaNs, when enabled.
  addPass(createAGXStrictFPPass());

  TargetPassConfig::addIRPasses();
}

static cl::opt<bool> AGXUnifyExitsAfterSimplifyCFG(
    "agx-unify-exits-after-simplifycfg", cl::init(true), cl::Hidden,
    cl::desc("Unify the exits again after the pre-ISel SimplifyCFG"));

static cl::opt<unsigned> AGXPreISelSimplifyCFG(
    "agx-preisel-simplifycfg", cl::init(2), cl::Hidden,
    cl::desc("SimplifyCFG before structurization with this bonus instruction "
             "threshold (0: not run)"));

bool AGXPassConfig::addPreISel() {
  // The special registers' bounds after loop strength reduction, which
  // would form 64-bit offsets from the bounded indices (EXPERIMENTS.md
  // L14-E1), but before AGXNarrowInductions, which proves counters from
  // thread positions non-negative with them (L15-E224).
  addPass(createAGXSpecialRegisterRangesPass());
  // After loop strength reduction, which makes 64-bit counters of PTX's
  // 64-bit address arithmetic.
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createAGXNarrowInductionsPass());
  // 64-bit sums of zero-extended 32-bit values in 32-bit words (L15-E678).
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createAGXSplitWideAccumulatorsPass());
  // 64-bit byte offsets that remain, of word accesses: a copy with a 32-bit
  // counter of words, the accesses' index register (L15-E674).
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createAGXWordCounterVersioningPass());
  // Word accesses at a counter of bytes or halves indexed by a counter of
  // words, once the counters are 32 bits (L15-E594).
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createAGXWordIndicesPass());
  // The phis of an if's empty side (left by AGXCodeGenPrepare's merging of
  // || and && conditions) become selects, as SimplifyCFG folds two-entry
  // phis: an if is execution mask operations, which cost more than the
  // speculated selects. Without SimplifyCFG's branch simplification, which
  // would make switches of comparison chains.
  if (getOptLevel() != CodeGenOptLevel::None && AGXPreISelSimplifyCFG)
    addPass(createCFGSimplificationPass(
        SimplifyCFGOptions()
            .bonusInstThreshold(AGXPreISelSimplifyCFG)
            .setSimplifyCondBranch(false)));
  // (SimplifyCFG ends the blocks of a llvm.trap, noreturn, with unreachable
  // again: back to branches to the single return, as AGXCodeGenPrepare left
  // them)
  if (getOptLevel() != CodeGenOptLevel::None && AGXPreISelSimplifyCFG &&
      AGXUnifyExitsAfterSimplifyCFG)
    addPass(createAGXUnifyExitsPass());
  // After that SimplifyCFG, which would speculate the fallbacks again.
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createAGXRareFallbacksPass());
  // Bring the CFG into the structured form AGXAnnotateControlFlow expects.
  addPass(createLowerSwitchPass());
  addPass(createFixIrreduciblePass());
  addPass(createUnifyLoopExitsPass());
  // Every branch is lowered to execution mask operations, so uniform
  // regions are structurized too.
  addPass(createStructurizeCFGPass(/*SkipUniformRegions=*/false));
  addPass(createAGXAnnotateControlFlowPass());
  // Before LCSSA, which would put phis between addresses and their uses.
  addPass(createAGXSinkAddressesPass());
  addPass(createLCSSAPass());
  return false;
}

bool AGXPassConfig::addInstSelector() {
  addPass(createAGXISelDag(getAGXTargetMachine()));
  return false;
}

static cl::opt<bool> AGXSinkRematEnabled(
    "agx-enable-sink-remat", cl::init(false), cl::Hidden,
    cl::desc("Give each block that uses a constant from another block its own "
             "move of it, for fewer registers held"));

void AGXPassConfig::addMachineSSAOptimization() {
  // (before MachineLICM and MachineCSE, which hoist and share its constant:
  // L15-E631)
  addPass(createAGXPairedHalfBitopsPass());
  // (a loop's pointer stepped before its accesses: L15-E642)
  addPass(createAGXAdvancePointersPass());
  TargetPassConfig::addMachineSSAOptimization();
  addPass(createAGXUniformConstantsPass());
  addPass(createAGXFoldOperandsPass());
  // MachineCSE and MachineSink may have put code where only the threads of
  // a then side run it.
  addPass(createAGXHoistFromElseFlowPass());
  if (AGXSinkRematEnabled)
    addPass(createAGXSinkRematPass());
  // (a loop's values of a 16-bit value under a zero half carried through it,
  // the zero half set once: L15-E559)
  addPass(createAGXInvariantHalvesPass());
  // (last in SSA form: a load's lanes allocated until its first use)
  addPass(createAGXLoadLanesLivePass());
}

void AGXPassConfig::addPostRegAlloc() {
  // (before frame lowering: a callee-saved register it gives a reload is saved)
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createAGXHoistReloadsPass());
}

void AGXPassConfig::addPreEmitPass() {
  addPass(createAGXLowerControlFlowPass());
  addPass(createAGXInsertWaitsPass());
  // Rarely entered if-bodies out of line (after the waits: paths unchanged).
  addPass(createAGXColdRegionsPass());
}
