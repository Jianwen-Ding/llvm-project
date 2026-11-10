//===- AIRDialect.cpp - AIR dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::air;

#include "mlir/Dialect/AIR/IR/AIROpsDialect.cpp.inc"

void AIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AIR/IR/AIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AIR/IR/AIROpsAttributes.cpp.inc"
      >();
  declarePromisedInterface<mlir::gpu::TargetAttrInterface,
                           mlir::air::AIRTargetAttr>();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AIR/IR/AIROpsAttributes.cpp.inc"

static LogicalResult verifyMatrixAccess(Operation *op, MemRefType base,
                                        size_t numIndices, VectorType matrix) {
  if (numIndices != 2)
    return op->emitOpError("takes two indices");
  if (base.getElementType() != matrix.getElementType())
    return op->emitOpError("matrix and memref element types differ");
  return success();
}

LogicalResult MatrixLoadOp::verify() {
  return verifyMatrixAccess(*this, getBase().getType(), getIndices().size(),
                            getResult().getType());
}

LogicalResult MatrixStoreOp::verify() {
  return verifyMatrixAccess(*this, getBase().getType(), getIndices().size(),
                            getValue().getType());
}

LogicalResult MatrixMultiplyAccumulateOp::verify() {
  Type ab = getA().getType().getElementType();
  Type cd = getC().getType().getElementType();
  if (ab != cd && !((ab.isF16() || ab.isBF16()) && cd.isF32()))
    return emitOpError("multiplies ")
           << ab << " matrices into " << cd
           << " (only the same type, or f16 or bf16 into f32)";
  return success();
}

/// `: T` when A, B, C and D all have type T, else `: TA, TC`.
static ParseResult parseMatrixMMATypes(OpAsmParser &parser, Type &a, Type &c) {
  if (parser.parseType(a))
    return failure();
  c = a;
  if (succeeded(parser.parseOptionalComma()))
    return parser.parseType(c);
  return success();
}

static void printMatrixMMATypes(OpAsmPrinter &printer, Operation *, Type a,
                                Type c) {
  printer << a;
  if (a != c)
    printer << ", " << c;
}

namespace {
/// An element-wise operation (arith, math: ElementwiseMappable) whose matrix
/// operands are all fills is the fill of the operation on their values,
/// since every element of a fill is its value: one scalar operation and one
/// fill instead of an operation on the thread's elements of each matrix (and
/// a fill per operand). For example
///   %a = air.simdgroup_matrix_fill %x : vector<64xf32>
///   %b = air.simdgroup_matrix_fill %y : vector<64xf32>
///   %c = arith.addf %a, %b : vector<64xf32>
/// is
///   %s = arith.addf %x, %y : f32
///   %c = air.simdgroup_matrix_fill %s : vector<64xf32>
/// and the same for arith.extf / arith.truncf of a fill.
struct ElementwiseOfFills : RewritePattern {
  ElementwiseOfFills(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<OpTrait::Elementwise>() || op->getNumResults() != 1 ||
        op->getNumRegions() != 0 || isa<MatrixFillOp>(op))
      return failure();
    auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!resultType || resultType.getShape() != ArrayRef<int64_t>{64} ||
        resultType.isScalable() ||
        !isa<Float16Type, BFloat16Type, Float32Type>(
            resultType.getElementType()))
      return failure();
    SmallVector<Value> scalars;
    bool anyFill = false;
    for (Value operand : op->getOperands()) {
      if (auto fill = operand.getDefiningOp<MatrixFillOp>()) {
        scalars.push_back(fill.getValue());
        anyFill = true;
      } else if (!isa<ShapedType>(operand.getType())) {
        scalars.push_back(operand); // a scalar operand (a select's condition)
      } else {
        return failure();
      }
    }
    if (!anyFill)
      return failure();
    OperationState state(op->getLoc(), op->getName(), scalars,
                         {resultType.getElementType()}, op->getAttrs());
    Operation *scalar = rewriter.create(state);
    rewriter.replaceOpWithNewOp<MatrixFillOp>(op, resultType,
                                              scalar->getResult(0));
    return success();
  }
};
} // namespace

void AIRDialect::getCanonicalizationPatterns(RewritePatternSet &results) const {
  results.add<ElementwiseOfFills>(getContext());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/AIR/IR/AIROps.cpp.inc"
