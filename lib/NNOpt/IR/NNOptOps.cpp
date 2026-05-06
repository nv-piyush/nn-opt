#include "NNOpt/IR/NNOptOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

// ConstantOp folder: returns the constant value attribute directly.
OpFoldResult nn::ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

#define GET_OP_CLASSES
#include "NNOpt/IR/NNOptOps.cpp.inc"
