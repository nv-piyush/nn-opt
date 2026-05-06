#include "NNOpt/IR/NNOptDialect.h"
#include "NNOpt/IR/NNOptOps.h"

using namespace mlir;

#include "NNOpt/IR/NNOptDialect.cpp.inc"

void nn::NNOptDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NNOpt/IR/NNOptOps.cpp.inc"
      >();
}
