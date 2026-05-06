// nn-opt: A standalone MLIR optimizer for the NNOpt dialect.
//
// Usage:
//   nn-opt input.mlir --nn-relu-simplify
//   nn-opt input.mlir --nn-conv-bias-fusion
//   nn-opt input.mlir --nn-matmul-add-fusion

#include "NNOpt/IR/NNOptDialect.h"
#include "NNOpt/IR/NNOptOps.h"
#include "NNOpt/Passes/NNOptPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<nn::NNOptDialect>();
  registry.insert<mlir::func::FuncDialect>();

  nn::registerPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NNOpt optimizer driver\n", registry));
}
