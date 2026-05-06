// NNOpt Dialect definition
// This dialect models common neural network operations for learning
// how to write MLIR optimization passes.

#ifndef NNOPT_DIALECT
#define NNOPT_DIALECT

include "mlir/IR/OpBase.td"

def NNOpt_Dialect : Dialect {
  let name = "nn";
  let summary = "A dialect for neural network operations and optimizations";
  let cppNamespace = "::nn";
}

#endif // NNOPT_DIALECT
