#ifndef NNOPT_PASSES_H
#define NNOPT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace nn {

#define GEN_PASS_DECL
#include "NNOpt/Passes/NNOptPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "NNOpt/Passes/NNOptPasses.h.inc"

} // namespace nn

#endif // NNOPT_PASSES_H
