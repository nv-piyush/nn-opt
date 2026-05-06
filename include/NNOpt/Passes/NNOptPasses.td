// Pass definitions for NNOpt dialect optimizations.

#ifndef NNOPT_PASSES
#define NNOPT_PASSES

include "mlir/Pass/PassBase.td"

def ReluSimplify : Pass<"nn-relu-simplify"> {
  let summary = "Simplify redundant ReLU operations";
  let description = [{
    This pass simplifies chains of ReLU activations.
    Since ReLU is idempotent (relu(relu(x)) == relu(x)),
    nested ReLU calls can be collapsed.
  }];
}

def ConvBiasFusion : Pass<"nn-conv-bias-fusion"> {
  let summary = "Fuse Conv2D + Add into Conv2DBias";
  let description = [{
    This pass detects patterns where a Conv2D is followed by
    a bias addition and fuses them into a single Conv2DBias op.
    This is a common optimization in neural network compilers
    as it reduces memory traffic and enables kernel fusion.
  }];
}

def MatMulAddFusion : Pass<"nn-matmul-add-fusion"> {
  let summary = "Fuse MatMul + Add into Linear";
  let description = [{
    Detects matmul followed by bias addition and fuses them
    into a single Linear operation.
  }];
}

#endif // NNOPT_PASSES
