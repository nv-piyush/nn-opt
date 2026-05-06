// ReluSimplify Pass
//
// This pass demonstrates a simple peephole optimization:
// relu(relu(x)) -> relu(x)
//
// ReLU is idempotent: applying it twice produces the same result as once.
// This pattern appears when composing neural network layers or after
// inlining functions.

#include "NNOpt/IR/NNOptOps.h"
#include "NNOpt/Passes/NNOptPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace nn {
#define GEN_PASS_DEF_RELUSIMPLIFY
#include "NNOpt/Passes/NNOptPasses.h.inc"
} // namespace nn

using namespace mlir;

namespace {

// Pattern: relu(relu(x)) -> relu(x)
struct RemoveDoubleRelu : public OpRewritePattern<nn::ReluOp> {
  using OpRewritePattern<nn::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nn::ReluOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the input to this ReLU is another ReLU
    auto inputOp = op.getInput().getDefiningOp<nn::ReluOp>();
    if (!inputOp)
      return failure();

    // Replace relu(relu(x)) with relu(x)
    // We reuse the inner ReLU — its output type already matches.
    rewriter.replaceOp(op, inputOp.getOutput());
    return success();
  }
};

struct ReluSimplifyPass
    : public nn::impl::ReluSimplifyBase<ReluSimplifyPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveDoubleRelu>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

