// ConvBiasFusion Pass
//
// This pass demonstrates operator fusion, one of the most important
// optimizations in neural network compilers.
//
// Pattern: add(conv2d(input, filter), bias) -> conv2d_bias(input, filter, bias)
//
// Why this matters:
// - Reduces memory traffic (no intermediate tensor between conv and add)
// - Enables hardware-specific fused kernels (cuDNN, oneDNN)
// - This is exactly what TensorRT, XLA, and TVM do

#include "NNOpt/IR/NNOptOps.h"
#include "NNOpt/Passes/NNOptPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace nn {
#define GEN_PASS_DEF_CONVBIASFUSION
#include "NNOpt/Passes/NNOptPasses.h.inc"
} // namespace nn

using namespace mlir;

namespace {

struct FuseConvBias : public OpRewritePattern<nn::AddOp> {
  using OpRewritePattern<nn::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nn::AddOp op,
                                PatternRewriter &rewriter) const override {
    // Check if either operand of the Add is a Conv2D
    auto lhsConv = op.getLhs().getDefiningOp<nn::Conv2DOp>();
    auto rhsConv = op.getRhs().getDefiningOp<nn::Conv2DOp>();

    nn::Conv2DOp convOp;
    Value bias;

    if (lhsConv) {
      convOp = lhsConv;
      bias = op.getRhs();
    } else if (rhsConv) {
      convOp = rhsConv;
      bias = op.getLhs();
    } else {
      return failure();
    }

    // Only fuse if the conv result is used only by this add
    if (!convOp.getOutput().hasOneUse())
      return failure();

    // Create the fused conv2d_bias op
    rewriter.replaceOpWithNewOp<nn::Conv2DBiasOp>(
        op, op.getOutput().getType(), convOp.getInput(), convOp.getFilter(),
        bias, convOp.getStrides(), convOp.getPadding());

    // Remove the now-dead conv op
    rewriter.eraseOp(convOp);

    return success();
  }
};

struct ConvBiasFusionPass
    : public nn::impl::ConvBiasFusionBase<ConvBiasFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConvBias>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

