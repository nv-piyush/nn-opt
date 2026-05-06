// MatMulAddFusion Pass
//
// Fuses matmul + add into a single linear (fully-connected) operation.
// Pattern: add(matmul(input, weight), bias) -> linear(input, weight, bias)
//
// This is the dense-layer fusion that every NN compiler implements.
// A fully-connected layer is: output = input @ weight + bias
// Hardware libraries (cuBLAS, MKL) provide fused GEMM+bias kernels.

#include "NNOpt/IR/NNOptOps.h"
#include "NNOpt/Passes/NNOptPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace nn {
#define GEN_PASS_DEF_MATMULADDFUSION
#include "NNOpt/Passes/NNOptPasses.h.inc"
} // namespace nn

using namespace mlir;

namespace {

struct FuseMatMulAdd : public OpRewritePattern<nn::AddOp> {
  using OpRewritePattern<nn::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nn::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsMatMul = op.getLhs().getDefiningOp<nn::MatMulOp>();
    auto rhsMatMul = op.getRhs().getDefiningOp<nn::MatMulOp>();

    nn::MatMulOp matmulOp;
    Value bias;

    if (lhsMatMul) {
      matmulOp = lhsMatMul;
      bias = op.getRhs();
    } else if (rhsMatMul) {
      matmulOp = rhsMatMul;
      bias = op.getLhs();
    } else {
      return failure();
    }

    if (!matmulOp.getOutput().hasOneUse())
      return failure();

    rewriter.replaceOpWithNewOp<nn::LinearOp>(
        op, op.getOutput().getType(), matmulOp.getLhs(), matmulOp.getRhs(),
        bias);

    rewriter.eraseOp(matmulOp);

    return success();
  }
};

struct MatMulAddFusionPass
    : public nn::impl::MatMulAddFusionBase<MatMulAddFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatMulAdd>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

