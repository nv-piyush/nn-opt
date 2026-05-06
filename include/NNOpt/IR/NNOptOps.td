// NNOpt Operations
// Defines the operations in the NNOpt dialect.

#ifndef NNOPT_OPS
#define NNOPT_OPS

include "NNOpt/IR/NNOptDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class NNOpt_Op<string mnemonic, list<Trait> traits = []> :
    Op<NNOpt_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Activation functions
//===----------------------------------------------------------------------===//

def NNOpt_ReluOp : NNOpt_Op<"relu", [Pure]> {
  let summary = "Rectified Linear Unit activation";
  let description = [{
    Applies ReLU: output = max(0, input).
    Operates element-wise on tensors.
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def NNOpt_SigmoidOp : NNOpt_Op<"sigmoid", [Pure]> {
  let summary = "Sigmoid activation function";
  let description = [{
    Applies sigmoid: output = 1 / (1 + exp(-input)).
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Linear algebra operations
//===----------------------------------------------------------------------===//

def NNOpt_MatMulOp : NNOpt_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication: output = lhs @ rhs.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

def NNOpt_AddOp : NNOpt_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition (with broadcast)";
  let description = [{
    Element-wise addition of two tensors, supporting broadcast for bias add.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Convolution
//===----------------------------------------------------------------------===//

def NNOpt_Conv2DOp : NNOpt_Op<"conv2d", [Pure]> {
  let summary = "2D convolution";
  let description = [{
    Applies a 2D convolution over an input tensor using the given filter.
    Input shape: [N, C_in, H, W]
    Filter shape: [C_out, C_in, kH, kW]
    Output shape: [N, C_out, H', W']
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{0, 0}">:$padding
  );
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Fused operations (targets for fusion passes)
//===----------------------------------------------------------------------===//

def NNOpt_Conv2DBiasOp : NNOpt_Op<"conv2d_bias", [Pure]> {
  let summary = "Fused 2D convolution + bias add";
  let description = [{
    Fused conv2d + bias addition. This is a target for the Conv+Bias
    fusion optimization pass.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$bias,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{0, 0}">:$padding
  );
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $filter `,` $bias attr-dict `:` type($input) `,` type($filter) `,` type($bias) `->` type($output)
  }];
}

def NNOpt_LinearOp : NNOpt_Op<"linear", [Pure]> {
  let summary = "Fused linear layer (matmul + bias)";
  let description = [{
    Fused matmul + bias: output = input @ weight + bias.
    This is the target for matmul+add fusion.
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$weight, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict `:` type($input) `,` type($weight) `,` type($bias) `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

def NNOpt_ConstantOp : NNOpt_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Neural network tensor constant";
  let description = [{
    Produces a constant tensor value.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  let hasFolder = 1;
}

#endif // NNOPT_OPS
