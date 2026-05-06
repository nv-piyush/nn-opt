// RUN: nn-opt %s --nn-conv-bias-fusion | FileCheck %s

// Test: conv2d + add -> conv2d_bias
// CHECK-LABEL: func.func @fuse_conv_bias
// CHECK:         nn.conv2d_bias
// CHECK-NOT:     nn.conv2d
// CHECK-NOT:     nn.add
// CHECK:         return
func.func @fuse_conv_bias(
    %input: tensor<1x3x28x28xf32>,
    %filter: tensor<16x3x3x3xf32>,
    %bias: tensor<16x1x1xf32>) -> tensor<1x16x26x26xf32> {
  %conv = nn.conv2d %input, %filter : tensor<1x3x28x28xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x26x26xf32>
  %result = nn.add %conv, %bias : tensor<1x16x26x26xf32>, tensor<16x1x1xf32> -> tensor<1x16x26x26xf32>
  return %result : tensor<1x16x26x26xf32>
}

// Test: conv with multiple uses should NOT be fused
// CHECK-LABEL: func.func @no_fuse_multi_use
// CHECK:         nn.conv2d
// CHECK:         nn.add
// CHECK-NOT:     nn.conv2d_bias
func.func @no_fuse_multi_use(
    %input: tensor<1x3x28x28xf32>,
    %filter: tensor<16x3x3x3xf32>,
    %bias: tensor<16x1x1xf32>) -> (tensor<1x16x26x26xf32>, tensor<1x16x26x26xf32>) {
  %conv = nn.conv2d %input, %filter : tensor<1x3x28x28xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x26x26xf32>
  %result = nn.add %conv, %bias : tensor<1x16x26x26xf32>, tensor<16x1x1xf32> -> tensor<1x16x26x26xf32>
  return %conv, %result : tensor<1x16x26x26xf32>, tensor<1x16x26x26xf32>
}
