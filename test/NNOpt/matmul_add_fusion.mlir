// RUN: nn-opt %s --nn-matmul-add-fusion | FileCheck %s

// Test: matmul + add -> linear
// CHECK-LABEL: func.func @fuse_matmul_add
// CHECK:         nn.linear
// CHECK-NOT:     nn.matmul
// CHECK-NOT:     nn.add
// CHECK:         return
func.func @fuse_matmul_add(
    %input: tensor<32x784xf32>,
    %weight: tensor<784x128xf32>,
    %bias: tensor<128xf32>) -> tensor<32x128xf32> {
  %mm = nn.matmul %input, %weight : tensor<32x784xf32>, tensor<784x128xf32> -> tensor<32x128xf32>
  %result = nn.add %mm, %bias : tensor<32x128xf32>, tensor<128xf32> -> tensor<32x128xf32>
  return %result : tensor<32x128xf32>
}

// Test: chained linear layers (both should fuse)
// CHECK-LABEL: func.func @two_linear_layers
// CHECK:         nn.linear
// CHECK:         nn.relu
// CHECK:         nn.linear
// CHECK-NOT:     nn.matmul
// CHECK:         return
func.func @two_linear_layers(
    %input: tensor<32x784xf32>,
    %w1: tensor<784x128xf32>, %b1: tensor<128xf32>,
    %w2: tensor<128x10xf32>, %b2: tensor<10xf32>) -> tensor<32x10xf32> {
  %mm1 = nn.matmul %input, %w1 : tensor<32x784xf32>, tensor<784x128xf32> -> tensor<32x128xf32>
  %fc1 = nn.add %mm1, %b1 : tensor<32x128xf32>, tensor<128xf32> -> tensor<32x128xf32>
  %a1 = nn.relu %fc1 : tensor<32x128xf32> -> tensor<32x128xf32>
  %mm2 = nn.matmul %a1, %w2 : tensor<32x128xf32>, tensor<128x10xf32> -> tensor<32x10xf32>
  %fc2 = nn.add %mm2, %b2 : tensor<32x10xf32>, tensor<10xf32> -> tensor<32x10xf32>
  return %fc2 : tensor<32x10xf32>
}
