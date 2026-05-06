// RUN: nn-opt %s --nn-relu-simplify | FileCheck %s

// Test: relu(relu(x)) should be simplified to relu(x)
// CHECK-LABEL: func.func @double_relu
// CHECK:         nn.relu
// CHECK-NOT:     nn.relu
// CHECK:         return
func.func @double_relu(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = nn.relu %arg0 : tensor<2x3xf32> -> tensor<2x3xf32>
  %1 = nn.relu %0 : tensor<2x3xf32> -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// Test: triple relu should also collapse
// CHECK-LABEL: func.func @triple_relu
// CHECK:         nn.relu
// CHECK-NOT:     nn.relu
// CHECK:         return
func.func @triple_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = nn.relu %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>
  %1 = nn.relu %0 : tensor<4x4xf32> -> tensor<4x4xf32>
  %2 = nn.relu %1 : tensor<4x4xf32> -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test: single relu should be left alone
// CHECK-LABEL: func.func @single_relu
// CHECK:         nn.relu
// CHECK:         return
func.func @single_relu(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = nn.relu %arg0 : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
