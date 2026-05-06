# nn-opt

An out-of-tree [MLIR](https://mlir.llvm.org/) project that implements optimization passes for neural network computation. Built as a learning project to understand how modern ML compilers work under the hood.

## What This Project Does

Neural network compilers like TensorRT, XLA, and TVM optimize models before running them on hardware. This project implements a subset of those optimizations as MLIR passes, using a custom **`nn` dialect** that models common neural network operations.

```
 Input (unoptimized)                    Output (optimized)
 ─────────────────────                  ────────────────────
 %0 = nn.conv2d %in, %f                %0 = nn.conv2d_bias %in, %f, %b
 %1 = nn.add %0, %b       ──pass──▶    return %0
 return %1
```

## The `nn` Dialect

| Operation | Description | Example |
|-----------|-------------|---------|
| `nn.relu` | ReLU activation: max(0, x) | `%0 = nn.relu %x` |
| `nn.sigmoid` | Sigmoid activation | `%0 = nn.sigmoid %x` |
| `nn.matmul` | Matrix multiplication | `%0 = nn.matmul %a, %b` |
| `nn.add` | Element-wise addition (with broadcast) | `%0 = nn.add %a, %b` |
| `nn.conv2d` | 2D convolution | `%0 = nn.conv2d %in, %filter` |
| `nn.constant` | Constant tensor | `%0 = nn.constant dense<0.0>` |
| `nn.conv2d_bias` | Fused conv2d + bias (optimization target) | `%0 = nn.conv2d_bias %in, %f, %b` |
| `nn.linear` | Fused matmul + bias (optimization target) | `%0 = nn.linear %in, %w, %b` |

## Optimization Passes

### 1. ReLU Simplification (`--nn-relu-simplify`)

ReLU is idempotent: since `relu(x) = max(0, x)`, applying it twice changes nothing. This pass collapses redundant chains.

```
 Before                          After
 ──────                          ─────
 %0 = nn.relu %arg0              %0 = nn.relu %arg0
 %1 = nn.relu %0     ───▶       return %0
 return %1
```

**Source:** [`lib/NNOpt/Passes/ReluSimplify.cpp`](lib/NNOpt/Passes/ReluSimplify.cpp)

### 2. Conv + Bias Fusion (`--nn-conv-bias-fusion`)

Detects a `conv2d` followed by a bias `add` and fuses them into a single `conv2d_bias` op. This is one of the most impactful optimizations in real NN compilers — it eliminates an intermediate tensor and enables hardware-specific fused kernels (cuDNN, oneDNN).

```
 Before                                After
 ──────                                ─────
 %0 = nn.conv2d %in, %filter           %0 = nn.conv2d_bias %in, %filter, %bias
 %1 = nn.add %0, %bias      ───▶      return %0
 return %1
```

Safety: only fuses when the conv result has a single use (otherwise the intermediate tensor is still needed).

**Source:** [`lib/NNOpt/Passes/ConvBiasFusion.cpp`](lib/NNOpt/Passes/ConvBiasFusion.cpp)

### 3. MatMul + Add Fusion (`--nn-matmul-add-fusion`)

Fuses a `matmul` followed by a bias `add` into a single `linear` (fully-connected layer) op. Hardware libraries like cuBLAS and MKL provide fused GEMM+bias kernels that are faster than separate calls.

```
 Before                                After
 ──────                                ─────
 %mm = nn.matmul %in, %weight          %0 = nn.linear %in, %weight, %bias
 %fc = nn.add %mm, %bias    ───▶      return %0
 return %fc
```

**Source:** [`lib/NNOpt/Passes/MatMulAddFusion.cpp`](lib/NNOpt/Passes/MatMulAddFusion.cpp)

## Building

### Prerequisites

- CMake >= 3.20
- Ninja
- LLVM/MLIR (installed via Homebrew or built from source)

```bash
# macOS
brew install cmake ninja llvm
```

### Build

```bash
cmake -G Ninja -B build \
  -DCMAKE_PREFIX_PATH="$(brew --prefix llvm)" \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DMLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir" \
  -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"

ninja -C build
```

The `nn-opt` binary will be at `build/tools/nn-opt`.

## Usage

```bash
# Run a single pass
./build/tools/nn-opt test/NNOpt/relu_simplify.mlir --nn-relu-simplify

# Chain multiple passes
./build/tools/nn-opt input.mlir --nn-matmul-add-fusion --nn-relu-simplify

# List all available passes
./build/tools/nn-opt --help
```

### Example: Optimizing a Two-Layer Network

Input (`test/NNOpt/matmul_add_fusion.mlir`):
```mlir
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
```

After `--nn-matmul-add-fusion`:
```mlir
func.func @two_linear_layers(...) -> tensor<32x10xf32> {
  %0 = nn.linear %input, %w1, %b1 : ... -> tensor<32x128xf32>
  %1 = nn.relu %0 : tensor<32x128xf32> -> tensor<32x128xf32>
  %2 = nn.linear %1, %w2, %b2 : ... -> tensor<32x10xf32>
  return %2 : tensor<32x10xf32>
}
```

Both `matmul + add` pairs are fused into `nn.linear` ops.

## Project Structure

```
nn-opt/
├── CMakeLists.txt                        # Top-level build config
├── include/NNOpt/
│   ├── IR/
│   │   ├── NNOptDialect.td               # Dialect definition (TableGen)
│   │   ├── NNOptOps.td                   # Operation definitions (TableGen)
│   │   ├── NNOptDialect.h                # Dialect C++ header
│   │   └── NNOptOps.h                    # Ops C++ header
│   └── Passes/
│       ├── NNOptPasses.td                # Pass declarations (TableGen)
│       └── NNOptPasses.h                 # Pass C++ header
├── lib/NNOpt/
│   ├── IR/
│   │   ├── NNOptDialect.cpp              # Dialect registration
│   │   └── NNOptOps.cpp                  # Op implementations (constant folding)
│   └── Passes/
│       ├── ReluSimplify.cpp              # relu(relu(x)) -> relu(x)
│       ├── ConvBiasFusion.cpp            # conv2d + add -> conv2d_bias
│       └── MatMulAddFusion.cpp           # matmul + add -> linear
├── tools/
│   └── nn-opt.cpp                        # Main driver executable
└── test/NNOpt/
    ├── relu_simplify.mlir                # ReLU pass tests
    ├── conv_bias_fusion.mlir             # Conv fusion tests
    └── matmul_add_fusion.mlir            # MatMul fusion tests
```

## How It Works

Each pass follows the same MLIR pattern:

1. **Define a pattern** — a struct that inherits `OpRewritePattern<SomeOp>` and implements `matchAndRewrite()`
2. **Match** — inspect the IR to see if the optimization applies (e.g., "is this Add's input a Conv2D?")
3. **Rewrite** — replace the matched ops with the optimized version using `PatternRewriter`
4. **Register with the greedy driver** — `applyPatternsGreedily` runs all patterns to a fixpoint

```
 Pattern applied repeatedly until no more matches (fixpoint):

   relu(relu(relu(x)))
        │ iteration 1: remove outermost
        ▼
   relu(relu(x))
        │ iteration 2: remove outermost
        ▼
   relu(x)
        │ iteration 3: no match → done
        ▼
   relu(x)  ✓
```

## MLIR Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Defining a custom dialect | `include/NNOpt/IR/NNOptDialect.td` |
| Defining operations with TableGen | `include/NNOpt/IR/NNOptOps.td` |
| Op traits (`Pure`, `Commutative`, `ConstantLike`) | `include/NNOpt/IR/NNOptOps.td` |
| Constant folding (`hasFolder`) | `lib/NNOpt/IR/NNOptOps.cpp` |
| Pattern-based rewriting | `lib/NNOpt/Passes/ReluSimplify.cpp` |
| Operator fusion | `lib/NNOpt/Passes/ConvBiasFusion.cpp` |
| Greedy pattern rewrite driver | All pass files |
| Pass declaration with TableGen | `include/NNOpt/Passes/NNOptPasses.td` |
| Out-of-tree MLIR project structure | `CMakeLists.txt` |
| Building a custom `mlir-opt` tool | `tools/nn-opt.cpp` |

## License

MIT
