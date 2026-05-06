add_mlir_library(NNOptPasses
  ReluSimplify.cpp
  ConvBiasFusion.cpp
  MatMulAddFusion.cpp

  DEPENDS
  NNOptPassesIncGen

  LINK_LIBS PUBLIC
  NNOptIR
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
