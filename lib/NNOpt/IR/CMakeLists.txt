add_mlir_library(NNOptIR
  NNOptDialect.cpp
  NNOptOps.cpp

  DEPENDS
  NNOptOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
