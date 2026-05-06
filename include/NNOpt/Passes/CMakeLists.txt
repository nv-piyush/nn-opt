set(LLVM_TARGET_DEFINITIONS NNOptPasses.td)
mlir_tablegen(NNOptPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(NNOptPassesIncGen)
