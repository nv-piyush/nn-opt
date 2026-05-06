set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(nn-opt
  nn-opt.cpp
)

target_link_libraries(nn-opt PRIVATE
  NNOptIR
  NNOptPasses
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIROptLib
  MLIRFuncDialect
  MLIRSupport
)

llvm_update_compile_flags(nn-opt)
