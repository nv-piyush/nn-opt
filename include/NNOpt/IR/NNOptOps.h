#ifndef NNOPT_OPS_H
#define NNOPT_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "NNOpt/IR/NNOptOps.h.inc"

#endif // NNOPT_OPS_H
