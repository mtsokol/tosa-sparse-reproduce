import ctypes
import ctypes.util
import pathlib

import numpy as np

from mlir.ir import Context, Module
import mlir.runtime as rt
from mlir import execution_engine, passmanager


MLIR_C_RUNNER_UTILS = ctypes.util.find_library("mlir_c_runner_utils")


with Context():
    module_add = Module.parse(
    """
    #CSR = #sparse_tensor.encoding<{
        map = (i, j) -> (i : dense, j : compressed), posWidth = 64, crdWidth = 64
    }>
    #Dense = #sparse_tensor.encoding<{
        map = (i, j) -> (i : dense, j : dense), posWidth = 64, crdWidth = 64
    }>

    #map = affine_map<(d0, d1) -> (d0, d1)>
    func.func @add(%st_0 : tensor<3x4xi32, #CSR>, %st_1 : tensor<3x4xi32, #Dense>) attributes { llvm.emit_c_interface } {        
        
        sparse_tensor.print %st_0 : tensor<3x4xi32, #CSR>
        sparse_tensor.print %st_1 : tensor<3x4xi32, #Dense>

        %res = tosa.add %st_0, %st_1 : (tensor<3x4xi32, #CSR>, tensor<3x4xi32, #Dense>) -> tensor<3x4xi32, #CSR>
        sparse_tensor.print %res : tensor<3x4xi32, #CSR>

        return
    }
    """
    )

    CWD = pathlib.Path(".")
    (CWD / "module.mlir").write_text(str(module_add))

    pm = passmanager.PassManager.parse(
        """
        builtin.module(sparse-assembler{direct-out=true}, 
        sparsifier{create-sparse-deallocs=1 enable-runtime-library=false}, 
        func.func(tosa-make-broadcastable))
        """
    )
    pm.run(module_add.operation)

    (CWD / "module_opt.mlir").write_text(str(module_add))

    ee_add = execution_engine.ExecutionEngine(module_add, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])

    pos = np.array([0, 0, 3, 5], dtype=np.int64)
    idx = np.array([0, 1, 3, 2, 3], dtype=np.int64)
    data = np.array([1, 2, 1, 1, 3], dtype=np.int32)
    dense_arr = np.array([[1, 0, 0, 1], [0, 2, 2, 0], [0, 0, 0, 1]], dtype=np.int32)

    p_pos = rt.get_ranked_memref_descriptor(pos)
    p_idx = rt.get_ranked_memref_descriptor(idx)
    p_data = rt.get_ranked_memref_descriptor(data)
    p_dense_arr = rt.get_ranked_memref_descriptor(dense_arr.ravel())

    ret = ee_add.invoke(
        "add",
        # CSR
        ctypes.pointer(ctypes.pointer(p_pos)),
        ctypes.pointer(ctypes.pointer(p_idx)),
        ctypes.pointer(ctypes.pointer(p_data)),
        # Dense
        ctypes.pointer(ctypes.pointer(p_dense_arr)),
    )
