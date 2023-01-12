# export TVM_HOME=/home/ubuntu/tvm
# export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
# cmake .. -DCUDA_CUDA_LIBRARY=/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so

import numpy as np
import tvm
from tvm import te, auto_scheduler

""" Compute defination to be tuned """
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def my_batch_matmul(Batch, N, L, M, dtype):
    A = te.placeholder((Batch, N, L), name="A", dtype=dtype)
    B = te.placeholder((Batch, M, L), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (Batch, N, M),
        lambda b, i, j: te.sum(A[b, i, k] * B[b, j, k], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )

    return [A, B, matmul]

""" Shape Configs """
Batch = 32
N = M = 4096
L = 64

""" Search Task Configs """
target = tvm.target.Target("cuda")
task = auto_scheduler.SearchTask(func=my_batch_matmul, args=(Batch, N, L, M, "float32"), target=target)

""" Optional: Inspect the computational graph """
print("--- Computational DAG ---")
print(task.compute_dag)

log_file = "./batch_matmul.json"

measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300, device=0, timeout=30)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

""" Run auto-tuning (search) """
task.tune(tune_option)

""" Optional: Print the best schedule """
print("--- Best Schedule ---")
print(task.print_best(log_file))

""" Apply the best schedule """
sch, args = task.apply_best(log_file)

""" Optional: Print TIR in te/tir """
print("--- Lowered TIR ---")
print(tvm.lower(sch, args, simple_mode=True).script())

""" Optional: Print CUDA code """
func = tvm.build(sch, args, target)
print("--- CUDA code ---")
print(func.imported_modules[0].get_source())

""" Var preparation """
a_np = np.random.uniform(size=(Batch, N, L)).astype(np.float32)
b_np = np.random.uniform(size=(Batch, M, L)).astype(np.float32)
out_np = np.random.uniform(size=(Batch, N, M)).astype(np.float32)
out_np = np.einsum("bik,bjk->bij", a_np, b_np)

dev=tvm.cuda(0)
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)

""" Check correctness """
func(a_tvm, b_tvm, out_tvm)
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

""" Performance evaluation """
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=300, number=5, repeat=5)
print(evaluator(a_tvm, b_tvm, out_tvm).__str__())
