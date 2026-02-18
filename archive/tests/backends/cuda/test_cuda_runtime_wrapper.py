from __future__ import annotations

import numpy as np


def test_backend_runtime_forwards_compiled_module(monkeypatch) -> None:
    import backends.cuda.runtime as rt

    sentinel_module = object()
    seen: dict[str, object] = {}

    def _fake_run_cuda_kernel_io(**kwargs):
        seen["compiled_module"] = kwargs.get("compiled_module")
        return {"out": np.asarray([1.0], dtype=np.float32)}

    monkeypatch.setattr(rt, "run_cuda_kernel_io", _fake_run_cuda_kernel_io)
    out = rt.run_cuda_kernel(
        kernel_name="k",
        cuda_src='extern "C" __global__ void k() {}',
        io_spec={"arg_names": [], "tensors": {}, "scalars": {}},
        launch=rt.CudaLaunch(grid=(1, 1, 1), block=(1, 1, 1)),
        bindings={},
        inputs_np={},
        output_names=[],
        compiled_module=sentinel_module,
    )
    assert seen["compiled_module"] is sentinel_module
    assert out["out"].shape == (1,)
