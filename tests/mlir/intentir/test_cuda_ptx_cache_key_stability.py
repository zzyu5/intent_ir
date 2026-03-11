from __future__ import annotations

from pipeline.mlir_contract_artifacts import _cuda_ptx_cache_key


def test_cuda_ptx_cache_key_is_stable_and_sensitive() -> None:
    base = dict(
        llvm_ir_text='target triple = "nvptx64-nvidia-cuda"\n; ModuleID = "x"\n',
        llc_path="/usr/bin/llc",
        llc_ver="LLVM version 20.1.0",
        target_sm="sm_89",
        link_libdevice=True,
        libdevice_fingerprint="/usr/local/cuda/nvvm/libdevice/libdevice.10.bc:1:2",
    )
    k1 = _cuda_ptx_cache_key(**base)
    k2 = _cuda_ptx_cache_key(**base)
    assert k1 == k2

    k_text = _cuda_ptx_cache_key(**{**base, "llvm_ir_text": base["llvm_ir_text"] + "define void @f() { ret void }\n"})
    assert k_text != k1

    k_ver = _cuda_ptx_cache_key(**{**base, "llc_ver": "LLVM version 20.1.1"})
    assert k_ver != k1

    k_sm = _cuda_ptx_cache_key(**{**base, "target_sm": "sm_90"})
    assert k_sm != k1

    k_link = _cuda_ptx_cache_key(**{**base, "link_libdevice": False})
    assert k_link != k1
