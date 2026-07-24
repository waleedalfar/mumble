"""Tests for CUDA-capability detection from a whisper-server startup log."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gpu_check import MIN_VRAM_MIB, detect_cuda_capable


def _write(tmp_path, text):
    path = tmp_path / "whisper-server.log"
    path.write_text(text, encoding="utf-8")
    return path


class TestDetectCudaCapable:
    def test_real_gpu_log_format_is_capable(self, tmp_path):
        log = _write(tmp_path, "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):\n"
                               "  Device 0: NVIDIA GeForce RTX 5070 Ti\n")
        capable, reason = detect_cuda_capable(log)
        assert capable is True
        assert "16302" in reason

    def test_cpu_only_build_is_not_capable(self, tmp_path):
        log = _write(tmp_path, "whisper_model_load: loading model\nsystem_info: n_threads = 4\n")
        capable, reason = detect_cuda_capable(log)
        assert capable is False
        assert "no CUDA backend" in reason

    def test_low_vram_gpu_is_not_capable(self, tmp_path):
        log = _write(tmp_path, f"ggml_cuda_init: found 1 CUDA devices (Total VRAM: {MIN_VRAM_MIB - 1} MiB):\n")
        capable, reason = detect_cuda_capable(log)
        assert capable is False
        assert "VRAM" in reason

    def test_vram_exactly_at_threshold_is_capable(self, tmp_path):
        log = _write(tmp_path, f"ggml_cuda_init: found 1 CUDA devices (Total VRAM: {MIN_VRAM_MIB} MiB):\n")
        capable, _ = detect_cuda_capable(log)
        assert capable is True

    def test_missing_log_file_is_not_capable(self, tmp_path):
        capable, reason = detect_cuda_capable(tmp_path / "does_not_exist.log")
        assert capable is False
        assert "could not read" in reason
