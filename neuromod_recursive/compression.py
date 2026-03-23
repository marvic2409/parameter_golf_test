"""Quantization + compression utilities matching the Parameter Golf format.

Mirrors the int8 quantization + zlib compression from train_gpt.py.
Used during search to evaluate architectures AFTER compression,
so we never discover an architecture that breaks when quantized.
"""

from __future__ import annotations

import io
import zlib

import torch
import torch.nn as nn
from torch import Tensor


# Thresholds matching train_gpt.py
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

# Tensor name patterns that should stay in float (control signals)
CONTROL_PATTERNS = (
    "inhibition_gain", "amplitude", "frequency", "phase",  # our custom control params
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "skip_weight", "skip_weights",
)


def quantize_state_dict_int8(state_dict: dict[str, Tensor]) -> tuple[dict, dict]:
    """Quantize a state dict to int8, matching the challenge format.

    Returns (quantized_obj, stats).
    """
    quantized = {}
    scales = {}
    dtypes = {}
    passthrough = {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    stats = {
        "param_count": 0,
        "num_tensors": 0,
        "baseline_bytes": 0,
        "compressed_bytes": 0,
    }

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_bytes"] += t.numel() * t.element_size()

        # Non-float: passthrough
        if not t.is_floating_point():
            passthrough[name] = t
            continue

        # Small tensors or control tensors: keep as fp16
        is_control = any(p in name for p in CONTROL_PATTERNS)
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or is_control:
            if is_control:
                kept = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            else:
                kept = t
            passthrough[name] = kept
            continue

        # Quantize to int8
        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty(t32.shape[0])
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
            scales[name] = scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
            scales[name] = scale

        quantized[name] = q
        dtypes[name] = str(t.dtype).removeprefix("torch.")

    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes

    return obj, stats


def dequantize_state_dict_int8(obj: dict) -> dict[str, Tensor]:
    """Dequantize int8 state dict back to float."""
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        if meta.get("scheme") == "per_row":
            out[name] = (q.float() * s.float().unsqueeze(1)).to(dtype)
        else:
            out[name] = (q.float() * s.float()).to(dtype)

    for name, t in obj["passthrough"].items():
        if name in passthrough_orig_dtypes:
            orig_dtype = getattr(torch, passthrough_orig_dtypes[name])
            out[name] = t.to(dtype=orig_dtype)
        else:
            out[name] = t

    return out


def measure_compressed_size(model: nn.Module) -> dict:
    """Quantize + zlib compress a model and report sizes.

    This is what the challenge actually measures.
    """
    state_dict = model.state_dict()
    quant_obj, stats = quantize_state_dict_int8(state_dict)

    # Serialize and compress
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw_bytes = buf.getvalue()
    compressed = zlib.compress(raw_bytes, level=9)

    stats["raw_quant_bytes"] = len(raw_bytes)
    stats["zlib_compressed_bytes"] = len(compressed)
    stats["under_16mb"] = len(compressed) < 16_000_000

    return stats


def quantize_roundtrip_eval(
    model: nn.Module,
    eval_fn,
    **eval_kwargs,
) -> tuple[dict, dict]:
    """Quantize model, reload, and evaluate — tests if architecture survives compression.

    Args:
        model: The trained model
        eval_fn: Evaluation function (e.g., eval_fineweb_bpb)
        **eval_kwargs: Arguments to pass to eval_fn

    Returns:
        (pre_quant_result, post_quant_result) — compare these to measure quantization degradation.
    """
    # Pre-quantization eval
    pre_result = eval_fn(model=model, **eval_kwargs)

    # Quantize and dequantize
    state_dict = model.state_dict()
    quant_obj, _ = quantize_state_dict_int8(state_dict)
    dequant_state = dequantize_state_dict_int8(quant_obj)

    # Load dequantized weights
    model.load_state_dict(dequant_state)

    # Post-quantization eval
    post_result = eval_fn(model=model, **eval_kwargs)

    return pre_result, post_result
