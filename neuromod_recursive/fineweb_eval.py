"""FineWeb BPB evaluation — matches the official Parameter Golf scoring.

The challenge scores models on bits-per-byte (BPB) of the FineWeb validation set,
computed in a tokenizer-agnostic way:
  BPB = (val_loss / ln(2)) * (tokens / bytes)

This module loads the actual FineWeb shards and computes BPB identically to train_gpt.py.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

# --- Data loading (mirrors train_gpt.py) ---

def load_data_shard(file: Path) -> Tensor:
    """Load a .bin shard using the official 256-int32 challenge header."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    """Load all validation shards into a single contiguous tensor."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[:usable + 1]


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build lookup tables for BPB computation (byte counts per token)."""
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


# --- Token stream for training on real data ---

class TokenStream:
    """Sequential shard reader that wraps around forever."""

    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    """Per-rank loader that consumes one shared token stream without duplicating spans."""

    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_size: int, seq_len: int) -> tuple[Tensor, Tensor]:
        local_tokens = batch_size * seq_len
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(device=self.device, dtype=torch.long)
        x = local[:-1].reshape(batch_size, seq_len)
        y = local[1:].reshape(batch_size, seq_len)
        return x, y


# --- BPB evaluation (matches train_gpt.py exactly) ---

@torch.no_grad()
def eval_fineweb_bpb(
    model: torch.nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    vocab_size: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Evaluate a model on FineWeb validation tokens.

    The model's forward pass must accept (input_ids) and return (logits, details).
    For models without a details dict, wrap them accordingly.

    Returns:
        (val_loss, val_bpb) — matching the official challenge metrics.
    """
    model.eval()
    total_seqs = (val_tokens.numel() - 1) // seq_len
    val_loss_sum = 0.0
    val_token_count = 0.0
    val_byte_count = 0.0

    for batch_start in range(0, total_seqs, batch_size):
        batch_end = min(batch_start + batch_size, total_seqs)
        actual_batch = batch_end - batch_start

        raw_start = batch_start * seq_len
        raw_end = batch_end * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.long)

        x = local[:-1].reshape(actual_batch, seq_len)
        y = local[1:].reshape(actual_batch, seq_len)

        # Forward pass — handle both our model (returns logits, details) and standard models
        output = model(x)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Cross-entropy loss (sum, not mean, for proper averaging later)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="sum",
        )

        batch_token_count = float(y.numel())
        val_loss_sum += loss.item()
        val_token_count += batch_token_count

        # Byte counting for BPB
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        token_bytes += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).to(dtype=torch.int16)
        val_byte_count += token_bytes.to(torch.float64).sum().item()

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / val_byte_count
    val_bpb = bits_per_token * tokens_per_byte

    model.train()
    return float(val_loss), float(val_bpb)


def setup_fineweb_eval(
    data_path: str = "./data/datasets/fineweb10B_sp1024",
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model",
    vocab_size: int = 1024,
    seq_len: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load everything needed for FineWeb BPB evaluation.

    Returns a dict with all the components needed to call eval_fineweb_bpb().
    """
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
    )
    val_pattern = os.path.join(data_path, "fineweb_val_*.bin")
    train_pattern = os.path.join(data_path, "fineweb_train_*.bin")

    print(f"Loading validation tokens from {val_pattern}...")
    val_tokens = load_validation_tokens(val_pattern, seq_len)
    print(f"  Loaded {val_tokens.numel() - 1:,} validation tokens")

    print(f"Loading tokenizer from {tokenizer_path}...")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, vocab_size, device
    )

    return {
        "dataset_name": dataset_name,
        "actual_train_files": actual_train_files,
        "expected_train_files": expected_train_files,
        "val_tokens": val_tokens,
        "train_pattern": train_pattern,
        "tokenizer": sp,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "base_bytes_lut": base_bytes_lut,
        "has_leading_space_lut": has_leading_space_lut,
        "is_boundary_token_lut": is_boundary_token_lut,
    }


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    """Fail fast on mismatched dataset/tokenizer pairs when a manifest is available."""
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None

    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")

    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )

    return dataset_dir.name, actual_train_files, expected_train_files
