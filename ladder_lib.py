"""
Frozen helpers shared by train.py and ladder.py. DO NOT MODIFY (agent).

Contract between the ladder harness and train.py, passed via environment variables:
  LADDER_DEPTH              model depth for this rung (int)
  LADDER_WIDTH              model width (d_model) for this rung (int, 0 = depth * ASPECT_RATIO)
  LADDER_FLOPS              training compute budget in FLOPs (float)
  LADDER_SEED               seed: weight init + training data order (int)
  LADDER_DEVICE_BATCH_SIZE  per-device micro-batch (memory only, does not change the math)
  LADDER_OUT                path prefix; writes <prefix>.json (result) and <prefix>.steps.jsonl
  LADDER_DRY_RUN=1          build the model on the meta device, report sizes, exit (no GPU needed)

Compute is counted the same way for every idea, independently of train.py internals:
  C = 6 * N * D, N = all parameters except nn.Embedding tables, D = training tokens.
Attention FLOPs are deliberately excluded (Kaplan-style): the count only depends on
the parameter count, so architectural tricks cannot change how compute is booked.
"""

import json
import math
import os

import torch.nn as nn


def env_int(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else int(v)


def env_float(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else float(v)


def is_dry_run():
    return os.environ.get("LADDER_DRY_RUN") == "1"


def count_params(model):
    """(N_non_embedding, N_total). Must be called on the raw (uncompiled) model."""
    emb_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            emb_ids.update(id(p) for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    non_emb = sum(p.numel() for p in model.parameters() if id(p) not in emb_ids)
    return non_emb, total


def flops_per_token(model):
    non_emb, _ = count_params(model)
    return 6 * non_emb


def token_budget(model, flops_budget):
    """Number of training tokens allowed by the FLOPs budget."""
    return int(flops_budget // flops_per_token(model))


def make_seeded_train_loader(make_dataloader, tokenizer, B, T, seed):
    """Training loader whose data order depends on the seed (rotates the shard order).
    seed=0 keeps the original order. Validation data is never affected."""
    import prepare
    original = prepare.list_parquet_files

    def rotated():
        files = original()
        val = os.path.join(prepare.DATA_DIR, prepare.VAL_FILENAME)
        train = [f for f in files if f != val]
        k = seed % len(train) if train else 0
        return train[k:] + train[:k] + [f for f in files if f == val]

    prepare.list_parquet_files = rotated
    try:
        loader = make_dataloader(tokenizer, B, T, "train")
        first = next(loader)  # generator starts reading files here, with the rotated order
    finally:
        prepare.list_parquet_files = original

    def chained():
        yield first
        yield from loader

    return chained()


class StepLogger:
    """Per-step training log (JSON lines), used later by instability detectors."""

    def __init__(self, out_prefix):
        self.f = open(out_prefix + ".steps.jsonl", "w") if out_prefix else None

    def log(self, **kw):
        if self.f is None:
            return
        self.f.write(json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                 for k, v in kw.items()}) + "\n")

    def close(self):
        if self.f is not None:
            self.f.close()


def write_result(out_prefix, **kw):
    if not out_prefix:
        return
    with open(out_prefix + ".json", "w") as f:
        json.dump(kw, f, indent=2)
