"""Stand-in for train.py in tests: no GPU, follows a synthetic scaling law.

val_bpb = E + A * (C / C0) ** -alpha + effect(C) + seed_noise
An idea is expressed by a line in train.py:  # FAKE_EFFECT <a> <b>
meaning effect(C) = a + b * log10(C / C_mini): b > 0 means the gain shrinks with scale.
A line '# FAKE_CRASH' makes the run crash.
"""
import json
import math
import os
import random

E, A, ALPHA, C0, C_MINI = 0.80, 0.20, 0.25, 8.81e16, 8.81e16
NOISE = 0.0008

src = open("train.py").read()
if "# FAKE_CRASH" in src:
    raise SystemExit(1)
a = b = 0.0
for line in src.splitlines():
    if line.startswith("# FAKE_EFFECT"):
        _, _, a, b = line.split()
        a, b = float(a), float(b)

C = float(os.environ["LADDER_FLOPS"])
seed = int(os.environ["LADDER_SEED"])
rng = random.Random(hash((seed, round(math.log(C), 3), a, b)) & 0xFFFFFFFF)
loss = E + A * (C / C0) ** -ALPHA + a + b * math.log10(C / C_MINI) + rng.gauss(0, NOISE)
out = os.environ["LADDER_OUT"]
with open(out + ".steps.jsonl", "w") as f:
    f.write(json.dumps(dict(step=0, loss=loss)) + "\n")
with open(out + ".json", "w") as f:
    json.dump(dict(status="ok", val_bpb=loss, flops=C, seed=seed,
                   depth=int(os.environ["LADDER_DEPTH"])), f)
