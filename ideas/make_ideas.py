"""
Generate the gate-2 idea patches (ideas/<name>.patch) from train.py.
Each idea is a list of exact (old, new) replacements in train.py; the script fails if a line is not found.
Usage: python ideas/make_ideas.py
"""
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# name: (category, source, [(old, new), ...])
IDEAS = {
    # ---- Suspected traps: kept in short / small runs, doubtful at larger scale ----
    "halve_batch": ("trap", "Karpathy run (progress.png, kept): total batch 524K -> 262K", [
        ("TOTAL_BATCH_SIZE = 2**19 # ~524K", "TOTAL_BATCH_SIZE = 2**18 # ~262K")]),
    "seed_change": ("trap", "Karpathy run (kept: seed 42 -> 137); discussion #285", [
        ('SEED = L.env_int("LADDER_SEED", 42)', 'SEED = L.env_int("LADDER_SEED", 42) + 95')]),
    "window_1_8": ("trap", "Karpathy run (kept): short window 1/8 context (256 tokens)", [
        ("short_window = long_window // 2", "short_window = long_window // 8")]),
    "window_1_4": ("trap", "Karpathy run (kept): short window 1/4 context", [
        ("short_window = long_window // 2", "short_window = long_window // 4")]),
    "window_sssssl": ("trap", "Karpathy run (kept): SSSSL pattern, fewer full-attention layers", [
        ('WINDOW_PATTERN = "SSSL"', 'WINDOW_PATTERN = "SSSSL"')]),
    "warmup_5pct": ("trap", "Karpathy run (kept): add 5% warmup", [
        ("WARMUP_RATIO = 0.0 ", "WARMUP_RATIO = 0.05")]),
    "adam_beta2_999": ("trap", "issue #596 (RTX 3060, kept): Adam beta2 0.95 -> 0.999", [
        ("ADAM_BETAS = (0.8, 0.95)", "ADAM_BETAS = (0.8, 0.999)")]),
    "scalar_lr_03": ("trap", "issue #596 (RTX 3060, kept): SCALAR_LR 0.5 -> 0.3", [
        ("SCALAR_LR = 0.5 ", "SCALAR_LR = 0.3 ")]),
    # ---- Controls: reported to transfer from d12 to d24 (nanochat commit 6ed7d1d) ----
    "unembed_lr_x2": ("control", "nanochat 6ed7d1d + Karpathy run: unembedding LR 0.004 -> 0.008", [
        ("UNEMBEDDING_LR = 0.004", "UNEMBEDDING_LR = 0.008")]),
    "rope_100k": ("control", "nanochat 6ed7d1d + Karpathy run: RoPE base 10K -> 100K", [
        ("base=10000, device=None", "base=100000, device=None")]),
    "weight_decay_028": ("control", "nanochat 6ed7d1d: weight decay 0.2 -> 0.28", [
        ("WEIGHT_DECAY = 0.2 ", "WEIGHT_DECAY = 0.28")]),
    "warmdown_065_final_005": ("control", "nanochat 6ed7d1d: warmdown 0.5 -> 0.65, final LR frac 0 -> 0.05", [
        ("WARMDOWN_RATIO = 0.5 ", "WARMDOWN_RATIO = 0.65"),
        ("FINAL_LR_FRAC = 0.0 ", "FINAL_LR_FRAC = 0.05")]),
    "embed_init_08": ("control", "nanochat 6ed7d1d: embedding init std 1.0 -> 0.8", [
        ("self.transformer.wte.weight, mean=0.0, std=1.0)", "self.transformer.wte.weight, mean=0.0, std=0.8)")]),
    "muon_momentum": ("control", "nanochat 6ed7d1d: Muon beta2 0.95 -> 0.9, momentum target 0.95 -> 0.97 over 400 steps", [
        ("ns_steps=5, beta2=0.95,", "ns_steps=5, beta2=0.9,"),
        ("frac = min(step / 300, 1)", "frac = min(step / 400, 1)"),
        ("return (1 - frac) * 0.85 + frac * 0.95", "return (1 - frac) * 0.85 + frac * 0.97")]),
    "x0_lambda_005": ("control", "Karpathy run (kept): x0_lambda init 0.1 -> 0.05", [
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.05)")]),
}


def make_patch(name, edits):
    src = open(os.path.join(ROOT, "train.py")).read()
    new = src
    for old, rep in edits:
        assert new.count(old) == 1, f"{name}: '{old}' found {new.count(old)} times"
        new = new.replace(old, rep)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "a")), os.makedirs(os.path.join(d, "b"))
        open(os.path.join(d, "a", "train.py"), "w").write(src)
        open(os.path.join(d, "b", "train.py"), "w").write(new)
        diff = subprocess.run(["git", "diff", "--no-index", "a/train.py", "b/train.py"], cwd=d,
                              capture_output=True, text=True).stdout
    diff = diff.replace("a/a/train.py", "a/train.py").replace("b/b/train.py", "b/train.py")
    with open(os.path.join(HERE, f"{name}.patch"), "w") as f:
        f.write(diff)


if __name__ == "__main__":
    for name, (cat, source, edits) in IDEAS.items():
        make_patch(name, edits)
        print(f"{cat:8s} {name:24s} {source}")
