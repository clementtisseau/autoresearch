"""CPU tests of the ladder harness. Run: python -m pytest tests -q"""
import json
import math
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import ladder  # noqa: E402


def make_patch(tmp_path, name, body):
    """A git patch appending `body` to train.py."""
    lines = open(os.path.join(HERE, "train.py")).read().splitlines()
    n = len(lines)
    p = tmp_path / f"{name}.patch"
    p.write_text(
        "diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n"
        f"@@ -{n},1 +{n},2 @@\n {lines[-1]}\n+{body}\n")
    return str(p)


def run_ladder(tmp_path, *args):
    env = dict(os.environ, LADDER_RUNS_DIR=str(tmp_path / "runs"),
               LADDER_TRAIN_SCRIPT=os.path.join(HERE, "tests", "fake_train.py"))
    return subprocess.run([sys.executable, os.path.join(HERE, "ladder.py"), *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def test_predict_recovers_power_law():
    pts = [(c, 0.8 + 0.2 * (c / 1e17) ** -0.25) for c in (2e16, 3e16, 9e16)]
    assert ladder.predict(pts, 9e17, E=0.8) == pytest.approx(0.8 + 0.2 * 9 ** -0.25, rel=1e-9)
    # pure power law (E=0) is biased, but still decreasing with compute
    assert ladder.predict(pts, 9e17, E=0.0) < pts[-1][1]


def test_predict_fixed_slope_is_offset_only():
    pts = [(1e16, 1.0), (1e17, 0.9)]
    b = (math.log(0.9) - math.log(1.0)) / math.log(10)
    assert ladder.predict(pts, 1e18, slope=b) == pytest.approx(0.81, rel=1e-6)


def test_verdict_threshold():
    assert ladder.verdict(0.003, 0.001, 1, 3)[0] == "keep"     # thr = 2*0.001*sqrt(4/3) = 0.0023
    assert ladder.verdict(0.002, 0.001, 1, 3)[0] == "reset"
    assert ladder.verdict(float("nan"), 0.001, 1, 3)[0] == "n/a"


def test_spearman():
    assert ladder.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ladder.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_end_to_end_gates(tmp_path):
    # Gate 1: baseline noise on small rungs
    run_ladder(tmp_path, "run", "baseline", "--seeds", "0", "1", "2")
    out = run_ladder(tmp_path, "noise")
    assert "sigma" in out

    # Gate 2: ideas on small rungs
    good = make_patch(tmp_path, "good", "# FAKE_EFFECT -0.010 0.0")          # helps at every scale
    trap = make_patch(tmp_path, "trap", "# FAKE_EFFECT -0.006 0.012")       # helps small, hurts big
    crash = make_patch(tmp_path, "crash", "# FAKE_CRASH")
    for tag, p in [("good", good), ("trap", trap), ("crash", crash)]:
        run_ladder(tmp_path, "run", tag, "--patch", p)
    out = run_ladder(tmp_path, "report")
    rows = {l.split()[0]: l for l in out.splitlines() if l.split() and l.split()[0] in ("good", "trap", "crash")}
    assert " keep " in rows["good"]
    assert "DISAGREE" in rows["trap"]          # A keeps, B (trend) rejects
    assert "CRASH" in rows["crash"]

    # Gate 3 + 4: truth on the big rung
    run_ladder(tmp_path, "run", "baseline", "--rungs", "big", "--seeds", "0", "1")
    run_ladder(tmp_path, "run", "trap", "--patch", trap, "--rungs", "big")
    run_ladder(tmp_path, "run", "good", "--patch", good, "--rungs", "big")
    out = run_ladder(tmp_path, "report", "--E", "auto")
    assert "On A/B disagreements: B right 1/1" in out

    # re-running skips finished runs
    assert "[skip]" in run_ladder(tmp_path, "run", "good", "--patch", good)


def test_patch_touching_frozen_file_is_rejected(tmp_path):
    p = tmp_path / "bad.patch"
    line = open(os.path.join(HERE, "prepare.py")).read().splitlines()[0]
    p.write_text(f"diff --git a/prepare.py b/prepare.py\n--- a/prepare.py\n+++ b/prepare.py\n"
                 f"@@ -1,1 +1,2 @@\n {line}\n+# hack\n")
    with pytest.raises(subprocess.CalledProcessError):
        run_ladder(tmp_path, "run", "bad", "--patch", str(p))


def test_train_dry_run_matches_config():
    cfg = json.load(open(os.path.join(HERE, "ladder_config.json")))
    mini = next(r for r in cfg["rungs"] if r["name"] == "mini")
    env = dict(os.environ, LADDER_DRY_RUN="1", LADDER_DEPTH="8", LADDER_WIDTH="512")
    out = subprocess.run([sys.executable, "train.py"], cwd=HERE, env=env, capture_output=True,
                         text=True, check=True).stdout
    info = json.loads(out.split("LADDER_DRY_RUN ", 1)[1])
    assert info["n_total"] == 50332176          # original autoresearch baseline: 50.3M params
    assert info["n_non_emb"] == mini["n_non_emb"]
