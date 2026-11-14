"""
Ladder harness: run an idea on several model sizes, then compare decision criteria.
Frozen (the agent must not modify it).

  Criterion A (original autoresearch): val_bpb of the mini rung.
  Criterion B (scaling): fit log(L - E) = a + b * log(C) on the nano/micro/mini rungs,
                         predict L at the big rung's compute (~10x mini).
  Truth:                 val_bpb actually measured on the big rung.

All comparisons are "idea vs baseline" (gain = baseline - idea, positive = better), and a
verdict is "keep" only if the gain exceeds K_SIGMA standard errors of the difference,
where the noise sigma is measured on the baseline seeds (gate 1 / gate 3).

Commands (results go to runs/<tag>/<rung>_s<seed>.json, re-running skips finished runs):
  uv run ladder.py run baseline --seeds 0 1 2                      # gate 1: 3 small rungs x 3 seeds
  uv run ladder.py noise                                           # gate 1 verdict: is B too noisy?
  uv run ladder.py run halve_batch --patch ideas/halve_batch.patch # gate 2: one idea, small rungs
  uv run ladder.py report                                          # A/B verdicts, disagreements
  uv run ladder.py run baseline --rungs big --seeds 0 1            # gate 3: noise at big scale
  uv run ladder.py run halve_batch --patch ideas/halve_batch.patch --rungs big   # gate 4
  uv run ladder.py report                                          # + truth, Spearman, accuracy
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("LADDER_RUNS_DIR", os.path.join(HERE, "runs"))
SMALL = ["nano", "micro", "mini"]
BASELINE = "baseline"
K_SIGMA = 2.0
H100_PEAK = 989.5e12
MIN_MFU_FOR_TIMEOUT = 0.10  # a run slower than this is killed (also covers hangs)
FILES_FOR_RUN = ["train.py", "prepare.py", "ladder_lib.py"]


# ---------------------------------------------------------------------------
# Config and results I/O
# ---------------------------------------------------------------------------

def load_config():
    with open(os.path.join(HERE, "ladder_config.json")) as f:
        cfg = json.load(f)
    cfg["by_name"] = {r["name"]: r for r in cfg["rungs"]}
    return cfg


def result_path(tag, rung, seed):
    return os.path.join(RUNS, tag, f"{rung}_s{seed}")


def load_results(tag):
    """{rung: {seed: result_dict}} for every finished run of a tag."""
    out = {}
    d = os.path.join(RUNS, tag)
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if fn.endswith(".json") and "_s" in fn:
            rung, seed = fn[:-5].rsplit("_s", 1)
            with open(os.path.join(d, fn)) as f:
                out.setdefault(rung, {})[int(seed)] = json.load(f)
    return out


def all_tags():
    return sorted(t for t in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, t))) if os.path.isdir(RUNS) else []


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def prepare_workdir(patch):
    """Directory to run train.py from: the repo itself, or a temp copy with the patch applied."""
    if not patch:
        return HERE, None
    tmp = tempfile.mkdtemp(prefix="ladder_")
    for fn in FILES_FOR_RUN:
        shutil.copy(os.path.join(HERE, fn), tmp)
    subprocess.run(["git", "apply", os.path.abspath(patch)], cwd=tmp, check=True)
    for fn in ("prepare.py", "ladder_lib.py"):  # frozen files must be untouched
        if open(os.path.join(tmp, fn)).read() != open(os.path.join(HERE, fn)).read():
            raise SystemExit(f"patch modifies frozen file {fn}")
    return tmp, tmp


def run_one(cfg, tag, rung, seed, workdir):
    r = cfg["by_name"][rung]
    prefix = result_path(tag, rung, seed)
    if os.path.exists(prefix + ".json"):
        print(f"  [skip] {tag}/{rung} seed {seed} (already done)")
        return json.load(open(prefix + ".json"))
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    env = dict(os.environ, LADDER_DEPTH=str(r["depth"]), LADDER_WIDTH=str(r["width"]),
               LADDER_FLOPS=repr(r["flops"]), LADDER_SEED=str(seed),
               LADDER_DEVICE_BATCH_SIZE=str(r["device_batch_size"]), LADDER_OUT=prefix + ".tmp")
    script = os.environ.get("LADDER_TRAIN_SCRIPT", "train.py")  # override used by tests only
    timeout = r["flops"] / (H100_PEAK * MIN_MFU_FOR_TIMEOUT) + 600
    t0 = time.time()
    print(f"  [run ] {tag}/{rung} seed {seed} (depth {r['depth']}, width {r['width']}, "
          f"{r['flops']:.2e} FLOPs, timeout {timeout/60:.0f} min)", flush=True)
    with open(prefix + ".log", "w") as log:
        try:
            proc = subprocess.run([sys.executable, script], cwd=workdir, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    res = None
    if os.path.exists(prefix + ".tmp.json"):
        res = json.load(open(prefix + ".tmp.json"))
    if code != 0 or res is None or res.get("status") != "ok":
        res = dict(status="crash", reason=(res or {}).get("reason", f"exit code {code}"))
    if os.path.exists(prefix + ".tmp.steps.jsonl"):
        os.replace(prefix + ".tmp.steps.jsonl", prefix + ".steps.jsonl")
    if os.path.exists(prefix + ".tmp.json"):
        os.remove(prefix + ".tmp.json")
    res["wall_seconds"] = time.time() - t0
    with open(prefix + ".json", "w") as f:
        json.dump(res, f, indent=2)
    msg = f"val_bpb {res['val_bpb']:.6f}" if res["status"] == "ok" else f"CRASH ({res['reason']})"
    if res.get("final_epoch", 1) > 1:
        msg += "  WARNING: training data repeated (epoch > 1), download more shards with prepare.py"
    print(f"         -> {msg} in {res['wall_seconds']:.0f}s", flush=True)
    return res


def cmd_run(args):
    cfg = load_config()
    if args.tag == BASELINE and args.patch:
        raise SystemExit("the baseline tag must run the unmodified train.py")
    workdir, tmp = prepare_workdir(args.patch)
    try:
        for seed in args.seeds:
            for rung in args.rungs:
                res = run_one(cfg, args.tag, rung, seed, workdir)
                if res["status"] != "ok" and rung in SMALL:
                    print("  crash on a small rung: skipping the remaining rungs for this seed")
                    break
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Scaling fit and criteria
# ---------------------------------------------------------------------------

def linfit(xs, ys):
    """Least-squares line y = a + b x. Returns (a, b)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def predict(points, c_target, E=0.0, slope=None):
    """points: [(C, L)]. Fit log(L - E) = a + b log C (b fixed if slope is given), predict L(c_target)."""
    xs = [math.log(c) for c, _ in points]
    ys = [math.log(l - E) for _, l in points]
    if slope is None:
        a, b = linfit(xs, ys)
    else:
        b = slope
        a = sum(y - b * x for x, y in zip(xs, ys)) / len(xs)
    return E + math.exp(a + b * math.log(c_target))


def small_points(res_by_rung, seed):
    pts = []
    for rung in SMALL:
        r = res_by_rung.get(rung, {}).get(seed)
        if r is None or r.get("status") != "ok":
            return None
        pts.append((r["flops"], r["val_bpb"]))
    return pts


def per_seed_values(res, cfg, E, base_slope):
    """For one tag: lists of A, B (free slope), B_offset (baseline slope) and big values, one per seed."""
    A, B, B_off, big = [], [], [], []
    target = cfg["target_flops"]
    for seed in sorted(res.get("mini", {})):
        r = res["mini"][seed]
        if r.get("status") == "ok":
            A.append(r["val_bpb"])
        pts = small_points(res, seed)
        if pts:
            B.append(predict(pts, target, E))
            if base_slope is not None:
                B_off.append(predict(pts, target, E, slope=base_slope))
    for seed, r in sorted(res.get("big", {}).items()):
        if r.get("status") == "ok":
            big.append(r["val_bpb"])
    crashed = any(r.get("status") != "ok" for rr in res.values() for r in rr.values())
    return dict(A=A, B=B, B_off=B_off, big=big, crashed=crashed)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def std(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def baseline_fit(cfg, E):
    """Slope of the baseline scaling fit on the mean of its seeds (used by the B_offset variant)."""
    base = load_results(BASELINE)
    pts = []
    for rung in SMALL:
        vals = [r for r in base.get(rung, {}).values() if r.get("status") == "ok"]
        if not vals:
            return None
        pts.append((mean([r["flops"] for r in vals]), mean([r["val_bpb"] for r in vals])))
    xs = [math.log(c) for c, _ in pts]
    ys = [math.log(l - E) for _, l in pts]
    return linfit(xs, ys)[1]


def estimate_E(cfg):
    """Irreducible loss E for log(L - E) = a + b log C, fitted on baseline seed means.
    Uses every rung available (including big). Grid search on E in [0, min L)."""
    base = load_results(BASELINE)
    pts = []
    for rung in SMALL + ["big"]:
        vals = [r for r in base.get(rung, {}).values() if r.get("status") == "ok"]
        if vals:
            pts.append((mean([r["flops"] for r in vals]), mean([r["val_bpb"] for r in vals])))
    if len(pts) < 3:
        return None
    lmin = min(l for _, l in pts)
    best = None
    for i in range(0, 990):
        E = lmin * i / 1000
        xs = [math.log(c) for c, _ in pts]
        ys = [math.log(l - E) for _, l in pts]
        a, b = linfit(xs, ys)
        sse = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0] - 1e-15:
            best = (sse, E)
    return best[1]


def get_E(args, cfg):
    if args.E == "auto":
        E = estimate_E(cfg)
        return 0.0 if E is None else E
    return float(args.E)


def noise_table(cfg, E):
    base = per_seed_values(load_results(BASELINE), cfg, E, baseline_fit(cfg, E))
    return base, dict(A=std(base["A"]), B=std(base["B"]), B_off=std(base["B_off"]), big=std(base["big"]))


def verdict(gain, sigma, n_idea, n_base):
    """keep if the gain exceeds K_SIGMA standard errors of (baseline mean - idea mean)."""
    if not n_idea or not n_base or math.isnan(gain):
        return "n/a", float("nan")
    if math.isnan(sigma):
        return "?", float("nan")
    thr = K_SIGMA * sigma * math.sqrt(1 / n_idea + 1 / n_base)
    return ("keep" if gain > thr else "reset"), thr


def cmd_noise(args):
    cfg = load_config()
    E = get_E(args, cfg)
    base, sig = noise_table(cfg, E)
    print(f"Baseline noise (E = {E:.4f}; seeds: A={len(base['A'])}, B={len(base['B'])}, big={len(base['big'])})")
    for k, label in [("A", "A: mini val_bpb"), ("B", "B: predicted val_bpb at big (free slope)"),
                     ("B_off", "B_offset: predicted, baseline slope"), ("big", "truth: big val_bpb")]:
        v = base[k]
        print(f"  {label:42s} mean {mean(v):.5f}  sigma {sig[k]:.5f}  "
              f"(min detectable gain vs baseline, 1 idea seed: {K_SIGMA*sig[k]*math.sqrt(1+1/max(len(v),1)):.5f})"
              if v else f"  {label:42s} no data")
    if not math.isnan(sig["A"]) and not math.isnan(sig["B"]):
        print(f"\n  sigma_B / sigma_A = {sig['B']/sig['A']:.1f}x")
        print("  Gate 1 stops if the min detectable gain for B is larger than the typical gains of ideas "
              "(~0.002-0.005 bpb in Karpathy's run).")


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def cmd_report(args):
    cfg = load_config()
    E = get_E(args, cfg)
    slope = baseline_fit(cfg, E)
    base, sig = noise_table(cfg, E)
    if not base["A"]:
        raise SystemExit("no baseline results yet: run `ladder.py run baseline --seeds 0 1 2` first")
    nb = {k: len(base[k]) for k in ("A", "B", "B_off", "big")}
    print(f"E = {E:.4f} | baseline slope (log-log) = {slope if slope is None else round(slope, 4)} | "
          f"target = big rung, {cfg['target_flops']:.2e} FLOPs | gains are baseline - idea (positive = better)")
    print(f"sigma: A {sig['A']:.5f}  B {sig['B']:.5f}  B_off {sig['B_off']:.5f}  big {sig['big']:.5f}\n")
    hdr = (f"{'idea':24s} {'gainA':>8s} {'A':>5s} {'gainB':>8s} {'B':>5s} {'gainBoff':>8s} {'Boff':>5s} "
           f"{'gainBig':>8s} {'truth':>5s}  note")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tag in all_tags():
        if tag == BASELINE:
            continue
        v = per_seed_values(load_results(tag), cfg, E, slope)
        row = dict(tag=tag, crashed=v["crashed"])
        for k in ("A", "B", "B_off", "big"):
            gain = mean(base[k]) - mean(v[k]) if v[k] and base[k] else float("nan")
            row["gain_" + k] = gain
            row["v_" + k] = verdict(gain, sig[k], len(v[k]), nb[k])[0]
        if v["crashed"]:
            for k in ("A", "B", "B_off"):
                if row["v_" + k] == "n/a":
                    row["v_" + k] = "reset"
        disagree = row["v_A"] in ("keep", "reset") and row["v_B"] in ("keep", "reset") and row["v_A"] != row["v_B"]
        row["disagree"] = disagree
        rows.append(row)
        f = lambda x: "    n/a " if math.isnan(x) else f"{x:+.5f}"
        note = ("CRASH " if v["crashed"] else "") + ("A/B DISAGREE -> run big" if disagree and not v["big"] else
                                                     "A/B disagree" if disagree else "")
        print(f"{tag:24s} {f(row['gain_A'])} {row['v_A']:>5s} {f(row['gain_B'])} {row['v_B']:>5s} "
              f"{f(row['gain_B_off'])} {row['v_B_off']:>5s} {f(row['gain_big'])} {row['v_big']:>5s}  {note}")

    n_dis = sum(r["disagree"] for r in rows)
    print(f"\nGate 2: {n_dis} disagreement(s) between A and B out of {len(rows)} idea(s)."
          + ("  -> no disagreement: B makes the same decisions as A, stop here." if rows and not n_dis else ""))

    truth = [r for r in rows if not math.isnan(r["gain_big"])]
    if truth:
        print(f"\nGate 4: {len(truth)} idea(s) with a big-rung truth")
        for k, label in [("A", "A"), ("B", "B"), ("B_off", "B_offset")]:
            ok = [r for r in truth if not math.isnan(r["gain_" + k])]
            rho = spearman([r["gain_" + k] for r in ok], [r["gain_big"] for r in ok])
            decided = [r for r in ok if r["v_" + k] in ("keep", "reset") and r["v_big"] in ("keep", "reset")]
            acc = sum(r["v_" + k] == r["v_big"] for r in decided)
            fp = sum(r["v_" + k] == "keep" and r["v_big"] == "reset" for r in decided)
            print(f"  {label:9s} Spearman vs truth {rho:+.2f} | decisions right {acc}/{len(decided)} | "
                  f"false keeps {fp}")
        dis = [r for r in truth if r["disagree"] and r["v_big"] in ("keep", "reset")]
        if dis:
            wins = sum(r["v_B"] == r["v_big"] for r in dis)
            print(f"  On A/B disagreements: B right {wins}/{len(dis)}, A right {len(dis) - wins}/{len(dis)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an idea (or the baseline) on some rungs and seeds")
    r.add_argument("tag")
    r.add_argument("--patch", help="git diff to apply to train.py (omit for the current train.py)")
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--rungs", nargs="+", default=SMALL, choices=SMALL + ["big"])
    r.set_defaults(func=cmd_run)
    for name, fn, h in [("noise", cmd_noise, "gate 1: baseline noise of A, B and truth"),
                        ("report", cmd_report, "gates 2 and 4: verdicts, disagreements, Spearman")]:
        s = sub.add_parser(name, help=h)
        s.add_argument("--E", default="0", help="irreducible loss for the fit: a number, or 'auto' "
                                                "(fitted on baseline means; reliable once big baseline runs exist)")
        s.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
