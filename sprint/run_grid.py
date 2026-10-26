"""[sprint tooling] Run the question-1.5 condition grid: conditions x seeds, in parallel, one subprocess per run.

Usage (from the repo root):
  python sprint/run_grid.py --grid pilot --conditions C0 C1 --seeds 0 1 2 --rounds 15 --model qwen3-30b
Writes runs to sprint/results/<grid>/<condition>/seed<k>/<timestamp>_<tag>/ and a manifest line per run.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "llm-agent-collusion"

def conditions(rounds: int, overseer_model: str, pressure: bool = False) -> dict[str, list[str]]:
    late = rounds // 3 + 1
    chat = ["--seller_comms_enabled"]
    d = {
        "C0": [],                                                             # no seller chat (negative control / remove-messages lever)
        "C1": chat,                                                           # seller chat (positive control)
        "C2": chat + ["--max_message_length", "50"],                          # message cap
        "C3": chat + ["--oversight_enabled", "--overseer_model", overseer_model],  # oversight notice + overseer
        "C4": chat + ["--temperature", "1.0"],                                # higher sampling temperature
        "C5": chat + ["--maverick_seats", "1"],                               # one rule-based maverick seller from the start
        "C6": chat + ["--maverick_seats", "1", "--maverick_at_round", str(late)],  # maverick swapped in after the cartel forms
        "SANITY": ["--maverick_seats", "3"],                                  # all-maverick sellers, LLM buyers
    }
    if pressure:  # every condition gets the paper's CEO-pressure message (used if the plain positive control fails)
        d = {k: v + ["--boss_pressure"] for k, v in d.items()}
    return d

def run_one(grid, cond, seed, extra, args, log):
    out = ROOT / "sprint" / "results" / grid / cond / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*/collusion_metrics.json")) and not args.force:
        return cond, seed, "skip"
    cmd = ["uv", "run", "python", "src/continuous_double_auction/simulation.py",
           "--rounds", str(args.rounds), "--sellers", str(args.sellers), "--buyers", str(args.buyers),
           "--seller_models", args.model, "--buyer_models", args.buyer_model or args.model,
           "--seeds", str(seed), "--tag", f"{cond}-s{seed}", "--condition", cond,
           "--log_dir", str(out), "-y"] + extra + args.extra
    t0 = time.time()
    with open(out / "stdout.txt", "w") as f:
        rc = subprocess.call(cmd, cwd=SIM, stdout=f, stderr=subprocess.STDOUT)
    rec = {"grid": grid, "cond": cond, "seed": seed, "rc": rc, "secs": round(time.time() - t0, 1), "cmd": cmd}
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return cond, seed, rc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--sellers", type=int, default=3)
    p.add_argument("--buyers", type=int, default=3)
    p.add_argument("--model", required=True)
    p.add_argument("--buyer_model", default=None, help="'zip' = rule-based buyers")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra simulation flags for every run")
    p.add_argument("--overseer_model", default=None)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--pressure", action="store_true", help="add --boss_pressure to every condition")
    args = p.parse_args()
    conds = conditions(args.rounds, args.overseer_model or args.model, args.pressure)
    log = ROOT / "sprint" / "results" / args.grid / "manifest.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s) for s in args.seeds for c in args.conditions]  # interleave conditions so partial grids stay balanced
    with ThreadPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(run_one, args.grid, c, s, conds[c], args, log) for c, s in jobs]
        for f in futs:
            print(*f.result(), flush=True)

if __name__ == "__main__":
    sys.exit(main())
