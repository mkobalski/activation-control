#!/usr/bin/env python3
"""Build the cross-model compliance table for the activation-control battery.

Three provenance classes, because the raw is not uniformly retained:
  measured  -- results/raw/<run>/results.json is on disk -> metrics.compliance_rate
  wandb     -- raw pruned by the snapshot lane; recovered from the W&B project
  documented-- raw never landed on this volume (>150B panel); values transcribed
               from models.txt and the run write-ups

Writes COMPLIANCE_<date>.md and compliance_by_model.json beside this script.
"""
import json, os, re, glob, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
RAW = os.path.join(RESULTS, "raw")
OUT_MD = os.path.join(HERE, f"COMPLIANCE_{datetime.date.today():%Y-%m-%d}.md")
OUT_JSON = os.path.join(HERE, "compliance_by_model.json")

MAIN_N, LT_N = 8600, 8800

# ---------------------------------------------------------------- raw on disk
def from_raw(run_dir):
    """(rate, n_total, n_compliant) from a retained run dir, or None."""
    p = os.path.join(RAW, run_dir, "results.json")
    if not os.path.exists(p):
        return None
    size = os.path.getsize(p)
    with open(p, "rb") as f:                       # metrics live at the tail
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf8", "replace")
    rate = re.search(r'"compliance_rate":\s*([0-9.eE+-]+)', tail)
    n = re.search(r'"n_samples":\s*([0-9]+)', tail)
    if not (rate and n):
        return None
    rate, n = float(rate.group(1)), int(n.group(1))
    return rate, n, round(rate * n)


# ------------------------------------------------------------------- the W&B
def from_wandb(models):
    """{model: {'main': (...), 'lt': (...)}} for raw-pruned runs.

    Bound to the scoring artifact's run dir by created_at proximity; the set
    selector and trial count disambiguate main from lt. The ~172-trial smokes
    that precede each run are dropped by the trial-count filter.
    """
    import wandb
    api = wandb.Api(timeout=60)
    entity = api.default_entity
    runs = list(api.runs(f"{entity}/activation-control", per_page=300))

    cand = {}
    for r in runs:
        m = r.config.get("model")
        if m not in models or r.state != "finished":
            continue
        s = r.summary
        total = s.get("total_trials")
        if total not in (MAIN_N, LT_N):
            continue
        cand.setdefault(m, []).append({
            "created": r.created_at, "run": r.name,
            "sets": r.config.get("active_sets"), "total": total,
            "rate": s.get("compliance_rate"), "compliant": s.get("compliant_trials"),
        })
    return cand


def stamp_to_dt(s):
    """'20260723_161714' -> datetime."""
    return datetime.datetime.strptime(s[:15], "%Y%m%d_%H%M%S")


def pick(cands, kind, dir_stamp):
    """The candidate of the right kind whose W&B init is nearest the dir stamp."""
    want_n = LT_N if kind == "lt" else MAIN_N
    pool = [c for c in cands if c["total"] == want_n]
    if not pool:
        return None
    target = stamp_to_dt(dir_stamp)
    def delta(c):
        t = datetime.datetime.strptime(c["created"], "%Y-%m-%dT%H:%M:%SZ")
        return abs((t - target).total_seconds())
    best = min(pool, key=delta)
    best = dict(best, lag_s=int(delta(best)))
    return best


# --------------------------------------------------- documented large models
# the run write-ups and models.txt. No raw on this
# volume; the durable artifact:// references in the panel README are the only
# route to the two "not recorded" cells.
DOCUMENTED = {
    "qwen3_coder_480b": {
        "main": {"rate": 0.835, "note": "aggregate; four conditions below the 70% gate "
                                        "(persist_after_fourth 43.0%, think_intensely 52.8%, "
                                        "think_about 59.4%, loc_end 67.6%, each n=500)"},
        "lt": {"rate": 1.0, "note": None},
        "source": "run write-ups; models.txt",
    },
    "llama4_maverick": {
        "main": {"rate": 0.9559, "note": None},
        "lt": {"rate": 1.0, "note": None},
        "source": "run write-ups",
    },
    "qwen35_397b_a17b": {
        "main": {"rate": 0.9994, "compliant": 8595, "note": None},
        "lt": {"rate": 1.0, "compliant": 8800, "note": None},
        "source": "run write-ups; models.txt",
    },
    "glm52": {
        "main": {"rate": 0.9991, "note": "verified with zero think tags "
                                         "(enable_thinking=false pinned)"},
        "lt": None,
        "source": "run write-ups",
    },
    "qwen3_235b_a22b_2507": {
        "main": None,
        "lt": None,
        "source": "not recorded; only loc_end is documented "
                  "(441/500 = 88.2% fuzzy-compliant, 45/500 = 9.0% exact-normalized)",
    },
}


def main():
    # every scored model, with the run dirs its scores were actually computed from
    scored = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "SCORES_*.json"))):
        d = json.load(open(f))
        scored[d["model"]] = {
            "main_dir": os.path.basename(d["main_run"].rstrip("/")),
            "lt_dir": os.path.basename(d["lt_run"].rstrip("/")),
        }

    measured, pruned = {}, []
    for model, dirs in scored.items():
        main, lt = from_raw(dirs["main_dir"]), from_raw(dirs["lt_dir"])
        if main or lt:
            measured[model] = {"main": main, "lt": lt}
        else:
            pruned.append(model)

    snapshots = [m for m in pruned if m.startswith("olmo")]
    cand = from_wandb(snapshots)

    records = {}
    for model in sorted(scored):
        dirs = scored[model]
        rec = {"main_run": dirs["main_dir"], "lt_run": dirs["lt_dir"]}

        if model in measured:
            rec["provenance"] = "measured"
            rec["source"] = f"results/raw/<run>/results.json (metrics.compliance_rate)"
            for kind in ("main", "lt"):
                v = measured[model][kind]
                rec[kind] = (None if v is None else
                             {"rate": v[0], "n_trials": v[1], "n_compliant": v[2]})

        elif model in snapshots:
            rec["provenance"] = "wandb"
            rec["source"] = "W&B mkobalski-none/activation-control (raw pruned by olmo_snapshot_lane.sh)"
            for kind in ("main", "lt"):
                c = pick(cand.get(model, []), kind, dirs[f"{kind}_dir"])
                rec[kind] = (None if c is None else {
                    "rate": c["rate"], "n_trials": c["total"],
                    "n_compliant": c["compliant"], "wandb_run": c["run"],
                    "wandb_init_lag_s": c["lag_s"], "active_sets": c["sets"],
                })

        elif model in DOCUMENTED:
            doc = DOCUMENTED[model]
            rec["provenance"] = "documented"
            rec["source"] = doc["source"]
            for kind in ("main", "lt"):
                v = doc[kind]
                if v is None:
                    rec[kind] = None
                    continue
                cell = {"rate": v["rate"],
                        "n_trials": MAIN_N if kind == "main" else LT_N}
                if "compliant" in v:
                    cell["n_compliant"] = v["compliant"]
                if v.get("note"):
                    cell["note"] = v["note"]
                rec[kind] = cell
        else:
            rec["provenance"] = "unknown"

        records[model] = rec

    payload = {
        "generated": "2026-07-31",
        "definition": "compliance_rate = share of trials whose transcription reaches "
                      "Ratcliff-Obershelp similarity >= 0.85 to the target sentence "
                      "(results.json -> metrics.compliance_rate). Base checkpoints are "
                      "scored on the first |target| characters; reasoning models on the "
                      "final channel only. See the paper's Supplementary Material, 'Trial compliance'.",
        "trial_counts": {"main": MAIN_N, "layer_targeting": LT_N},
        "models": records,
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2, sort_keys=False)

    # ------------------------------------------------------------- markdown
    def cell(c):
        if c is None:
            return "not recorded"
        pct = f"{c['rate'] * 100:.2f}%"
        if "n_compliant" in c and c["n_compliant"] is not None:
            return f"{pct} ({c['n_compliant']:,}/{c['n_trials']:,})"
        return pct

    GROUPS = [
        ("Panel models — raw retained (measured)",
         [m for m in sorted(measured) if not m.startswith("olmo")],
         "Read directly from each model's `results.json`, resolved through the "
         "`main_run`/`lt_run` recorded in its `SCORES_*.json` (so the superseded "
         "`20260722_015304_gptoss_20b_low_lt` is correctly ignored)."),
        ("Olmo training-snapshot lane — raw pruned (recovered from W&B)",
         [m for m in sorted(scored) if m.startswith("olmo")],
         "`olmo_snapshot_lane.sh` deletes raw activations and weights on completion, "
         "and the derived JSONs store no trial counts — but the runner logs "
         "`compliance_rate` to W&B, which survives. Each run is bound to its scoring "
         "run dir by created_at (W&B init lags the dir stamp by ~10-90 s), by "
         "`active_sets`, and by trial count. The two Instruct anchors, which still have "
         "raw, reproduce their measured values exactly; the five 7B main values "
         "reproduce the stamps hardcoded in `scripts/snapshot_superplot.py`."),
        ("Large panel (>150B) — raw never on this volume (documented)",
         [m for m in DOCUMENTED],
         "Transcribed from the run write-ups and `models.txt`. These runs used "
         "the separate experiment platform and were never logged to W&B, so the two "
         "'not recorded' cells can only come from the `artifact://resolution-6d0064/...` "
         "sidecars referenced per-model in the panel README."),
    ]

    lines = [
        "# Instruction compliance by model — activation-control battery",
        "",
        "Generated 2026-07-31 from the artifacts on the RunPod volume. Regenerate with "
        "the script this file was built by; do not hand-edit the numbers.",
        "",
        "**Definition.** A trial is compliant when the generated transcription reaches "
        "Ratcliff–Obershelp sequence similarity ≥ 0.85 to the target sentence "
        "(`results.json → metrics.compliance_rate`). Base checkpoints (the pre-training "
        "and mid-training Olmo snapshots) are scored on the first |target| characters, "
        "since they lack end-of-sequence discipline; reasoning models (gpt-oss) are "
        "scored on the final channel only. Non-compliant trials are saved but flagged "
        "`is_compliant=False`, and every scorer drops them. See the paper's Supplementary "
        "Material, paragraph *Trial compliance*.",
        "",
        "Main runs are 8,600 trials (`intensity`, `token_location`, `persistence`); "
        "layer-targeting runs are 8,800 (`layer_location`).",
        "",
    ]
    for title, models, note in GROUPS:
        lines += [f"## {title}", "", note, "",
                  "| Model | Main run (n=8,600) | Layer-targeting (n=8,800) |",
                  "|---|---|---|"]
        for m in models:
            r = records[m]
            # the two Instruct anchors sit in the snapshot curve but kept their raw
            mark = " †" if title.startswith("Olmo") and r["provenance"] == "measured" else ""
            lines.append(f"| `{m}`{mark} | {cell(r.get('main'))} | {cell(r.get('lt'))} |")
        if title.startswith("Olmo"):
            lines.append("")
            lines.append("† Instruct anchor — raw retained, so these two rows are measured "
                         "from `results.json`, not recovered. They are what validates the "
                         "other ten: the W&B `compliance_rate` for each anchor equals its "
                         "measured value exactly.")
        lines.append("")

    notes = []
    for m in sorted(records):
        for k in ("main", "lt"):
            c = records[m].get(k)
            if isinstance(c, dict) and c.get("note"):
                notes.append(f"- **`{m}` {k}** — {c['note']}")
    if notes:
        lines += ["## Per-cell notes", ""] + notes + [""]

    lines += [
        "## Gaps",
        "",
        "- `qwen3_235b_a22b_2507` — neither run recorded. The only compliance figure "
        "documented anywhere for it is the `loc_end` condition: 441/500 (88.2%) "
        "fuzzy-compliant, 45/500 (9.0%) exact-normalized.",
        "- `glm52` — layer-targeting run not recorded (main is).",
        "",
        "Neither is recoverable: nothing on this volume or in W&B carries them.",
        "",
    ] + [""]
    open(OUT_MD, "w").write("\n".join(lines))
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")
    print(f"models: {len(records)} "
          f"(measured {len(measured)}, wandb {len(snapshots)}, documented {len(DOCUMENTED)})")


if __name__ == "__main__":
    main()
