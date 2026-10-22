"""Compare independent specimens versus technical repeats at fixed assay cost."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import html
import io
import json
import math
from pathlib import Path
import platform
import time
import tracemalloc

import numpy as np

from . import __version__
from .benchmark import array_hash, feature_map, source_hashes, write_csv
from .inference import conformal_threshold, metrics, prediction_sets
from .models import LogisticClassifier, Standardizer
from .set_model import DeepSetClassifier, SetStandardizer

MODELS = ("mean_interaction", "mean_sd_interaction", "deep_set")
DEFAULT_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]


def validate_config(config: dict) -> dict:
    config = dict(config)
    required = {"budget", "repeats", "noise_levels", "seeds", "test_specimens", "features", "alpha", "epochs",
                "logistic_learning_rate", "set_learning_rate", "l2", "set_hidden", "set_head_width", "max_seconds"}
    if not required <= set(config):
        raise ValueError(f"Missing config fields: {sorted(required-set(config))}")
    for name in ("budget", "test_specimens", "features", "epochs", "set_head_width"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["budget"] % 5 or config["features"] < 3:
        raise ValueError("Budget must be divisible by five and features >=3")
    for key in ("repeats", "seeds"):
        values = config[key]
        if not values or len(values) != len(set(values)) or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError(f"{key} must be a nonempty distinct integer list")
    if any(m < 1 or (config["budget"] // 5) % m for m in config["repeats"]) or min(config["seeds"]) < 0:
        raise ValueError("Each repeat count must divide the validation/calibration assay budget; seeds must be nonnegative")
    if min(config["budget"] // (5*m) for m in config["repeats"]) < 2:
        raise ValueError("At least two validation/calibration specimens per allocation required")
    if not config["noise_levels"] or len(set(config["noise_levels"])) != len(config["noise_levels"]) or any(not np.isfinite(v) or v < 0 for v in config["noise_levels"]):
        raise ValueError("Noise levels must be finite, distinct, and nonnegative")
    if not np.isfinite(config["alpha"]) or not 0 < config["alpha"] < 1:
        raise ValueError("alpha must lie in (0,1)")
    for name in ("logistic_learning_rate", "set_learning_rate", "max_seconds"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(config["l2"]) or config["l2"] < 0:
        raise ValueError("l2 must be finite and nonnegative")
    if len(config["set_hidden"]) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in config["set_hidden"]):
        raise ValueError("set_hidden must contain two positive integers")
    return config


def draw_pool(n: int, repeats: int, features: int, seed: int, role: str) -> dict:
    """Independent pool, fixed by role. Prefixes pair allocations without role reuse."""
    streams = np.random.SeedSequence(seed).spawn(4)
    z = np.random.default_rng(streams[0]).normal(size=(n, 3))
    logit = 2.5 * z[:, 0] * z[:, 1] + 1.2 * z[:, 2]
    y = np.random.default_rng(streams[1]).binomial(1, np.exp(-np.logaddexp(0., -logit)))
    nuisance = np.random.default_rng(streams[2]).normal(scale=1.5, size=(n, features-3))
    base = np.column_stack((z, nuisance))
    noise = np.random.default_rng(streams[3]).normal(size=(n, repeats, features))
    return {"base": base, "noise": noise, "y": y, "ids": [f"{role}-{i:05}" for i in range(n)]}


def pool_subset(pool: dict, n: int, repeats: int, noise: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = pool["base"][:n, None, :] + noise * pool["noise"][:n, :repeats, :]
    return X, np.ones(X.shape[:2], dtype=bool), pool["y"][:n]


def summary_features(X: np.ndarray, model: str) -> np.ndarray:
    means = X.mean(axis=1)
    interactions = feature_map(means, "interaction_logistic")
    # Population SD is exactly zero at m=1; it is a feature, not an unbiased variance estimator.
    return np.column_stack((interactions, X.std(axis=1, ddof=0))) if model == "mean_sd_interaction" else interactions


def summarize_runs(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    identities = [(r["seed"], r["noise"], r["repeats"], r["model"]) for r in rows]
    if not rows or len(set(identities)) != len(identities):
        raise ValueError("Run metrics must contain distinct seed/scenario/model rows")
    for noise, repeats in {(r["noise"], r["repeats"]) for r in rows}:
        observed = {r["model"] for r in rows if (r["noise"], r["repeats"]) == (noise, repeats)}
        if observed != set(MODELS):
            raise ValueError("Every scenario must contain all three model comparisons")
    summaries, paired = [], []
    keys = ("brier", "accuracy", "log_loss", "coverage", "mean_set_size", "empty_set_rate", "singleton_rate")
    for noise, repeats, model in sorted({(r["noise"], r["repeats"], r["model"]) for r in rows}):
        group = [r for r in rows if (r["noise"], r["repeats"], r["model"]) == (noise, repeats, model)]
        record = {"noise": noise, "repeats": repeats, "model": model, "n_seeds": len(group),
                  "development_specimens": group[0]["development_specimens"], "development_assays": group[0]["development_assays"],
                  "test_specimens": group[0]["n_test_specimens"], "test_assays": group[0]["test_assays"]}
        for key in keys:
            values = np.array([g[key] for g in group])
            record[key + "_mean"] = float(values.mean())
            record[key + "_mcse"] = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        summaries.append(record)
    for noise, repeats in sorted({(r["noise"], r["repeats"]) for r in rows}):
        group = {r["seed"]: r for r in rows if (r["noise"], r["repeats"], r["model"]) == (noise, repeats, "deep_set")}
        for baseline in MODELS[:2]:
            base = {r["seed"]: r for r in rows if (r["noise"], r["repeats"], r["model"]) == (noise, repeats, baseline)}
            if set(group) != set(base):
                raise ValueError("Paired comparison requires identical seed sets")
            delta = np.array([group[s]["brier"]-base[s]["brier"] for s in sorted(group)])
            paired.append({"noise": noise, "repeats": repeats, "baseline": baseline, "n_seeds": len(delta),
                           "deep_set_minus_baseline_brier": float(delta.mean()),
                           "mcse_paired_difference": float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None})
    return summaries, paired


def build_report(config: dict, summaries: list[dict], paired: list[dict]) -> str:
    lines = ["# ReplicateShift — measurement-allocation pilot", "",
             "Executed exploratory simulation. This evaluates complete development/deployment policies; it is not a biological result or an optimal laboratory allocation recommendation.", "",
             f"Development budget: **{config['budget']} assay rows** in every condition. "
             f"Independent test specimens: {config['test_specimens']} per condition, requiring an additional `test_specimens × repeats` assays. "
             f"Each condition uses {len(config['seeds'])} independent complete-run seeds.", "",
             "## Results", "",
             "Mean ± Monte Carlo standard error across independent runs. MCSE estimates simulation precision, not a biological confidence interval. All methods and scenarios within each seed are paired.", "",
             "| Noise SD | Repeats | Dev. specimens | Model | Brier ↓ | Coverage | Mean set size | Test assays |",
             "|---:|---:|---:|---|---:|---:|---:|---:|"]
    def fmt(row: dict, key: str) -> str:
        e = row[key+"_mcse"]
        return f"{row[key+'_mean']:.3f}" + (f" ± {e:.3f}" if e is not None else " (MCSE unavailable)")
    for r in summaries:
        lines.append(f"| {r['noise']:g} | {r['repeats']} | {r['development_specimens']} | {r['model']} | {fmt(r,'brier')} | {fmt(r,'coverage')} | {fmt(r,'mean_set_size')} | {r['test_assays']} |")
    lines += ["", "## Paired model differences", "", "Negative differences favor Deep Sets. These differences use each seed's complete-run Brier metrics; individual rows from different fitted procedures are not pooled as independent trials.", "",
              "| Noise SD | Repeats | Baseline | Deep Sets minus baseline Brier | MCSE of paired difference |", "|---:|---:|---|---:|---:|"]
    for r in paired:
        se = "unavailable" if r["mcse_paired_difference"] is None else f"{r['mcse_paired_difference']:.4f}"
        lines.append(f"| {r['noise']:g} | {r['repeats']} | {r['baseline']} | {r['deep_set_minus_baseline_brier']:.4f} | {se} |")
    lines += ["", "## Experimental contract", "",
              "- Separate maximum pools for training, validation, calibration, and testing prevent any specimen from changing roles across allocations. Prefix subsets and paired noise draws make policy comparisons within a seed paired.",
              "- 60/20/20% of the development assays fund training/validation/calibration respectively. Calibration sample size and finite-sample quantile resolution therefore change with repeat count.",
              "- All trainable methods select the minimum validation-BCE checkpoint from a fixed training budget. Scaling uses training specimens only; calibration/test labels never select checkpoints.",
              "- The two logistic baselines share all mean features and pairwise mean interactions. The second adds within-specimen SD, explicitly zero for one repeat. Deep Sets learns an embedding from the individual repeats and uses masked mean pooling.",
              "- The simulator has independent specimens and conditionally independent Gaussian technical noise. Sample SD is largely ancillary here; no neural-model advantage is assumed. This is not a calibrated model of RNA chemistry or an LC-MS instrument.",
              f"- Nominal conformal coverage is {1-config['alpha']:.0%} under matching exchangeable calibration/test specimen policies. The guarantee is marginal, not conditional on every specimen, subgroup, or fixed calibration realization.",
              "- Each repeat count changes both the development allocation and the test-time measurement policy. Test assay costs are additional and explicitly reported. The study does not isolate a training-only effect.",
              "- The budget counts equal-cost assay rows only; specimen procurement, preparation, failed measurements, and computing costs are excluded.",
              "- Ten seeds by default provide a small pilot, not a powered confirmatory comparison. Selecting the best-looking allocation from this table is exploratory. No p-values, universal optimum, new statistical theorem, or real-data benefit are claimed.", "",
              "## Reproduce and inspect", "", "Replace `path/to/this/run` with the directory containing this report, then run from the repository root:", "", "```sh", "PYTHONPATH=src python -m replicateshift.allocation --config path/to/this/run/protocol.json --output results/local-allocation", "```", "",
              "`protocol.json` contains the executed configuration. `per_run_metrics.csv` records selected epochs and role-specific specimen counts; together with the manifest's ordered pool IDs these determine each prefix split. `summary.csv`, `paired_comparisons.csv`, and `predictions.csv.gz` retain the remaining results. `manifest.json` records role seeds, pool hashes, source/output hashes, runtime, and traced-memory scope. `report.html` is a self-contained filterable table with no external scripts or services.", ""]
    return "\n".join(lines)


def html_report(config: dict, summaries: list[dict]) -> str:
    # Standard HTML tables preserve the numeric results; no invented chart geometry.
    data = json.dumps(summaries, allow_nan=False).replace("<", "\\u003c")
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReplicateShift · Allocation results</title><style>
:root{font-family:system-ui,sans-serif;color:#183b38;background:#f5f3eb}body{max-width:1180px;margin:48px auto;padding:0 28px;line-height:1.55}h1{font-family:Georgia,serif;font-size:48px;font-weight:500;letter-spacing:-1.5px;margin:12px 0}header{border-top:5px solid #167364;padding-top:22px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#167364}.intro{font-size:19px;max-width:850px}label{display:inline-flex;gap:12px;align-items:center;margin:12px 28px 20px 0}select{padding:9px;border:1px solid #a9bab0;border-radius:4px;background:white;color:#183b38}table{border-collapse:collapse;width:100%;background:#fffdf7;font-size:14px}th,td{text-align:right;padding:13px 12px;border-bottom:1px solid #dce2d9}th:first-child,td:first-child{text-align:left}th{background:#e8eee5;white-space:nowrap}caption{text-align:left;margin:12px 0;color:#4b6561}.table-wrap{overflow:auto}aside{margin:24px 0;padding:20px 24px;background:#e8eee5;border-left:4px solid #167364}footer{font-size:13px;color:#49615c;border-top:1px solid #a9bab0;padding-top:18px;margin-top:32px}a{color:#136757}#count{font-size:13px;color:#4b6561}@media(max-width:640px){h1{font-size:36px}body{margin-top:24px;padding:0 16px}}</style></head><body>
<header><div class="eyebrow">ReplicateShift / Executed pilot</div><h1>More specimens, or more repeats?</h1>
<p class="intro">A repeat-aware neural model and two statistical baselines, compared under a fixed development measurement budget.</p></header>
<aside>''' + html.escape(f"{config['budget']} development assays per condition · {len(config['seeds'])} independent run seeds · {config['test_specimens']} additional test specimens · nominal coverage {1-config['alpha']:.0%}") + '''<br>Test measurement cost grows with repeats. These synthetic results describe a small exploratory study, not a laboratory allocation recommendation.</aside>
<label>Technical noise <select id="noise"><option value="all">All noise levels</option></select></label>
<label>Model <select id="model"><option value="all">All models</option></select></label><span id="count"></span>
<div class="table-wrap"><table><caption>Values show mean ± Monte Carlo SE across complete runs. Lower Brier score is better. Read coverage together with set size.</caption><thead><tr><th>Model</th><th>Noise SD</th><th>Repeats</th><th>Dev. specimens</th><th>Brier ↓</th><th>Coverage</th><th>Set size</th><th>Test assays</th></tr></thead><tbody id="rows"></tbody></table></div>
<p>Within each seed, pools are paired across allocation policies and independent across fitting, validation, calibration, and testing. Model choices use validation data only. A set containing both labels always covers the true label but offers no label discrimination.</p>
<footer><a href="REPORT.md">Full report and limitations</a> · <a href="summary.csv">Summary CSV</a> · <a href="per_run_metrics.csv">Per-run metrics</a> · <a href="predictions.csv.gz">All test predictions</a> · <a href="manifest.json">Provenance</a><br>Local, self-contained report. No analytics, external resources, or uploaded data.</footer>
<script>const data=''' + data + ''';
const noise=document.getElementById('noise'),model=document.getElementById('model');
for(const x of [...new Set(data.map(r=>r.noise))]){let o=new Option(String(x),String(x));noise.add(o)}
for(const x of [...new Set(data.map(r=>r.model))]){model.add(new Option(x,x))}
const f=(r,k)=>r[k+'_mean'].toFixed(3)+(r[k+'_mcse']===null?'':' ± '+r[k+'_mcse'].toFixed(3));
function render(){const chosen=data.filter(r=>(noise.value==='all'||String(r.noise)===noise.value)&&(model.value==='all'||r.model===model.value));const body=document.getElementById('rows');body.replaceChildren();for(const r of chosen){const tr=document.createElement('tr');for(const v of [r.model,r.noise,r.repeats,r.development_specimens,f(r,'brier'),f(r,'coverage'),f(r,'mean_set_size'),r.test_assays]){let td=document.createElement('td');td.textContent=v;tr.appendChild(td)}body.appendChild(tr)}document.getElementById('count').textContent=chosen.length+' conditions';}noise.addEventListener('change',render);model.addEventListener('change',render);render();</script></body></html>'''


def run(config: dict, output: Path) -> None:
    config = validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output/"protocol.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    sources = source_hashes()
    rows, predictions, provenance = [], [], []
    started = time.perf_counter()
    tracemalloc.start()
    for seed in config["seeds"]:
        streams = np.random.SeedSequence(seed).spawn(7)
        seeds = dict(zip(("train", "validation", "calibration", "test", *MODELS), [int(s.generate_state(1)[0]) for s in streams]))
        sizes = {"train": config["budget"]*3//5, "validation": config["budget"]//5,
                 "calibration": config["budget"]//5, "test": config["test_specimens"]}
        pools = {role: draw_pool(n, max(config["repeats"]), config["features"], seeds[role], role) for role,n in sizes.items()}
        provenance.append({"seed": seed, "role_seeds": seeds, "pools": {role: {
            "n_specimens": len(pool["y"]), "ids": pool["ids"],
            "base_label_noise_sha256": array_hash(pool["base"], pool["y"], pool["noise"])} for role,pool in pools.items()}})
        for noise in config["noise_levels"]:
            for repeats in config["repeats"]:
                counts = {role: n//repeats if role != "test" else n for role,n in sizes.items()}
                samples = {role: pool_subset(pools[role], n, repeats, noise) for role,n in counts.items()}
                xt, mt, yt = samples["train"]
                xv, mv, yv = samples["validation"]
                xc, mc, yc = samples["calibration"]
                xe, me, ye = samples["test"]
                for name in MODELS:
                    if time.perf_counter()-started > config["max_seconds"]:
                        tracemalloc.stop()
                        raise TimeoutError("Pilot exceeded its explicit runtime budget; no completed manifest written")
                    tick = time.perf_counter()
                    if name == "deep_set":
                        scaler = SetStandardizer().fit(xt, mt)
                        model = DeepSetClassifier(seed=seeds[name], epochs=config["epochs"], learning_rate=config["set_learning_rate"],
                            l2=config["l2"], hidden=tuple(config["set_hidden"]), head_width=config["set_head_width"])
                        model.fit(scaler.transform(xt, mt), mt, yt, validation=(scaler.transform(xv, mv), mv, yv))
                        pc = model.predict_proba(scaler.transform(xc, mc), mc)
                        pe = model.predict_proba(scaler.transform(xe, me), me)
                    else:
                        ft = summary_features(xt, name)
                        scaler = Standardizer().fit(ft)
                        model = LogisticClassifier(seed=seeds[name], epochs=config["epochs"], learning_rate=config["logistic_learning_rate"], l2=config["l2"])
                        model.fit(scaler.transform(ft), yt, validation=(scaler.transform(summary_features(xv, name)), yv))
                        pc = model.predict_proba(scaler.transform(summary_features(xc, name)))
                        pe = model.predict_proba(scaler.transform(summary_features(xe, name)))
                    q = conformal_threshold(pc, yc, config["alpha"])
                    rows.append({"seed": seed, "noise": noise, "repeats": repeats, "model": name,
                        "development_assays": sum(counts[r]*repeats for r in ("train","validation","calibration")),
                        "development_specimens": sum(counts[r] for r in ("train","validation","calibration")),
                        "n_train": len(yt), "n_validation": len(yv), "n_calibration": len(yc),
                        "test_assays": len(ye)*repeats, "calibration_rank": math.ceil((len(yc)+1)*(1-config["alpha"])),
                        "threshold": float(q) if np.isfinite(q) else "infinity", "best_epoch": model.best_epoch_,
                        "selected_validation_bce": min(model.validation_loss_history_), "train_prevalence": float(yt.mean()),
                        "test_prevalence": float(ye.mean()), "fit_predict_seconds": time.perf_counter()-tick, **metrics(pe,ye,q)})
                    sets = prediction_sets(pe,q)
                    predictions.extend({"seed":seed,"noise":noise,"repeats":repeats,"model":name,"specimen_id":i,
                        "label":int(y),"p_label_1":float(p),"include_0":int(s[0]),"include_1":int(s[1])}
                        for i,y,p,s in zip(pools["test"]["ids"],ye,pe,sets))
        print(f"Completed allocation seed {seed}",flush=True)
    elapsed = time.perf_counter()-started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summaries, paired = summarize_runs(rows)
    for name, records in (("per_run_metrics.csv",rows),("summary.csv",summaries),("paired_comparisons.csv",paired)):
        write_csv(output/name,records)
    with (output/"predictions.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as zipped:
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as text:
                writer=csv.DictWriter(text,fieldnames=list(predictions[0]))
                writer.writeheader();writer.writerows(predictions)
    (output/"REPORT.md").write_text(build_report(config,summaries,paired))
    (output/"report.html").write_text(html_report(config,summaries))
    if sources != source_hashes():
        raise RuntimeError("Source changed during execution; no completed manifest written")
    manifest={"status":"completed","version":__version__,"created_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"numpy":np.__version__,"platform":platform.system()+" "+platform.machine(),
        "configuration":config,"elapsed_seconds":elapsed,"peak_traced_bytes":peak,
        "memory_scope":"tracemalloc allocations, not full resident or device memory",
        "source_sha256":sources,"role_pool_provenance":provenance,
        "output_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(f"Saved {len(rows)} model-condition runs to {output}; {elapsed:.2f}s, {peak/1024**2:.2f} MiB traced peak",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/allocation_pilot.json"))
    parser.add_argument("--output",type=Path,default=Path("results/local-allocation"))
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.output)


if __name__=="__main__":
    main()
