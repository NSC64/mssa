#!/usr/bin/env python3
"""Opt-in NumPy geometry-head study on shared frozen MSSA context features.

Analytical gradients, no autodiff. Run tests: python3 -m unittest discover -s
scripts -p 'test_curved_head.py'. See docs/CURVED_TOKEN_GEOMETRY.md.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

# Set before importing NumPy; honor explicitly supplied thread settings.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np

SEEDS = (7501, 7502, 7503, 7504, 7505)
RATES = (0.003, 0.01)
CONFIGS = (("dot", 0.0), ("flat", 0.0), ("curved", 0.1), ("curved", 1.0), ("blend", 1.0))


def ball(raw, curvature):
    """Smooth rational chart, NOT an exponential map: raw/sqrt(1+c|raw|²)."""
    scale = 1.0 / np.sqrt(1.0 + curvature * np.sum(raw * raw, axis=1, keepdims=True))
    return raw * scale, scale


def ball_backward(raw, scale, grad, curvature):
    return scale * grad - curvature * scale**3 * raw * np.sum(raw * grad, axis=1, keepdims=True)


def distance(q, p, curvature):
    """d_c(q,p)^2/4 and intermediates, including the exact flat branch."""
    delta = np.maximum(np.sum(q*q, axis=1)[:, None] + np.sum(p*p, axis=1)[None, :] - 2*q@p.T, 0.0)
    if curvature == 0:
        return delta, (delta, None, None, None)
    a = 1.0 - curvature * np.sum(q*q, axis=1)[:, None]
    b = 1.0 - curvature * np.sum(p*p, axis=1)[None, :]
    if np.any(a <= 1e-12) or np.any(b <= 1e-12):
        raise FloatingPointError("point too close to the Poincare boundary")
    root = np.sqrt(curvature * delta / (a*b))
    arc = np.arcsinh(root)
    ratio = np.divide(arc, root, out=np.ones_like(root), where=root > 1e-10)
    factor = ratio / np.sqrt(1.0 + root*root) / (a*b)
    return arc*arc / curvature, (delta, a, b, factor)


def distance_backward(q, p, cache, grad, curvature):
    delta, a, b, factor = cache
    weighted = grad if curvature == 0 else grad * factor
    q_coeff = np.sum(weighted, axis=1, keepdims=True)
    p_coeff = np.sum(weighted, axis=0)[:, None]
    if curvature:
        q_coeff += np.sum(weighted * curvature * delta / a, axis=1, keepdims=True)
        p_coeff += np.sum(weighted * curvature * delta / b, axis=0)[:, None]
    return 2*(q_coeff*q - weighted@p), 2*(p_coeff*p - weighted.T@q)


def probabilities(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    normalizer = np.sum(exps, axis=1, keepdims=True)
    return exps / normalizer, shifted - np.log(normalizer)


class Head:
    def __init__(self, width, rank, vocab, seed, kind, curvature, counts):
        self.kind, self.curvature = kind, curvature
        rng = np.random.default_rng(seed)
        self.params = {
            "projection": rng.normal(0, 0.2 / np.sqrt(width), (width, rank)),
            "tokens": rng.normal(0, 0.1, (vocab, rank)),
            "bias": np.log((counts + 1.0) / np.sum(counts + 1.0)),
        }
        if kind == "blend":
            self.params["gate"] = np.zeros(1)

    def forward(self, features):
        u = features @ self.params["projection"]
        v = self.params["tokens"]
        if self.kind == "dot":
            energy = 2 * u @ v.T
            cache = (u, v)
        else:
            flat, flat_cache = distance(u, v, 0.0)
            if self.kind == "flat":
                energy, cache = -flat, (u, v, flat_cache)
            else:
                q, qs = ball(u, self.curvature)
                p, ps = ball(v, self.curvature)
                curved, curved_cache = distance(q, p, self.curvature)
                gate = 1.0 if self.kind == "curved" else 1.0/(1.0+np.exp(-self.params["gate"][0]))
                energy = -(1-gate)*flat - gate*curved
                cache = (u, v, flat_cache, q, p, qs, ps, curved_cache, flat, curved, gate)
        logits = energy + self.params["bias"]
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("non-finite output logits")
        return logits, cache

    def loss_grad(self, features, targets):
        logits, cache = self.forward(features)
        probs, log_probs = probabilities(logits)
        loss = -np.mean(log_probs[np.arange(len(targets)), targets])
        # Preserve the target tail instead of subtracting a saturated p from 1.
        error = probs.copy()
        error[np.arange(len(targets)), targets] = 0
        error[np.arange(len(targets)), targets] = -np.sum(error, axis=1)
        error /= len(targets)
        grads = {"bias": np.sum(error, axis=0)}
        u, v = cache[:2]
        if self.kind == "dot":
            gu, gv = 2*error@v, 2*error.T@u
        elif self.kind == "flat":
            gu, gv = distance_backward(u, v, cache[2], -error, 0.0)
        else:
            _, _, flat_cache, q, p, qs, ps, curved_cache, flat, curved, gate = cache
            gq, gp = distance_backward(q, p, curved_cache, -gate*error, self.curvature)
            gu = ball_backward(u, qs, gq, self.curvature)
            gv = ball_backward(v, ps, gp, self.curvature)
            if self.kind == "blend":
                fu, fv = distance_backward(u, v, flat_cache, -(1-gate)*error, 0.0)
                gu += fu
                gv += fv
                grads["gate"] = np.array([np.sum(error*(flat-curved))*gate*(1-gate)])
        grads["projection"], grads["tokens"] = features.T@gu, gv
        if not all(np.all(np.isfinite(x)) for x in grads.values()):
            raise FloatingPointError("non-finite head gradient")
        return float(loss), grads


def evaluate(head, features, targets):
    loss, correct, max_radius = 0.0, 0, 0.0
    for start in range(0, len(targets), 256):
        x, y = features[start:start+256], targets[start:start+256]
        logits, _ = head.forward(x)
        _, log_probs = probabilities(logits)
        loss -= np.sum(log_probs[np.arange(len(y)), y])
        correct += np.count_nonzero(np.argmax(logits, axis=1) == y)
        if head.curvature:
            q, _ = ball(x@head.params["projection"], head.curvature)
            max_radius = max(max_radius, float(np.max(np.sqrt(head.curvature*np.sum(q*q, axis=1)))))
    ce = float(loss / len(targets))
    return {"cross_entropy": ce, "perplexity": math.exp(ce), "accuracy": correct/len(targets),
            "targets": len(targets), "max_dimensionless_query_radius": max_radius}


def train_head(head, splits, indices, rate):
    x, y = splits["train"]
    first = evaluate(head, *splits["dev"])
    best = (first["cross_entropy"], 0, {k: v.copy() for k,v in head.params.items()})
    curve = [{"update": 0, "development": first}]
    moments = {k: (np.zeros_like(v), np.zeros_like(v)) for k,v in head.params.items()}
    seconds = 0.0
    wall = time.perf_counter()
    for step, batch in enumerate(indices, 1):
        at = time.perf_counter()
        loss, grads = head.loss_grad(x[batch], y[batch])
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        clip = 1/max(float(norm), 1.0)
        warmup = min(32, len(indices))
        multiplier = step/warmup if step <= warmup else 0.1+0.45*(1+np.cos(np.pi*(step-warmup)/(len(indices)-warmup)))
        for key, grad in grads.items():
            m, v = moments[key]
            g = grad*clip
            m *= 0.9; m += 0.1*g
            v *= 0.999; v += 0.001*g*g
            head.params[key] -= rate*multiplier*(m/(1-0.9**step))/(np.sqrt(v/(1-0.999**step))+1e-8)
        seconds += time.perf_counter()-at
        if step % 256 == 0 or step == len(indices):
            metrics = evaluate(head, *splits["dev"])
            curve.append({"update": step, "target_exposures": step*len(batch), "training_seconds": seconds,
                          "last_batch_loss": loss, "development": metrics})
            if metrics["cross_entropy"] < best[0]:
                best = (metrics["cross_entropy"], step, {k:v.copy() for k,v in head.params.items()})
    head.params = best[2]
    return {"kind": head.kind, "curvature": head.curvature, "rate": rate, "curve": curve,
            "best_development_ce": best[0], "selected_update": best[1], "training_seconds": seconds,
            "training_and_development_seconds": time.perf_counter()-wall,
            "parameter_count": sum(x.size for x in head.params.values())}


def benchmark(head, features):
    for _ in range(3):
        head.forward(features[:256])
    samples = []
    for _ in range(7):
        at = time.perf_counter()
        for start in range(0, min(1024,len(features)), 256):
            probabilities(head.forward(features[start:start+256])[0])
        samples.append(time.perf_counter()-at)
    return {"batch_targets": min(1024,len(features)), "seconds_samples": samples,
            "median_seconds": float(np.median(samples))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    corpus_hash = digest(args.corpus)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                   "thread_settings": {n:os.environ[n] for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                   "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        environment["cpu"] = next(s.split(":",1)[1].strip() for s in cpuinfo.read_text().splitlines() if s.startswith("model name"))
    results = []
    for seed in args.seeds:
        feature_dir = out / f"features-{seed}"
        subprocess.run([str(args.exporter.resolve()),str(args.corpus.resolve()),str(feature_dir),str(seed)],check=True)
        meta = json.loads((feature_dir/"metadata.json").read_text())
        splits = {}
        hashes = {}
        for split in ("train","dev","test"):
            xf, yf = feature_dir/f"{split}.f32", feature_dir/f"{split}.u32"
            x = np.fromfile(xf,dtype="<f4").reshape(-1,meta["feature_width"]).astype(np.float64)
            y = np.fromfile(yf,dtype="<u4").astype(np.int64)
            if len(x)!=len(y) or not np.all(np.isfinite(x)) or np.any(y>=meta["vocab"]):
                raise ValueError("invalid feature cache")
            splits[split] = (x,y)
            hashes[split] = {"features":digest(xf),"targets":digest(yf)}
        train_x, train_y = splits["train"]
        mean, scale = train_x.mean(axis=0), np.maximum(train_x.std(axis=0),1e-6)
        splits = {s:((x-mean)/scale,y) for s,(x,y) in splits.items()}
        rng = np.random.default_rng(seed ^ 0x48454144)
        indices = rng.integers(0,len(train_y),size=(args.updates,64))
        index_hash = hashlib.sha256(indices.astype("<u4").tobytes()).hexdigest()
        counts = np.bincount(train_y,minlength=meta["vocab"])
        trials, selected = [], []
        configs = [(kind,c,rate) for kind,c in CONFIGS for rate in RATES]
        rng.shuffle(configs)
        trained = {}
        for kind,c,rate in configs:
            print(f"seed={seed} kind={kind} curvature={c:g} rate={rate:g}",flush=True)
            head = Head(meta["feature_width"],args.rank,meta["vocab"],seed,kind,c,counts)
            record = train_head(head,splits,indices,rate)
            trials.append(record)
            key = (kind,c)
            if key not in trained or record["best_development_ce"] < trained[key][1]["best_development_ce"]:
                trained[key] = (head,record)
        for head,record in trained.values():
            item = {k:v for k,v in record.items() if k!="curve"}
            item["test"] = evaluate(head,*splits["test"])
            item["timing"] = benchmark(head,splits["test"][0])
            if head.kind=="blend":
                item["curved_gate"] = float(1/(1+np.exp(-head.params["gate"][0])))
            selected.append(item)
            np.savez(out/f"head-{seed}-{head.kind}-{head.curvature:g}.npz", **head.params,
                     feature_mean=mean,feature_scale=scale)
        worker = {"seed":seed,"feature_metadata":meta,"feature_sha256":hashes,"batch_index_sha256":index_hash,
                  "trials":trials,"selected":selected}
        (out/f"seed-{seed}.json").write_text(json.dumps(worker,indent=2,allow_nan=False)+"\n")
        results.append(worker)
    aggregate = []
    for kind,c in CONFIGS:
        rows = [next(r for r in w["selected"] if r["kind"]==kind and r["curvature"]==c) for w in results]
        dot = [next(r for r in w["selected"] if r["kind"]=="dot") for w in results]
        deltas = np.array([r["test"]["cross_entropy"]-d["test"]["cross_entropy"] for r,d in zip(rows,dot)])
        bootstrap = np.random.default_rng(913).choice(deltas,size=(4096,len(deltas)),replace=True).mean(axis=1)
        aggregate.append({"kind":kind,"curvature":c,"pairs":len(rows),"parameter_count":rows[0]["parameter_count"],
                          "mean_test_ce":float(np.mean([r["test"]["cross_entropy"] for r in rows])),
                          "mean_test_accuracy":float(np.mean([r["test"]["accuracy"] for r in rows])),
                          "paired_ce_delta_vs_dot":float(deltas.mean()),"paired_ce_delta_95pct":np.quantile(bootstrap,[.025,.975]).tolist(),
                          "ce_better_than_dot_pairs":int(np.count_nonzero(deltas<0)),
                          "median_head_training_seconds":float(np.median([r["training_seconds"] for r in rows])),
                          "median_head_inference_seconds":float(np.median([r["timing"]["median_seconds"] for r in rows]))})
    record = {"corpus_sha256":corpus_hash,"environment":environment,"rank":args.rank,"head_updates":args.updates,
              "batch_size":64,"seeds":args.seeds,"rate_grid":RATES,"configs":CONFIGS,
              "source_sha256":{str(p):digest(p) for p in (Path(__file__),Path("examples/curved_features.rs"))},
              "aggregate":aggregate,"workers":results}
    (out/"results.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    lines = ["# Curved token-head results", "", "Frozen shared MSSA contexts; development-only checkpoint and rate selection.", "",
             "| Head | Curvature c | Parameters | Test CE | Accuracy | CE delta vs dot | Paired 95% interval | Head training s | Head inference s/1024 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for r in aggregate:
        lines.append(f"| {r['kind']} | {r['curvature']:g} | {r['parameter_count']} | {r['mean_test_ce']:.6f} | {r['mean_test_accuracy']:.2%} | {r['paired_ce_delta_vs_dot']:+.6f} | {r['paired_ce_delta_95pct']} | {r['median_head_training_seconds']:.3f} | {r['median_head_inference_seconds']:.6f} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--exporter",type=Path,default=Path("target/release/examples/curved_features"))
    parser.add_argument("--updates",type=int,default=1024)
    parser.add_argument("--rank",type=int,default=8)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    args = parser.parse_args()
    if args.rank<=0 or args.updates<=0 or len(set(args.seeds))!=len(args.seeds):
        parser.error("rank and updates must be positive; seeds must be distinct")
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        run(args)


if __name__ == "__main__":
    main()
