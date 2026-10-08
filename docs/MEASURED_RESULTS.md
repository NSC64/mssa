# Measured quality, throughput, and certificate fallback

Measured October 8, 2026 on an Intel Core i7-7500U CPU using the normal release
profile. These tables replace proxy-only performance statements with elapsed-time
measurements. They accompany the local, unpublished sparse-inference manuscript;
the manuscript's **2.171x** is a conditional coordinate-work calculation.

The current tables were rerun after upstream integration using GCSR with four-slot
memory groups. The benchmark source and raw JSON remain reproducible from the
command below; timing artifacts are kept outside the repository.

## Reproduce

```bash
cargo run --release -- benchmark --feature interdiffusion --out __agent__/review_results
cargo run --release -- benchmark --feature sparse --out __agent__/review_results
```

Both commands generate `summary.md`. `interdiffusion.json` contains all training
trials, paired target-token checkpoints, stream digests, and uncertainty estimates.
`sparse_inference.json` contains individual timing batches, certificate counts,
index-build times, and fidelity checks. The measured local artifacts are at
`/tmp/omnirush/mssa-review-confirmatory/`.

Completed worker records survive an interrupted training comparison. After
rerunning a missing worker, `benchmark --feature interdiffusion_report --out PATH`
regenerates the aggregate, verifying seed, stream, exposure and schedule
compatibility. Use a new directory for a changed optimizer/protocol.

## 1. Matched-token cross entropy and actual training throughput

The paired seeds are **7401, 7402, 7403, 7404, 7405**. Optimizer development used
7301–7305 before freezing the implementation. Both AdamW and full Interdiffusion
receive the same four peak-rate candidates: **0.001, 0.003, 0.01, 0.08**. Lower
rates were added after development overfitting was detected. These are transparent
exploratory experiments, not preregistered confirmatory claims.

Every trial has 2,048 updates, 32-update linear warmup, cosine decay to 10% of
peak, zero weight decay, identical initialization for each pair, the same document
resets, and a frozen empty episodic bank. Learning-rate selection uses only final
development CE. Test curves never select or rerank a trial. Data-stream digests,
target-token counts, tokenizer IDs, model configurations and schedule multipliers
are checked against the paired AdamW record before generating the report.

Training seconds include actual forward passes, derivatives/probes, validation,
clipping and optimizer updates. Total wall time also includes data handling and
periodic development/test scoring. Throughput counts **scheduled target tokens**;
extra probe passes are reported separately and cannot inflate the numerator.
The baseline uses the existing scalar/reference CPU trainer, not a benchmark of
an optimized CUDA or batched AdamW implementation.

| Task / method | Paired seeds | Mean final test CE ↓ | Median training seconds | Median target tokens/s ↑ | Median total wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cycle / AdamW | 5 | 2.478190e-5 | 0.4219 | 77,675 | 0.4328 |
| Cycle / Interdiffusion | 5 | **3.075881e-15** | 0.8331 | 39,332 | 0.8467 |
| Recall / AdamW | 5 | 0.387383 | 0.1750 | 94,234 | 0.1849 |
| Recall / Interdiffusion | 5 | **0.385568** | 0.2144 | 76,944 | 0.2239 |
| Byte text / AdamW | 5 | 3.925210 | 1.6840 | 17,925 | 1.7647 |
| Byte text / Interdiffusion | 5 | **3.010211** | 2.3154 | 13,043 | 2.4045 |

These are per-seed development-selected trials; all candidate runs are retained.
The median total four-candidate search costs are respectively 1.9963/2.3308 s
(cycle), 0.7730/0.9521 s (recall), and 7.9592/8.4160 s (byte text), AdamW/v2.
Medians of paired speed ratios differ from ratios of marginal medians.

### Exposure and held-out splits

- **Cycle:** fixed shared token IDs, vocabulary 8; 1,812 parameters. Each
  32-token training document uses two 16-target chunks with detached carry
  between updates. Every seed sees **32,768 scheduled target tokens**. Development
  length is 24; test length is 48. Both methods have 100% test last-token accuracy.
- **Recall:** the same small model and shared IDs. Training distractor gaps are
  2–10, development gaps 8/10, and held-out gaps 11–14. Target exposure for the
  five pairs is **16,424 / 16,429 / 16,493 / 16,531 / 16,226**. Both methods have
  100% held-out recall accuracy.
- **Byte text:** UTF-8 bytes map to `byte + 1`, with ID zero reserved; vocabulary
  257, latent width 24, state width 4, chunk length 16. There is no learned
  tokenizer or test-vocabulary leakage. The built-in nine-line science corpus
  uses lines 1–5 for training, 6–7 for development, 8–9 for testing, with resets
  per document and carry across chunks. Target exposure is **30,185 / 30,099 /
  30,201 / 30,200 / 30,097**. This is a tiny local-text diagnostic, not a
  corpus-scale language-model result.

The JSON records test CE at every 128-update checkpoint alongside exact target
exposure and the paired AdamW CE. Thus final and intermediate comparisons match
training tokens, not merely elapsed time or nominal epochs.

### Paired outcomes and uncertainty

| Task | V2 lower test CE | V2 within 5% of AdamW CE and at least its last-token accuracy | Paired mean CE difference, v2 − AdamW | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | --- |
| Cycle | 3/5 | 3/5 | −2.47819e-5 | [−7.43454e-5, 4.42396e-15] |
| Recall | 1/5 | 4/5 | −0.001815 | [−0.041002, 0.021087] |
| Byte text | **5/5** | 4/5 | **−0.914999** | [−1.288015, −0.588515] |

Intervals resample the five **paired differences**, using 4,096 deterministic
bootstrap draws. Five pairs are a small sample; overlapping zero on cycle/recall
does not establish a statistically reliable universal improvement.

| Cycle seed | AdamW test CE | Interdiffusion test CE |
| --- | ---: | ---: |
| 7401 | 1.091e-15 | 0.0 (numeric underflow) |
| 7402 | 7.646e-10 | 7.014e-44 |
| 7403 | 4.728e-65 | 5.959e-65 |
| 7404 | 1.239e-4 | 1.770e-43 |
| 7405 | 3.774e-15 | 1.538e-14 |

Cycle loss is substantially better **on average**, but not on every pair. Both
methods already solve this deterministic task at 100% accuracy. Stable binary64
CE uses `ln_1p` for confident tails rather than rounding `ln(1 + tiny_tail)` to
zero; even binary64 eventually underflows. Very small cycle CE is not evidence
of better general reasoning or a meaningful trillion-fold intelligence gain.

### What improved, and what did not

V2 uses rotating cosine eligibility plus exact local derivatives. Unit-L2
cosine-coordinate steps are scaled by `sqrt(width)` to remove the earlier
per-coordinate shrinkage. Its readout accumulates diagonal CE curvature and
uses a trust-radius-clipped curvature update on confident chunks (`CE < 0.1`),
retaining local Adam on ambiguous chunks. Gradients and curvature preserve the
tiny target error directly rather than computing `probability - 1` at saturation.

CLI-shape owned numeric storage is **19,780,080 bytes / 18.86 MiB**, versus
AdamW's **35,468,224 bytes / 33.83 MiB**: **44.23% less**. Counts include curvature,
gradients, moments, traces and scratch; learning above uses smaller models.
Observed worker peaks are 23,968 versus 31,456 KiB. Numeric capacity and resident
memory are distinct; model construction is excluded from training throughput.

The current implementation **is slower per scheduled training token**. Median
paired speedups to the strict final AdamW development-quality target are 0.35x
(cycle) and 0.79x (recall). Byte text misses one target, so no all-seed speedup is
reported. Earlier three-seed constant-rate time-to-quality results do not
establish a current speed advantage under this stronger protocol.

## 2. Certificate termination and measured inference speed

The inference audit also uses five paired seeds and the paper shape:
`V=2048`, `d=256`, state width 16, key width 32, **512 populated memory slots**.
Each seed has five timing rounds, rotating the four modes' order, with 256
teacher-forced input tokens per round and resets every 32 tokens. All modes
share exactly the same frozen model and memory; every fallback is timed.
Index construction is excluded from steady-state timing and reported separately.
Build-inclusive single-batch speed is retained in the JSON.

| Workload / mode | Pairs | Median tokens/s ↑ | Median paired wall speedup | CSR exact fallback | CVP full scan |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained bank / dense | 5 | 1,610 | 1.000x | — | — |
| Trained bank / CSR | 5 | 2,162 | **1.322x** | **0.00%** | — |
| Trained bank / CVP | 5 | 2,416 | **1.504x** | — | **0.16%** |
| Trained bank / dual | 5 | 3,837 | **2.411x** | **0.00%** | **0.16%** |
| Held-out bank / dense | 5 | 1,922 | 1.000x | — | — |
| Held-out bank / CSR | 5 | 1,831 | **1.000x** | **17.19%** | — |
| Held-out bank / CVP | 5 | 2,891 | **1.567x** | — | **0.23%** |
| Held-out bank / dual | 5 | 2,864 | **1.700x** | **17.19%** | **0.23%** |
| Diffuse fixture / dense | 5 | 2,008 | 1.000x | — | — |
| Diffuse fixture / CSR | 5 | 1,955 | **0.984x** | **100.00%** | — |
| Diffuse fixture / dual | 5 | 1,876 | **0.970x** | **100.00%** | **100.00%** |
| Separated fixture / dense | 5 | 1,919 | 1.000x | — | — |
| Separated fixture / CSR | 5 | 2,283 | **1.189x** | **0.00%** | — |
| Separated fixture / dual | 5 | 4,298 | **2.262x** | **0.00%** | **0.00%** |

For the trained bank, median index construction costs are **3.892 ms** (CSR),
**3.041 ms** (CVP), and **7.456 ms** (dual). Each is a separate measured build,
so dual's median need not equal the sum of the other marginal medians. The
trained-bank CSR build-inclusive speedup is **1.284x**; the held-out-bank value is
**0.973x**, so the steady-state trained-bank gain does not establish a general
held-out speedup.

CSR termination means it certified omitted mass before scanning the full bank;
fallback means exhaustive retrieval. CVP termination means at least one output
row was pruned; full scan means no row was pruned. Counts are recorded separately
for every seed. Empty banks do not inflate the success rate.

### Workloads and fidelity

- **Trained bank:** 32 reference AdamW updates × 16 targets on the built-in science
  corpus at constant rate 0.003, then 512 naturally generated query/value pairs
  inserted into a frozen bank. UTF-8 byte IDs are shared; the 2048-row head
  retains unused rows beyond the 257-byte vocabulary to match the paper shape.
  Those unused rows can make output pruning easier than a fully used 2048-token
  vocabulary. This is a small trained-model probe, not a representative LLM.
- **Diffuse fixture:** randomly initialized weights and a random diffuse bank;
  tests the genuine all-fallback cost.
- **Held-out bank:** trained and populated from the first five corpus lines, then
  evaluated on the remaining four; it is the main check against replay-only
  clustering behavior.
- **Separated fixture:** deliberately engineered separated memory and vocabulary
  rows; demonstrates the fast path, not trained-model generality.

Every workload/mode measured **100% greedy agreement**. Trained-bank, diffuse,
and separated rows had zero observed CE delta. Held-out GCSR's worst observed
absolute CE delta was `1.628e-5`, with maximum actual omitted mass `2.929e-3`
against a requested `0.01` bound and maximum absolute logit error `4.227e-2`.
This is expected: the memory certificate bounds omitted retrieval softmax mass;
it does not certify unchanged logits, cross-entropy, or sampled-token identity.
CVP IDs are additionally checked against the exhaustive head on identical
features. Normalized CE always uses the full vocabulary projection: CVP currently
certifies **greedy selection**, not normalized cross entropy or temperature
sampling. Quality scoring is separate from timed greedy batches.

### Does 2.171x survive?

The current GCSR result is a measured trained-bank gain: CSR is **1.322x** and
dual is **2.411x**. The held-out CSR row is only **1.000x** steady state and
**0.973x** including index build, while diffuse CSR is slower than dense. The
result therefore supports a replay-like trained-bank speedup, not a general
trained-model claim. The historical `2.171x` number remains a conditional
coordinate-work proxy rather than a universal throughput result.

Repeated bound computation during sorting, redundant full-distance rescanning
on fallback, and mixing values for omitted zero-weight rows were removed.
Bounds are cached once per query; fallback reuses exact distances with the
production accumulation order; exact vocabulary rows use the same SIMD dot
kernel as dense inference, with conservative numerical bound guards. Routing,
sorting and fallback still incur costs, all present in the measured batches.

Certificate sparsity reduces **read/aggregation work**, not allocated trained
parameter memory. It adds index storage. Interdiffusion's training-memory saving
is independently measured and does not depend on certificate termination.

## Verification

- After upstream integration, 424 library tests passed and two were ignored; all
  190 integration tests passed in the serial verification run.
- CUDA-feature release test compilation passed. Integration checks used release
  optimization with LTO disabled and 16 codegen units; the CPU timing tables
  above used the normal release profile.
- Tests cover curvature versus finite differences, projected derivatives versus
  TBPTT, failed-update rollback, sparse scan counts, exact fallback accumulation,
   and CVP agreement with SIMD dense logits across five random seeds.
- WebGPU forward/gradient parity, packed-lane parity, and strict training-shape
  forward/backward dispatch passed on an NVIDIA GeForce 940MX using Vulkan.
  Shader syntax is also validated without requiring an adapter. Integration
  fixed invalid shader syntax, storage-buffer limits, recurrent-stage bindings,
  and unused EGL backend discovery during concurrent/live-context initialization.
- No accelerator speed, energy, long-form corpus quality, or universal per-seed
  improvement is established by these local CPU experiments.
