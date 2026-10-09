# Interdiffusion: tape-free cosine/eligibility learning

Interdiffusion v2 is an opt-in CPU experiment combining local analytic gradients,
streaming forward eligibility, and cosine-coordinate updates. It learns without
a reverse-time activation tape. The current five-seed audit reduces owned numeric
storage by 44.23% and improves mean cycle and byte-text test loss against an
exposure/schedule-matched AdamW baseline. Measured training throughput is lower;
strict loss parity on every seed and task remains incomplete. See
[`MEASURED_RESULTS.md`](MEASURED_RESULTS.md) for the latest wall-clock and paired
tables. Older three-seed measurements below are historical evidence.

`ZerothOrderTrainer` retains v1's pure tensor-block spectral probes. The v2 trainer
is a hybrid derivative-based method: it owns local gradients and optimizer state,
which are included in every storage result. Historical v1 evidence is retained
below alongside the new same-protocol ablations.

## Run the experiment

```bash
cargo run --release -- benchmark \
  --feature interdiffusion \
  --out __agent__/interdiffusion_results
```

The runner starts nine separate method processes: AdamW, SGD, plain zeroth-order,
uniform spectral ZO, smoothed v1, readout-only, readout plus generic probes,
recurrent-only eligibility, and full v2 including input eligibility. Output:

- `interdiffusion.json`: protocol, all trials, selected trials, loss/accuracy curves,
  per-seed AdamW-quality crossings, and aggregates.
- `summary.md`: quality, matched-quality time, tuning time, storage, and process RSS.
- One method directory with its complete worker record.

Each worker uses five paired seeds `7401`–`7405` and 2,048 updates per trial.
AdamW and full Interdiffusion share peak-rate candidates `0.001`, `0.003`, `0.01`,
and `0.08`, including lower rates to avoid development overfitting. Interdiffusion
uses the same head/body base rate. Readout/probe/recurrent ablations use `0.01`
and `0.08`; SGD/pure-ZO use `0.02` and `0.08`. Every method uses 32-update linear
warmup followed by cosine decay to 10% of its peak. Development and frozen test
curves are scored every 128 updates; scoring preserves carry and memory.
The final development loss selects the learning rate separately for each
method/task/seed. Test results never select a trial. All candidate runs remain
in the record, including candidates that perform poorly.

## Use it for ordinary MSSA training

The normal training command accepts an explicit optimizer choice:

```bash
mssa train data/train.txt \
  --optimizer interdiffusion \
  --tokenizer bpe \
  --batch-size 1 --accumulate 1 \
  --out data/interdiffusion.pssa
```

This checkpoint-capable path is CPU-only and serial. It writes a regular `.pssa`
checkpoint, so inference and scoring use the normal MSSA commands. Resume it
with the same optimizer flag:

```bash
mssa train data/train.txt \
  --optimizer interdiffusion \
  --batch-size 1 --accumulate 1 \
  --resume data/interdiffusion.pssa \
  --out data/interdiffusion-02.pssa
```

Readout Adam moments are restored from the checkpoint. Eligibility/probe scratch
and body optimizer state are rebuilt on resume; an uninterrupted run is not
bit-equivalent to a resumed run. Packed batches, gradient accumulation other
than `--accumulate 1`, dream replay, loss CSV logging, and repeated Ouro loops
currently reject explicitly rather than silently using different semantics.
AdamW remains available with
`--optimizer adamw`.

## Lightweight runtime and library API

`ForwardModel` initializes exactly the same weights, timescales and adapters as
the existing MSSA constructor, including its random draw order. It allocates
weights, recurrent carries, bounded episodic banks, inference buffers and a
single streaming logits row. Legacy gradient arrays, Adam moments, reverse-mode
tapes, scan workspaces and chunk-wide output arrays are empty from construction;
they are not allocated and subsequently discarded.

V2 adds its own local gradient/moment arrays, bounded forward tangents, cosine
workspace, and scalar input-row RMS statistics. Cross entropy and readout
gradients accumulate online; there is no chunk-wide feature or error cache.
Activation slopes are computed once per token and reused across tangents.

The public wrapper exposes streaming loss/evaluation and storage information.
Its internal model is private because the legacy backward and checkpoint APIs
require optimizer arrays. This prototype supports CPU, one temporal loop, and
independently weighted depth blocks. Forward eligibility is supported for depth
one with an empty episodic bank. Stacked/populated-bank models use local readout
learning and generic spectral probes; their quality has not been established by
this benchmark.

An application can experiment with its own token arrays through the library:

```rust
use pssa::interdiffusion::{InterdiffusionConfig, InterdiffusionTrainer};
use pssa::pssa::PSSAConfigV2;

fn main() -> Result<(), String> {
    let cfg = PSSAConfigV2 { weight_decay: 0.0, ..Default::default() };
    let options = InterdiffusionConfig {
        head_learning_rate: 0.01,
        body_learning_rate: 0.0025,
        ..Default::default()
    };
    let mut trainer = InterdiffusionTrainer::new(cfg, 42, options)?;
    let inputs = [1, 2, 3, 4];
    let targets = [2, 3, 4, 5];
    let report = trainer.train_step(&inputs, &targets, true, false)?;
    let held_out_loss = trainer.forward.evaluate(&[5, 6, 1, 2], &[6, 1, 2, 3])?;
    println!("training={} evaluation={held_out_loss}", report.loss);
    Ok(())
}
```

The example demonstrates the API; meaningful evaluation needs separate data.
`train_step` arguments after the token arrays select document reset and optional
episodic writing. `evaluate` preserves incoming carry and never writes memory.
Streaming evaluation length is independent of the historical tape length. V2
updates are bounded by `chunk_len`; longer training documents can use successive
chunks with `reset = false` after the first chunk. Incoming carry is detached at
each update. New optimizer state is not exposed through legacy checkpoint APIs.

## Fused readout statistics

The v2 base pass now computes the stabilized softmax partition once per token:
the maximum logit, the target exponential, the sum of non-target exponentials,
and the total normalization. It reuses those values for stable cross entropy,
the readout error, and diagonal CE curvature instead of recomputing the same
exponentials in each consumer. The target curvature deliberately retains the
`other / sum` form rather than `1 - probability`, preserving representable
confident-tail values.

This is an algebraic implementation optimization, not a new optimizer or a
changed objective. A before/after release run on an Intel Core i7-7500U with
five paired seeds (`7401`–`7405`) produced identical selected development/test
curves for all 15 Interdiffusion task/seed trials. The single-run median
training-only times were:

| Task | Previous pass | Fused pass | Fused target tokens/s |
| --- | ---: | ---: | ---: |
| Cycle | `0.8809 s` | `0.8090 s` | `40,506` |
| Delayed recall | `0.2340 s` | `0.2190 s` | `75,006` |
| Byte text | `2.0469 s` | `1.8073 s` | `16,710` |

These are paired engineering measurements, not a repeated performance study;
OS scheduling and release-process startup can move short timings. The normal
feature benchmark remains the source of quality, storage, and method-vs-AdamW
claims.

## V2 learning rule

- **Readout, C projection, adapters, and MLP:** exact current-token local
  derivatives, accumulated over the chunk, with explicit Adam state. Local MLP
  and adapter differentiation does not traverse previous tokens.
  The readout uses a clipped diagonal CE-curvature direction on confident chunks
  (`CE < 0.1`) and local Adam on uncertain chunks, preserving moments in both
  paths. Curvature is accumulated online, not from an activation-history tape.
- **Cell A:** independent forward eligibility per recurrent cell. Each trace
  propagates through that cell's decay and accumulates its current-parameter
  derivative; current-token loss derivatives contract with the traces.
- **Delta/B:** one independent orthonormal cosine coordinate per projection row,
  rotated through the complete frequency band, with forward tangents and
  coordinate-specific Adam moments. This supplies multiple row-wise learning
  signals instead of one global finite-difference slope.
  Adaptive steps are multiplied by `sqrt(width)`, so a unit-L2 basis direction
  has unit-RMS update scale rather than shrinking each coordinate as width grows.
- **Input embeddings and affine RMSNorm:** up to eight distinct input rows and
  two norm directions carry forward tangents through normalization, Delta/B/C,
  the recurrent cells, and the adapter. The row budget is fixed; selected rows
  rotate through the current document's sorted unique tokens. Per-row visit
  counters rotate frequencies even with sparse or periodic token occurrence.
  A single bias-corrected RMS statistic per row preconditions updates; momentum
  is not mixed across unrelated coordinates and dense embedding moments are
  not allocated.

Parameters stay fixed during the base pass. Traces start at zero at the chunk
edge, including when incoming carry is nonzero. The tested forward gradients
and selected directional derivatives match the equivalent TBPTT derivatives;
the optimizer itself is not full-gradient AdamW.

Defaults are head/body rate `0.01`, clipping norm `1`, eligibility/input eligibility
and readout curvature/RMS scaling enabled, and fallback cadence `16`.
`curvature_readout = false` retains local readout Adam;
`coordinate_rms_scaling = false` retains the earlier cosine step scale.
`set_learning_rates(head, body)`
updates a caller-owned schedule without resetting moments.
`body_every = 0` freezes the body for the
readout-only ablation. `input_eligibility = false` keeps recurrent eligibility but
uses intermittent embedding/norm probes; `eligibility = false` uses generic
probes for the body. Epsilon `0.001`, eight sampled modes, and smoothing `0` affect
these probes. Eligible full v2 performs **one forward pass per update**.

Local Adam updates use `PSSAConfigV2`'s beta/epsilon/decay settings. Projected
cosine/input and finite-difference updates use data-loss directions without
decoupled weight decay. Benchmarks and the API example use zero decay. Clipping
is separate for the readout, eligibility body, and any active probe tensor;
backpropagation baselines use one global norm clip.

### Carry, memory, and failure handling

The base pass caches only terminal carry and memory-write keys/values. Fallback
probes restore the same incoming carry, see frozen memory, and restore the active
tensor exactly. The cached unperturbed state supplies the commit, so v2 avoids
v1's extra original-weight replay. Memory writes follow the existing loss
threshold/refractory policy and precede weight updates.

All candidate updates are validated before publishing memory, moments, row
statistics, or optimizer steps. A failed probe or late input validation restores
exact weights and incoming carry. If memory becomes occupied, the generic
fallback grows reusable probe scratch on demand; increased capacities are
included in storage accounting.

## V1 pure zeroth-order rule

Each update visits one parameter tensor in a deterministic cyclic order. Plain
and spectral optimizers use the same ordering, and all trainable tensors remain
reachable. The experiment is block-coordinate normalized-direction SPSA, rather
than a reproduction of full-model Gaussian MeZO.

For an active `R x C` tensor, the spectral direction is a sum of up to eight
separable, orthonormal DCT-II modes. This is a real Fourier/cosine basis, evaluated
directly with reusable row/column basis buffers. Frequencies are sampled without
replacement across the complete two-dimensional band at every update.

```text
u = sum_k a_k * cosine_row_k * cosine_column_k
a_k = Gaussian_k / sqrt(1 + smoothing * normalized_frequency_k_squared)
u = u / RMS(u)

positive_loss = loss(weights + epsilon * u, incoming_carry, frozen_memory)
negative_loss = loss(weights - epsilon * u, incoming_carry, frozen_memory)
slope = (positive_loss - negative_loss) / (2 * epsilon)

gradient_norm = abs(slope) * sqrt(active_tensor_elements)
clip = min(1, max_gradient_norm / gradient_norm)
weights = original_weights - learning_rate * clip * slope * u
```

Plain directions use Gaussian values, with the same unit-RMS normalization.
Both spectral variants use eight modes; uniform sampling sets smoothing to zero,
and v1 sets it to four. Perturbation epsilon is `0.001`, and the
estimated active-tensor gradient norm is clipped to one in all ZO trials.
Backpropagation baselines also clip gradients to a global norm of one.

RMS normalization changes the estimator's distribution. Sparse modes and
frequency attenuation also change its covariance: the estimated update is not
generally an unbiased full-parameter gradient. The spectral basis is an
optimization hypothesis, since neighboring tensor entries need not have related
meaning. The algorithm does not force learned weight matrices toward constants.

### State and rollback

Both probes restore the same incoming recurrent carry and see the same episodic
bank. The original active tensor is buffered, so restoring weights uses exact
copies rather than repeated floating-point additions/subtractions.

A third forward evaluation replays the original weights, advances detached carry,
and supplies terminal keys/values for the existing optional memory-write policy.
Memory writing occurs only at this commit point, before the weight update, using
the existing loss threshold and refractory insertion. Failed probes restore
weights and incoming carry without publishing memory or an optimizer step.
Checked inference reports invalid norms, queries and logits as recoverable errors.

One tensor-sized direction and original-weight buffer are retained, plus cosine
basis buffers for spectral variants. These buffers are included in reported
storage. No model-wide gradient or optimizer-moment array is used.

## Controlled tasks

All methods start from the same stable MSSA initialization. The tiny model has
vocabulary 8, width 12, state width 3, key width 4, memory capacity 8, depth/loops
one, and training chunk capacity 16: 1,812 parameters.

- **Cycle:** predict a six-token cyclic sequence. Training samples random phases
  in 32-token documents, split into two 16-token updates with carried state.
  Development uses four phases at length 24; testing uses the other two phases
  at length 48. Training covers all phases, so this tests longer-sequence behavior
  on the learned pattern, not unseen subject matter.
- **Delayed recall:** remember a binary key across repeated distractors, then
  predict it at a query. Training gaps are 2–10, development gaps 8/10, and test
  gaps 11–14. Final-token accuracy measures recall specifically; cross entropy
  includes every next-token prediction.
- **Byte text:** AdamW/full v2 also share a fixed UTF-8 byte tokenizer on disjoint
  line splits of the small built-in science corpus, with vocabulary 257, width
  24 and state width 4. Training/development/test use lines 1–5 / 6–7 / 8–9.
  See `MEASURED_RESULTS.md` for exact target-token budgets and its limitations.

Documents reset carry, with detached carry retained between cycle chunks.
The empty episodic bank remains frozen, and writes and consolidation are disabled
for all methods to isolate weight learning. Training
tokens count each scheduled document once; repeated synthetic documents count
again. Separate forward-token counts include any repeated probe evaluations, and
backward-token counts distinguish backpropagation work. Elapsed time includes
training and periodic development evaluation, excluding model construction and
final test scoring.

## Historical v2 streaming results: three seeds, constant rates

These results predate the confidence-gated readout, five-seed audit, shared
four-rate grid, and common warmup/cosine schedule. The current command produces
the updated results in `MEASURED_RESULTS.md`, not these historical values. In
particular, the earlier `3.89x` training time-to-quality figure is not reproduced
by the latest fairer five-seed protocol and is not a current speed claim.

Measured on an Intel Core i7-7500U CPU at 2.70 GHz, using one normal release
build for all nine method workers. Figures are arithmetic means over three
development-selected seed runs under the streaming protocol above. Memory is
profiled at the common 1,544,704-parameter CLI shape; learning uses the small
1,812-parameter model. These are distinct measurements, not a large-model
training-throughput result.

### Held-out quality

| Method | Cycle test CE ↓ | Recall test CE ↓ | Long-gap recall accuracy ↑ |
| --- | ---: | ---: | ---: |
| AdamW | 3.4143e-7 | 0.304061 | 100.00% |
| SGD | 0.000818 | 0.412794 | 100.00% |
| Plain blockwise ZO | 0.063332 | 0.521300 | 33.33% |
| Uniform spectral ZO | 0.276140 | 0.633644 | 50.00% |
| Interdiffusion v1, smoothing 4 | 0.250117 | 0.560379 | 50.00% |
| Readout-only | 0.199932 | 1.023745 | 83.33% |
| Readout + generic probes | 0.063678 | 0.797555 | 70.83% |
| Recurrent eligibility, input probes | 6.6710e-6 | 0.530019 | 87.50% |
| Full Interdiffusion v2 | 7.0824e-6 | 0.296445 | 100.00% |

Full v2 improves mean recall CE by about 47% relative to v1 in this same
experiment. The input-eligibility ablation exposes its contribution: recurrent
eligibility alone does not preserve full long-gap recall accuracy. Both v2 and
AdamW achieve 100% cycle last-token accuracy, but v2's very small cycle loss
still exceeds AdamW's stricter final loss.

### Time to AdamW development quality

For each task/seed, the target is the selected AdamW trial's **final development
loss multiplied by 1.05**, together with at least its final development
last-token accuracy. We report the first sampled curve point meeting both
conditions. Test data do not define the target or select the learning rate.
Unreached targets remain `null`; all-seed speedup is omitted if any seed misses.

| Method | Cycle target reached | Cycle median paired speedup | Recall target reached | Recall median paired speedup |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 3/3 | 1.00x | 3/3 | 1.00x |
| Interdiffusion v1 | 0/3 | — | 0/3 | — |
| Recurrent eligibility, input probes | 3/3 | 2.51x | 3/3 | 2.41x |
| Full Interdiffusion v2 | 2/3 | — | 3/3 | 3.89x |

For recall, v2's sampled crossings occurred at updates `384`, `256`, and `128`;
AdamW's at `512`, `2048`, and `384`. Paired speedups were `1.61x`, `9.37x`, and
`3.89x`. The median crossing times were `0.0248 s` for v2 and `0.0603 s` for
AdamW; the median paired ratio is not the ratio of these two medians.

These are **development** crossings, with 128-update sampling resolution and
short CPU intervals. They do not verify test quality at the crossing checkpoint.
At the final budget, only **2/3 recall seeds** match AdamW test CE within 5% and
its accuracy; **0/3 cycle seeds** match the strict final test-loss criterion.
Better mean recall CE and matched accuracy therefore do not establish universal
per-seed loss parity.

Timing includes training and periodic development evaluation, but excludes
construction, final test scoring, and learning-rate search. The separate median
total time for both LR candidates on recall was `0.4188 s` for v2 versus
`0.4709 s` for AdamW; on cycle it was `0.8387 s` versus `1.0828 s`. All trials
and per-seed curves are retained. The legacy `CE <= 0.75` crossing is also
recorded but is not used to claim final-quality parity.

### Owned storage and process RSS

Workers first allocate the common CLI shape (vocabulary 2,048, width 256, state
16, key 32, memory capacity 512, chunk capacity 64), run one eight-token update,
and drop it before learning trials. Numeric bytes count all owned numeric Vec
capacities, including gradients, moments, cosine bases, forward tangents,
row counters/statistics, carry snapshots, and any allocated probe scratch.
Linux `VmHWM` is read after the trials, before final report serialization.

| Method | CLI-shape numeric storage | Worker peak RSS |
| --- | ---: | ---: |
| AdamW | 33.83 MiB | 31,588 KiB |
| SGD | 33.83 MiB | 19,944 KiB |
| Plain blockwise ZO | 10.56 MiB | 14,860 KiB |
| Uniform spectral ZO | 10.72 MiB | 15,036 KiB |
| Interdiffusion v1 | 10.72 MiB | 15,120 KiB |
| Readout-only | 12.59 MiB | 17,076 KiB |
| Readout + generic probes | 13.65 MiB | 17,084 KiB |
| Recurrent eligibility, input probes | 16.82 MiB | 21,620 KiB |
| Full Interdiffusion v2 | 16.86 MiB | 21,992 KiB |

Full v2 owns **17,682,928 bytes**, versus AdamW's **35,468,224**: **50.14% less**,
or a `2.01x` reduction. Observed peak RSS is **30.38% lower** (`1.44x` reduction).
The extra input-eligibility state adds only 44,976 bytes over the recurrent-only
variant at this shape, while replacing its empty-bank input probe workspace.

Numeric storage excludes Vec headers, allocator overhead, executable mappings,
and datasets. Zeroed buffers may remain nonresident, so owned capacity and RSS
are different metrics. SGD still allocates unused legacy Adam arrays and is not
an optimized SGD-memory baseline. No CUDA/VRAM or corpus-scale FSS claim follows
from these CPU synthetic results.

## Historical v1 experiment

The initial protocol used reset 16-token cycle documents, development length 16,
test length 32, recall training gaps 2–6, development gaps 3/5, and test gaps
7–10. It predates the streaming/context-validation improvements above. The
following historical values are retained for context; the current command runs
the new protocol, whose v1 row provides the controlled same-protocol comparison.

Measured on an Intel Core i7-7500U CPU at 2.70 GHz with the release build.
Quality figures below are arithmetic means over the three development-selected
seed runs. Times are the median time to the first sampled development loss
at or below `0.75` on the cycle task, with 128-update sampling resolution.

| Method | Cycle test CE ↓ | Recall test CE ↓ | Long-gap recall accuracy ↑ | Cycle time to target ↓ |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 0.010614 | 0.789099 | 100.00% | 0.0292 s |
| SGD | 0.000080 | 0.946474 | 100.00% | 0.0240 s |
| Plain blockwise ZO | 0.070447 | 1.097281 | 79.17% | 0.1132 s |
| Uniform spectral ZO | 0.250395 | 1.315943 | 50.00% | 0.1759 s |
| Interdiffusion, smoothing 4 | 0.232207 | 1.149540 | 50.00% | 0.1400 s |

Short CPU timing samples are indicative rather than accelerator-throughput
measurements. All trials and per-seed curves are retained to expose variation.

### Memory

Each isolated worker first allocates the common CLI-sized shape (vocabulary
2,048, width 256, state 16, key 32, memory capacity 512, chunk capacity 64), runs
one eight-token update, and drops it before the tiny learning trials. Numeric
storage refers to this 1,544,704-parameter shape. Worker peak RSS is Linux
`VmHWM`, observed after the trials and before final report serialization.

| Method | CLI-shape numeric storage | Observed worker peak RSS |
| --- | ---: | ---: |
| AdamW | 33.83 MiB | 31,412 KiB |
| SGD | 33.83 MiB | 19,868 KiB |
| Plain blockwise ZO | 10.56 MiB | 14,588 KiB |
| Uniform spectral ZO | 10.72 MiB | 14,676 KiB |
| Interdiffusion, smoothing 4 | 10.72 MiB | 15,012 KiB |

Interdiffusion uses **3.15x less numeric storage** and has **2.09x lower observed
worker peak RSS** than this AdamW run. Numeric storage counts owned Vec capacity,
including probe and cosine workspace, but excludes headers, allocator overhead,
executable mappings and corpus storage. RSS differs because unused zeroed buffers
can remain nonresident. The SGD baseline still allocates the legacy unused Adam
arrays; its storage result is not an optimized SGD-memory comparison.

### Interpretation

The lightweight runtime delivers a concrete CPU memory reduction. Every ZO
variant learns the cyclic pattern from initialization. However, the smoothed
spectral version has worse test loss than plain ZO on both tasks, achieves chance
long-gap recall, and takes about 4.8x the observed AdamW time to the cycle loss
target. Uniform spectral sampling also fails to improve the overall trade-off.

The first Interdiffusion method is therefore retained as an opt-in experiment.
These results do not establish corpus-scale pretraining performance, general
knowledge improvements, or CUDA/VRAM savings.

## Verification and research foundations

Tests check identical initialization and streaming/chunk loss/carry parity at
depths 1/2/4, finite differences against backpropagated directional derivatives
with nonempty memory and nonzero incoming carry, exact failed-probe rollback,
single-commit memory writing, and cosine-basis orthonormality. V2 additionally
checks local readout gradient/Adam parity, forward A/Delta/B/input/norm derivatives
against TBPTT with nonzero carry and nonzero MLP/adapter projections, preserved
optimizer state after failed probes and late validation, original-weight memory
keys, accounted fallback workspace growth, bounded input-row budgets, and
frequency coverage under sparse token visits. Matched-quality reporting tests
require both loss and accuracy and preserve missing targets.

After upstream integration, the verified suite passed 421 library tests and all
190 integration tests (two library and two manual timing tests ignored).
CUDA-feature release test compilation also passed. Validation used release
optimization with LTO disabled and 16 codegen units; measured timing tables use
the normal release profile. WebGPU checks exercise the upstream trainer, while
Interdiffusion remains CPU-only.

```bash
cargo test --release --lib interdiffusion::
```

Relevant prior work:

- [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333).
- [Parameter-Efficient Fine-Tuning with Discrete Fourier Transform](https://arxiv.org/abs/2405.03003).
- [Laplacian Smoothing Gradient Descent](https://arxiv.org/abs/1806.06317).
- [Gradients without Backpropagation](https://arxiv.org/abs/2202.08587), for
  forward directional derivatives as a learning ingredient.

These establish useful ingredients; their published results do not validate this
particular MSSA optimizer or its from-scratch training behavior.
