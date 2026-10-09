# MSSA

## Memory-Augmented State-Space Architecture

MSSA is an experimental recurrent language-model architecture implemented in
Rust. It combines selective state-space recurrence, bounded episodic memory in
hyperbolic space, plastic adapters, optional certified sparse inference, and a
terminal workflow for training and evaluation.

This repository is a research fork of
[PSSA (Plastic State-Space Architecture)](https://github.com/Sparticle62ops/pssa),
created by [Sparticle62ops](https://github.com/Sparticle62ops). MSSA preserves
the upstream architecture, implementation history, GPL-3.0-or-later license,
and contributor attribution while developing additional research features under
the [NSC64/mssa](https://github.com/NSC64/mssa) project.

## Project status

MSSA is research software. Results in this repository are small-scale,
reproducible engineering measurements rather than claims of production language
quality or general-purpose acceleration. The current research areas are:

- certified sparse memory and vocabulary inference;
- Interdiffusion, an opt-in CPU training experiment;
- Interdiffusion can be selected for ordinary MSSA training with
  `--optimizer interdiffusion`; AdamW remains the explicit baseline;
- context-conditioned curved token geometry, currently an isolated negative
  result on a frozen-backbone study;
- WebGPU and CUDA execution paths inherited from upstream;
- bounded episodic memory, plastic adapters, and dream replay.

Large-corpus quality, sustained accelerator throughput, energy usage, and broad
generalization remain open research questions.

## Naming and compatibility

The project and primary command are branded **MSSA**. The historical `pssa`
package/library name, compatibility executable, Rust module identifiers, and
`.pssa` checkpoint extension remain available so existing code and checkpoints
continue to work. Cargo uses `mssa` as the default executable and also builds
the compatibility executable `pssa`.

Use the following conventions:

| Purpose | MSSA interface |
| --- | --- |
| Repository | `https://github.com/NSC64/mssa` |
| Primary executable | `mssa` |
| Compatibility executable | `pssa` |
| Rust library import | `pssa` (historical crate identity) |
| MSSA checkpoint | `.pssa` (historical wire format) |
| Transformer baseline checkpoint | `.trfm` |

This compatibility layer is deliberate: rebranding user-facing tooling does
not invalidate the upstream format or silently fork the public Rust API.

## Evidence policy

Performance numbers in this repository are claims about a named workload, not
about MSSA in general. A reported comparison must name its baseline, model
shape, backend, hardware, build profile, seeds, exposure, and whether setup or
fallback work is included. Quality comparisons use held-out data and matched
target-token exposure; timing comparisons use elapsed wall-clock measurements.
Coordinate counts, theoretical operation reductions, and memory capacity are
not substitutes for throughput or resident-memory measurements.

The current headline baselines are:

| Feature | Baseline | Measured result | Scope and limitation |
| --- | --- | --- | --- |
| Certified sparse inference | Dense exact reader | Trained-bank GCSR `1.322x`; dual GCSR+CVP `2.411x` | Five paired CPU seeds; held-out CSR is `1.000x` steady state and `0.973x` including index build; diffuse CSR is `0.984x`. |
| Interdiffusion training | Existing scalar/reference CPU AdamW | `44.23%` less owned numeric storage; slower per scheduled token in the current five-seed audit | Synthetic cycle/recall tasks and a nine-line byte-text diagnostic; not a corpus-scale or accelerator result. |
| Interdiffusion readout pass | Previous mathematically equivalent implementation | Fused softmax statistics preserved every selected trial's development/test curve in a paired five-seed rerun | One before/after release run; timing is exploratory until repeated on controlled hardware. |

The full protocols, raw-record schema, uncertainty treatment, and limitations
are in [`docs/MEASURED_RESULTS.md`](docs/MEASURED_RESULTS.md),
[`docs/INTERDIFFUSION.md`](docs/INTERDIFFUSION.md), and
[`docs/COMPARISON.md`](docs/COMPARISON.md).
The current four-study research run is documented in
[`docs/RESEARCH_STUDIES.md`](docs/RESEARCH_STUDIES.md).
The opt-in curved token-head study is documented in
[`docs/CURVED_TOKEN_GEOMETRY.md`](docs/CURVED_TOKEN_GEOMETRY.md).

## Architecture

For a normalized input embedding `x`, MSSA computes token-conditioned recurrence
parameters:

```text
delta = softplus(W_delta x)
B     = W_B x
C     = W_C x
A     = -softplus(A_raw)

A_bar = exp(delta * A)
B_bar = delta * B
h     = A_bar * h + B_bar * x
y     = sum(C * h)
```

The recurrent carry crosses token and chunk boundaries. A memory query combines
the normalized input and recurrent output:

```text
q  = W_qx x + W_qh y
qh = PoincareProjection(q)
```

The episodic bank stores projected keys and values. Retrieval uses hyperbolic
distance and a temperature-scaled softmax. A learned gate controls the memory
contribution to the residual stream before the plastic adapter and SiLU MLP.

The model does not use transformer attention or an unbounded key/value cache.
Its recurrent carry and episodic bank are fixed-size for a given configuration.

## Certified inference

The opt-in certified inference path is implemented in
[`src/sparse_inference.rs`](src/sparse_inference.rs):

- GCSR (geometric confidence-scheduled retrieval) groups memory keys, computes
  conservative interval distance bounds, and certifies an upper bound on
  omitted retrieval softmax mass.
- A failed memory certificate falls back to the exhaustive reader. Fallback
  work is included in benchmark timings and is never counted as sparse success.
- The vocabulary index uses centroid/radius bounds to prune rows while
  preserving exact greedy selection and tie behavior.
- Runtime indexes are not serialized and must be rebuilt after memory or weight
  changes.

The memory certificate bounds omitted retrieval mass. It does not certify
unchanged logits, cross-entropy, sampled-token identity, or accumulated
recurrent error. Epsilon-zero mode therefore uses the exact reader.

The five-seed CPU benchmark at `V=2048`, latent width `256`, and `512` populated
memory slots measured the following with four-slot GCSR groups:

| Workload | CSR paired speed | CSR fallback | Dual GCSR+CVP speed |
| --- | ---: | ---: | ---: |
| Trained bank | `1.322x` | `0.00%` | `2.411x` |
| Held-out bank | `1.000x` | `17.19%` | `1.700x` |
| Diffuse fixture | `0.984x` | `100.00%` | `0.970x` |
| Separated diagnostic fixture | `1.189x` | `0.00%` | `2.262x` |

The trained-bank result supports a replay-like speedup, not a universal
held-out claim. The complete protocol and certificate audit are in
[`docs/MEASURED_RESULTS.md`](docs/MEASURED_RESULTS.md).

Run the benchmark with the normal release profile:

```bash
cargo run --release -- benchmark --feature sparse --out __agent__/sparse_results
```

## Research features

### Interdiffusion

Interdiffusion v2 combines local analytic gradients with streaming forward
eligibility and rotating cosine-coordinate updates. It avoids reverse-time
activation tapes, but still owns explicitly counted gradients, curvature,
optimizer moments, input tangents, and fallback probe buffers. The original
zeroth-order spectral optimizer remains available as an ablation.

This is an opt-in CPU experiment. The fast eligibility path is restricted to
depth-one, empty-bank models; populated or stacked models use spectral-probe
fallback. The measured trade-off is quality and storage, not higher training
throughput. The base pass uses one stabilized target/non-target softmax
partition per token and reuses it for cross entropy, readout error, and
diagonal curvature; this changes no learning rule. See
[`docs/INTERDIFFUSION.md`](docs/INTERDIFFUSION.md).

The optimization was checked against the previous implementation on five
paired seeds (`7401`–`7405`) with the existing release benchmark. Every selected
development/test curve was semantically identical. One before/after run on an
Intel Core i7-7500U measured these median training-only times:

| Task | Previous pass | Fused pass | Fused target tokens/s |
| --- | ---: | ---: | ---: |
| Cycle | `0.8809 s` | `0.8090 s` | `40,506` |
| Delayed recall | `0.2340 s` | `0.2190 s` | `75,006` |
| Byte text | `2.0469 s` | `1.8073 s` | `16,710` |

These timings are a single paired run and are evidence that the change is
worth measuring further, not a universal Interdiffusion speed claim. The
reproduction command and exact protocol are documented with the other results.

```bash
cargo run --release -- benchmark \
  --feature interdiffusion \
  --out __agent__/interdiffusion_results
```

### Dream replay

Dream replay is disabled by default and operates outside checkpoint state. It
can replay memory values, generate short sequences, or perform both operations
through the plastic adapters. GPU training synchronizes weights around this
host-only phase. Invalid generation inputs return errors instead of aborting the
training process.

## Installation and requirements

- Rust 1.88 or newer, Edition 2024.
- Cargo.
- Network access only when using remote dataset sources.
- For CUDA: `--features cuda`, a compatible NVIDIA driver, and cuBLAS runtime
  libraries.
- For speech: `--features speech` and the documented local speech tools.

Clone and build:

```bash
cd mssa
cargo build --release
```

The compatibility command remains available:

```bash
cargo run --bin pssa -- help
```

## Training

Train on the built-in reference corpus:

```bash
mkdir -p runs
cargo run --release -- train science \
  --tokenizer bpe \
  --max-tokens 200000 \
  --epochs 1 \
  --out runs/model.pssa
```

Train on a local corpus and resume a bounded window:

```bash
cargo run --release -- train data/corpus.txt \
  --out runs/ck01.pssa --epochs 1

cargo run --release -- train data/corpus.txt \
  --skip-tokens 200000 --max-tokens 200000 \
  --resume runs/ck01.pssa --epochs 1 --out runs/ck02.pssa
```

The persistent token cache is opt-in and validated against the source and
tokenizer identity:

```bash
cargo run --release -- train data/corpus.txt \
  --token-cache data/corpus.txt.tok \
  --out runs/model.pssa
```

Important defaults are:

| Option | Default | Description |
| --- | ---: | --- |
| `--latent` | `256` | Latent width. |
| `--state` | `16` | Recurrent states per latent channel. |
| `--key` | `32` | Hyperbolic memory-key width. |
| `--memory` | `512` | Episodic memory capacity. |
| `--chunk` | `64` | Training chunk length. |
| `--batch-size` | `1` | Independent document lanes. |
| `--accumulate` | `8` | Chunks per optimizer update. |
| `--backend` | `auto` | `auto`, `cpu`, `webgpu`, or `cuda`. |
| `--tokenizer` | `bpe` | `bpe` or `word`. |

Runtime-only safeguards such as `--grad-clip 1.0` and
`--memory-value-cap 512` must be repeated on resumed runs; they are not stored
in checkpoints.

## Generation and evaluation

```bash
cargo run --release -- generate \
  --model runs/model.pssa \
  --prompt "The scientist observed" \
  --temperature 0 \
  --max-new-tokens 64

cargo run --release -- chat --model runs/model.pssa

cargo run --release -- score data/heldout.txt \
  --model runs/model.pssa \
  --max-tokens 50000
```

The repository includes a CPU decoder-only transformer baseline. It imports
tokenizer metadata from an MSSA checkpoint but uses an independent `.trfm`
checkpoint format:

```bash
cargo run --release -- train-transformer data/corpus.txt \
  --tokenizer-from runs/model.pssa \
  --out runs/baseline.trfm
```

See [`docs/COMPARISON.md`](docs/COMPARISON.md) for token-matched comparisons.

## Datasets and terminal interface

Training accepts local files, directories, HTTPS URLs, the built-in `science`
corpus, and Hugging Face repositories:

```bash
cargo run --release -- train data/corpus.txt
cargo run --release -- train data/
cargo run --release -- train https://example.org/corpus.txt
cargo run --release -- train hf:owner/dataset
```

The terminal interface provides training setup, live monitoring, checkpoint
chat, memory inspection, held-out evaluation, hardware information, dataset
management, sweeps, and run timelines:

```bash
cargo run --release -- tui
cargo run --release -- train data/corpus.txt \
  --out runs/ck001.pssa --backend cpu --no-tui \
  | target/release/mssa tui --chain runs
```

Documentation for the setup wizard and additional TUI panels is available in
[`docs/training-setup.md`](docs/training-setup.md) and
[`docs/TUI-EXTRAS.md`](docs/TUI-EXTRAS.md).

## Development and verification

Use the incremental optimized profile for local iteration and the normal
release profile for published timing:

```bash
cargo check --profile fast --tests --locked
cargo build --profile fast
cargo test --profile fast --locked
cargo run --profile fast -- help

cargo build --release --locked
cargo test --release --locked
```

The scalar CPU implementation is the numerical reference. CPU-optimized,
CUDA, WebGPU, batched, stacked-depth, and shared-loop paths are expected to
remain within the tolerances covered by the test suite.

Run the complete feature benchmark suite with:

```bash
cargo run --release -- benchmark --feature all --out __agent__/feature_results
```

Benchmark records are JSON files accompanied by a generated `summary.md`.
Performance reports should identify the model shape, backend, hardware, and
workload. Accelerator measurements require execution on the corresponding
device.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/pssa.rs` | Core MSSA model, recurrence, memory, plasticity, and training. |
| `src/memory.rs` | Hyperbolic episodic memory bank. |
| `src/sparse_inference.rs` | Certified memory and vocabulary indexes. |
| `src/interdiffusion*.rs` | Interdiffusion runtime and benchmark components. |
| `src/inference.rs` | Generation and sampling APIs. |
| `src/tui/` | Terminal-interface subsystems. |
| `src/feature_benchmark.rs` | Deterministic feature benchmarks. |
| `tests/` | Numerical, checkpoint, backend, allocation, and interface tests. |
| `docs/` | Methods, measured results, comparisons, and operational guides. |

## Attribution and license

MSSA is derived from PSSA. Credit for the original architecture and upstream
implementation belongs to
[Sparticle62ops and upstream contributors](https://github.com/Sparticle62ops/pssa/graphs/contributors).
The MSSA research fork and its additional work are maintained at
[NSC64/mssa](https://github.com/NSC64/mssa).

Community links inherited from the upstream project are retained for
compatibility:

- Discord / PSlabs: <https://discord.gg/9sqfKeqWYF>
- Optional SOL donations: `4XPZ9uAa2BMoth6msoHRxTWL4mUrMfq3LGrxbAGja96h`

MSSA is distributed under the
[GNU General Public License v3.0 or later](LICENSE). Contributions should
preserve upstream attribution, compatibility with `.pssa` checkpoints, and
clear separation between measured results and research hypotheses.
