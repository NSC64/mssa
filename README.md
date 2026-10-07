# MSSA — memory-augmented state-space architecture

MSSA is a recurrent language-model research project written from scratch in
Rust. It combines a selective state-space recurrence, bounded episodic memory
in hyperbolic space, plastic adapters, and a terminal workflow for training and
evaluation.

MSSA does not use transformer attention or a growing key/value cache. Each
token updates a fixed-size recurrent carry and can read from a bounded memory
bank. This gives inference a constant-size state while still allowing
long-lived episodic information to be stored and retrieved.

> **Current command name:** the repository is branded MSSA, while the Rust
> package and executable retain the historical `pssa` compatibility name for
> existing checkpoints and scripts. The examples below use `cargo run --release`
> so they work without installing the binary.

## Design

For a normalized input embedding `x`, MSSA computes token-conditioned recurrence
parameters:

```text
delta = softplus(W_delta x)
B     = W_B x
C     = W_C x
A     = -softplus(A_raw)
```

The diagonal state update is:

```text
A_bar = exp(delta * A)
B_bar = delta * B
h     = A_bar * h + B_bar * x
y     = sum(C * h)
```

The recurrent carry crosses token and chunk boundaries. A memory query combines
the normalized input and current recurrent output:

```text
q  = W_qx x + W_qh y
qh = PoincareProjection(q)
```

The episodic bank stores projected keys and values. Retrieval uses hyperbolic
distance and a temperature-scaled softmax. A learned gate controls how much of
the memory value reaches the residual stream, followed by the plastic adapter
and SiLU MLP.

### Plasticity

- Novel inputs can write new episodic slots.
- Refractory counters prevent repeated contradictory writes from immediately
  erasing stable memories.
- Fast adapter updates can be consolidated into persistent weights with a
  closed-form ridge step.
- Training and inference retain scalar/reference paths so optimized CPU, CUDA,
  and WebGPU paths can be checked against known behavior.

## Paper implementation

[`paper.pdf`](paper.pdf) is the source paper for the current MSSA work. The
repository contains an opt-in prototype of its certified inference ideas:

- `CertifiedMemoryIndex` clusters hyperbolic memory keys and computes a
  conservative omitted-softmax-mass bound.
- If the requested memory certificate cannot be established, retrieval falls
  back to the exact full-bank reader.
- `CertifiedVocabularyIndex` uses centroid/radius bounds to skip vocabulary
  clusters while preserving exact greedy token selection and tie behavior.
- Runtime indexes are not serialized and are invalidated after memory or weight
  updates.

The certified path is an inference optimization. It does not change the
training rule or checkpoint format. The main implementation is in
[`src/sparse_inference.rs`](src/sparse_inference.rs), with integration in
[`src/pssa.rs`](src/pssa.rs) and generation routing in
[`src/inference.rs`](src/inference.rs).

## BitNet research status

MSSA also contains an opt-in BitNet b1.58 probe in
[`src/bitnet.rs`](src/bitnet.rs). It uses packed ternary output-head weights
and symmetric per-token int8 activations while keeping recurrent carries,
normalization, and memory values in FP32.

The local CPU benchmark showed a `12.8x` output-head memory reduction, but only
`44.53%` argmax agreement and a `0.448x` BitNet/FP32 speed ratio. This
post-training conversion is therefore experimental and is not enabled by
default. A trained-from-scratch or QAT implementation and a hardware-specific
integer kernel are still needed before it can be considered a useful deployment
path.

## Build and run

### Requirements

- Rust 1.85 or newer with Edition 2024 support.
- Cargo.
- Network access only for HTTP or Hugging Face dataset sources.
- Optional CUDA toolkit/driver support for `--features cuda`.
- Optional speech tools when building with `--features speech`.

Build the release binary:

```bash
cargo build --release
```

Run the terminal interface or command help:

```bash
cargo run --release -- tui
cargo run --release -- help
```

## Training

Train on the built-in reference corpus:

```bash
cargo run --release -- train science \
  --tokenizer bpe \
  --max-tokens 200000 \
  --epochs 1 \
  --out runs/model.pssa
```

Train on a local file:

```bash
cargo run --release -- train data/corpus.txt \
  --out runs/model.pssa \
  --epochs 4
```

Resume a long corpus as a sequence of bounded windows:

```bash
cargo run --release -- train data/corpus.txt \
  --skip-tokens 0 --max-tokens 200000 \
  --epochs 1 --out runs/ck01.pssa

cargo run --release -- train data/corpus.txt \
  --skip-tokens 200000 --max-tokens 200000 \
  --resume runs/ck01.pssa --epochs 1 --out runs/ck02.pssa
```

Important training defaults:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--latent` | `256` | Latent width. |
| `--state` | `16` | Recurrent states per latent channel. |
| `--key` | `32` | Hyperbolic memory-key width. |
| `--memory` | `512` | Episodic memory capacity. |
| `--chunk` | `64` | Training chunk length. |
| `--batch-size` | `1` | Independent document lanes. |
| `--accumulate` | `8` | Chunks per optimizer update. |
| `--lr` | `1e-3` | Base learning rate. |
| `--seed` | `42` | Initialization seed. |
| `--tokenizer` | `bpe` | `bpe` or `word`. |
| `--backend` | `auto` | `auto`, `cpu`, `webgpu`, or `cuda`. |

For bounded divergence containment, the runtime-only options
`--grad-clip 1.0` and `--memory-value-cap 512` can be enabled. Repeat these
options on every resumed run; they are not stored in checkpoints.

## Generation and evaluation

Generate from a checkpoint:

```bash
cargo run --release -- generate \
  --model runs/model.pssa \
  --prompt "The scientist observed" \
  --temperature 0 \
  --max-new-tokens 64
```

Run an interactive chat turn:

```bash
cargo run --release -- chat --model runs/model.pssa
```

Score a checkpoint on a held-out slice:

```bash
cargo run --release -- score data/heldout.txt \
  --model runs/model.pssa \
  --skip-tokens 0 \
  --max-tokens 50000
```

The repository also includes a CPU decoder-only transformer baseline for
comparison:

```bash
cargo run --release -- train-transformer data/corpus.txt \
  --tokenizer-from runs/model.pssa \
  --out runs/baseline.trfm
```

See [`docs/COMPARISON.md`](docs/COMPARISON.md) for the token-matched comparison
workflow.

## Datasets

Training accepts local files, directories, URLs, the built-in `science` corpus,
and Hugging Face repositories:

```bash
cargo run --release -- train data/corpus.txt
cargo run --release -- train data/
cargo run --release -- train https://example.org/corpus.txt
cargo run --release -- train hf:owner/dataset
```

Download a dataset locally:

```bash
cargo run --release -- download wikimedia/wikipedia \
  --out data/wikipedia.txt
```

Clean a raw WikiText text file before starting a new chain:

```bash
cargo run --release -- clean-wikitext \
  wiki.train.raw --out data/wikitext-clean.txt
```

Cleaning writes a new file and never overwrites the input. Do not change the
corpus or cleaning policy halfway through a resume chain because token offsets
would no longer refer to the same data.

## Terminal interface

The TUI provides:

- Training setup and live progress monitoring.
- Local checkpoint chat and streaming generation.
- Memory inspection and retrieval telemetry.
- Held-out evaluation and checkpoint history.
- Hardware/device information and backend selection.
- Dataset library, corpus mixing, sweeps, and run timeline views.

Press `Tab` to change views, `Ctrl+K` to open the command palette, and `F1` or
`?` for keyboard help. See:

- [`docs/training-setup.md`](docs/training-setup.md)
- [`docs/TUI-EXTRAS.md`](docs/TUI-EXTRAS.md)
- [`docs/STACKED-DEPTH.md`](docs/STACKED-DEPTH.md)

## Benchmarks

Run the deterministic feature benchmark suite:

```bash
cargo run --release -- benchmark \
  --feature all \
  --out __agent__/feature_results
```

Run one feature:

```bash
cargo run --release -- benchmark \
  --feature bitnet \
  --out __agent__/bitnet_results
```

Benchmark records are JSON files with a generated `summary.md`. GPU timings
must be collected on an actual CUDA/WebGPU device; the local CPU benchmark is
not a substitute for accelerator measurements.

## Development

Run the release tests:

```bash
cargo test --release
```

Check all test targets without running them:

```bash
cargo check --release --tests
```

The scalar CPU implementation is the reference for numerical checks. Optimized
CPU, CUDA, WebGPU, batched, stacked-depth, and shared-loop paths should preserve
the reference behavior within the tolerances covered by the test suite.

## Checkpoints and compatibility

The current model checkpoint extension is `.pssa`; transformer baseline files
use `.trfm`. Checkpoints include model weights, tokenizer metadata, optimizer
state, recurrent state, episodic memory, and schedule information as applicable.
Runtime-only sparse indexes and quantized inference caches are rebuilt after
loading and are not serialized.

Checkpoint repair and compatibility tests live in
[`tests/checkpoint_repair.rs`](tests/checkpoint_repair.rs).

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/pssa.rs` | MSSA model, recurrence, memory integration, plasticity, and training. |
| `src/memory.rs` | Hyperbolic episodic memory bank. |
| `src/sparse_inference.rs` | Certified memory and vocabulary inference indexes. |
| `src/bitnet.rs` | Experimental packed ternary inference primitives. |
| `src/inference.rs` | Generation and sampling APIs. |
| `src/feature_benchmark.rs` | Deterministic feature benchmarks. |
| `src/tui/` | Terminal interface. |
| `paper.pdf` | Source paper for the certified inference work. |
| `tests/` | Numerical, checkpoint, backend, and interface tests. |

## Project status

MSSA is a research prototype. The certified sparse inference path is covered by
exactness and fallback tests. BitNet post-training quantization currently saves
memory but does not meet the quality/speed bar for a default path. Larger-scale
training, paper-specific datasets, and real accelerator measurements remain
future work.

## License

MSSA is distributed under the GPL-3.0-or-later license.
