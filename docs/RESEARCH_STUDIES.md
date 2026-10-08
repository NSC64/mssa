# MSSA research studies

These are opt-in CPU studies. They are designed to answer bounded engineering
questions, not to establish corpus-scale language-model superiority.

## Reproduce

Build the isolated study worker:

```bash
cargo build --release --locked --example research_studies
```

Download a fixed corpus outside the repository, then run the complete suite:

```bash
curl -L --fail --output /tmp/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
taskset -c 0 python3 scripts/research_studies.py \
  --corpus /tmp/tinyshakespeare.txt \
  --out __agent__/research_studies
```

The script makes contiguous 80/10/10 byte splits, records the source SHA-256,
and refuses to overwrite an existing study output. JSON records contain the
full per-seed/per-method measurements. The published run used:

- Corpus SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Corpus bytes: `1,115,394`; split sizes: `892,315 / 111,539 / 111,540`.
- Seeds: `7401`–`7405`.
- Language updates: `1,024`; byte vocabulary 257; latent width 32; state width 4.
- CPU timing and sparse workers were pinned to CPU 0 for the published run.

## 1. Real-data Interdiffusion

The language study compares AdamW, full Interdiffusion, recurrent-only
Interdiffusion, and readout-only learning. AdamW and full Interdiffusion use a
development-only rate grid of `0.001`, `0.003`, and `0.01`; ablations use
`0.003`. Every candidate receives the same training stream and token budget.
The selected rate is the lowest final development CE; test data are not used
for selection.

The selected five-seed results were:

| Method | Mean test CE ↓ | Mean test perplexity ↓ | Mean next-token accuracy ↑ | Mean training seconds | Mean target tokens/s | Owned numeric bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | `2.3763` | `10.77` | `31.95%` | `1.933` | `16,954` | `762,120` |
| Interdiffusion | `2.5372` | `12.65` | `28.44%` | `1.673` | `19,583` | `342,716` |

Interdiffusion used **55.0% less owned numeric storage** and was faster in this
small run, but its mean held-out CE was worse by `0.1609` nats. The result is a
memory/quality trade-off, not a quality win. The four-method fixed-rate
diagnostic also showed recurrent-only CE `2.7291` and readout-only CE `4.8715`.
The corpus is small and the 1,024-update budget is not a pretraining result.

## 2. Repeated fused-readout timing

The timing worker warms the current Interdiffusion implementation for 32 steps,
then measures 2,048 reset-separated updates in a fresh process. Five runs pinned
to one CPU produced:

| Seed | Seconds | Target tokens/s |
| ---: | ---: | ---: |
| 7401 | `8.016` | `8,176` |
| 7402 | `8.724` | `7,512` |
| 7403 | `8.976` | `7,301` |
| 7404 | `8.634` | `7,591` |
| 7405 | `7.752` | `8,454` |

Median: `8.634 s`, `7,591 target tokens/s`. The earlier paired before/after
check remains in `MEASURED_RESULTS.md`; it found identical selected curves and
lower time after softmax-statistics fusion. These repeated runs establish the
current timing distribution, not a universal speedup over the old binary.

## 3. Sparse-inference crossover

The crossover worker varies:

- memory capacity `64` and `512`;
- vocabulary `257` and `2048`;
- trained, diffuse, and separated banks;
- replay-like and shifted queries;
- GCSR group size `4` and `16`;
- certificate epsilon `0`, `0.01`, and `0.05`;
- timed lengths `32` and `256`;
- dense, CSR, CVP, and dual modes.

It uses a five-seed CPU sweep with latent width 64, state width 4, and key width
8. Greedy IDs, certificate mass, CE deltas, logit deltas, fallback rates, index
bytes, and build-inclusive speed are recorded.

Representative median results at capacity 512, vocabulary 2048, group 4,
epsilon `0.01`, length 256 were:

| Bank/query | Mode | Steady-state speed | Build-inclusive speed | Memory fallback | Greedy agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| Trained / replay | CSR | `4.989x` | `4.666x` | `0%` | `100%` |
| Trained / replay | Dual | `7.590x` | `6.551x` | `0%` | `100%` |
| Diffuse / shifted | CSR | `0.990x` | `0.973x` | `100%` | `100%` |
| Separated / shifted | CSR | `1.456x` | `1.451x` | `0%` | `100%` |

These are not directly comparable to the existing latent-256 benchmark because
the sweep intentionally changes the shape. The result still supports the same
boundary: clustered trained banks can benefit, diffuse banks do not, and the
index cost matters for short sequences. All representative CE deltas and
certificate checks passed; the full crossover matrix is in the raw JSON.

## 4. Controlled dream replay

The dream worker trains task A, inserts detached A representations, trains a
contradictory task B, and measures A retention and B learning. It compares no
replay, replay, replay with consolidation, and consolidation-only controls, with
both trainable and frozen adapters. Five seeds use 64 updates per task and four
48-token evaluation documents per task.

For trainable adapters, mean old-task CE increase after task B was:

| Policy | Mean old-task CE increase | Mean new-task CE |
| --- | ---: | ---: |
| None | `14.1531` | `0.1602` |
| Replay | `13.6074` | `0.1633` |
| Replay + consolidation | `13.6074` | `0.1633` |
| Consolidation only | `14.1531` | `0.1601` |

Replay improved the old-task CE increase slightly in this controlled task but
also slightly worsened new-task CE. With frozen adapters, replay had no measured
effect, as expected. Every condition preserved the detached memory-bank digest;
the study therefore measures adapter replay rather than accidental memory-bank
mutation. This is evidence for a small trade-off, not a claim that dream replay
prevents forgetting generally.

## Verification

The study worker has a unit test for full-target scoring and carry restoration.
The existing full suite remains the authority for checkpoint, numerical,
backend, sparse-certificate, and dream API behavior. Study outputs are kept out
of Git because they are large raw records and depend on the host CPU.
