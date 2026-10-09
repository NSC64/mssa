# Curved token geometry study

This is an opt-in study of a context-conditioned geometric output head. It does
not change the default MSSA/PSSA readout or checkpoint format.

## Model

For a recurrent context feature `h`, the head projects to `u = W h` and gives
each vocabulary item a learned point `v_w`. The candidate score is

```text
p(w | h) = softmax_w(b_w - d_M(u, v_w)^2 / (4 T))
```

The current prototype uses `T = 1`. For curvature `c > 0`, raw vectors are
mapped into the Poincaré ball with the smooth chart

```text
q = raw / sqrt(1 + c ||raw||²)
```

and uses the geodesic distance

```text
d_c(q, v) = 2 / sqrt(c) * asinh(
    sqrt(c) ||q - v|| / sqrt((1 - c||q||²)(1 - c||v||²))
)
```

The chart is deliberately simple and differentiable; it is not an exponential
map. Distances reject points at or numerically beyond the ball boundary. The
implementation contains analytical gradients for the projection, distance, and
cross-entropy, with central-difference tests.

The comparison includes:

- `dot`: `2 u·v + b`;
- `flat`: `-||u-v||² + b`;
- `curved`: negative hyperbolic squared distance for `c = 0.1` or `1`;
- `blend`: a learned scalar blend of flat and `c = 1` curved energy.

The flat-distance head is an important control: with free candidate biases,
squared Euclidean distance differs from a dot-product classifier mainly by
candidate norm terms and optimization parameterization. It is not independent
evidence that any distance formulation is useful.

## Reproduce

```bash
cargo build --release --locked --example curved_features
python3 -m unittest discover -s scripts -p 'test_curved_head.py'
curl -L --fail --output /tmp/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
taskset -c 0 python3 scripts/curved_head.py \
  --corpus /tmp/tinyshakespeare.txt \
  --out /tmp/curved-head-study \
  --exporter target/release/examples/curved_features \
  --seeds 7501 7502 7503 7504 7505 \
  --updates 1024 --rank 8
```

The exporter trains a small byte-level MSSA backbone for 1,024 updates, then
writes frozen context features. The Python study trains only the output heads.
Existing output directories are refused. JSON records include source hashes,
feature hashes, batch-index hashes, environment information, every trial, and
the selected held-out test result.

Study settings:

- corpus: Tiny Shakespeare, 1,115,394 bytes;
- corpus SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`;
- contiguous byte splits: 80/10/10;
- byte vocabulary: 257;
- frozen backbone feature width: 32;
- head rank: 8;
- head batch size: 64 and 1,024 updates;
- development-only rate grid: `0.003`, `0.01`;
- five paired seeds: `7501`–`7505`;
- one pinned Intel i7-7500U CPU core for the reported run.

## Five-seed result

The retained raw run was `/tmp/omnirush/curved-head-study-5b/`. Test CE is
lower-is-better. The interval is a paired bootstrap 95% interval for the
per-seed CE difference from the dot-product head.

| Head | Curvature | Parameters | Test CE | Accuracy | CE delta vs dot | Paired 95% interval | Inference seconds / 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| dot | 0 | 2,569 | 2.464406 | 32.72% | +0.000000 | [0.000000, 0.000000] | 0.006046 |
| flat | 0 | 2,569 | 2.480750 | 32.76% | +0.016344 | [+0.011798, +0.021093] | 0.007989 |
| curved | 0.1 | 2,569 | 2.490574 | 32.87% | +0.026167 | [+0.019742, +0.032592] | 0.026714 |
| curved | 1 | 2,569 | 2.504046 | 33.08% | +0.039640 | [+0.028847, +0.047681] | 0.022779 |
| blend | 1 | 2,570 | 2.482548 | 32.82% | +0.018142 | [+0.013740, +0.022722] | 0.024598 |

All four non-dot heads were worse in all five paired seeds. Curved heads were
also approximately 3.8–4.4x slower than the dot head in this NumPy reference
implementation. Accuracy moved independently of CE, so CE is the primary
selection metric here.

## Interpretation and limits

This run does **not** establish that curved geometry cannot improve a language
model. It establishes a narrower result: under a frozen MSSA representation
whose backbone was trained with the native dot-product readout, replacing that
readout with this fixed-temperature Poincaré-distance family increased held-out
CE on this small five-seed byte-text study.

The prototype also does not yet tune curvature or temperature, does not train
the recurrent representation jointly with the curved head, and uses a small
feature slice for the retained cache. The next meaningful test is an opt-in
end-to-end training path where the context representation can adapt to the
chosen geometry, followed by the same held-out protocol. Until then, the
default output head remains unchanged.
