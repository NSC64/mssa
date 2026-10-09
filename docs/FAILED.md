# Failed and bounded experiments

This file records approaches that did not earn promotion to the default MSSA
path. A failure here means “did not meet the stated quality/performance bar on
the measured protocol”; it does not mean the code or idea has no useful niche.
Raw records stay outside Git under `/tmp/omnirush/` or the paths named by each
study.

## Curved token geometry

The context-conditioned output head mapped the full recurrent state and learned
candidate-token points into a Poincaré ball, then used negative squared
hyperbolic distance as the logit. Dot-product, flat-distance, curved-distance,
and flat/curved-blend heads were matched at rank 8.

Five paired Tiny Shakespeare seeds (`7501`–`7505`) selected rates on development
data only. The backbone was frozen after native dot-product training.

| Head | Held-out CE | Delta vs dot |
| --- | ---: | ---: |
| Dot product | **2.464406** | — |
| Flat distance | 2.480750 | +0.016344 |
| Curved, `c=0.1` | 2.490574 | +0.026167 |
| Curved, `c=1` | 2.504046 | +0.039640 |
| Flat/curved blend | 2.482548 | +0.018142 |

Every non-dot head lost all five paired comparisons. Curved inference was about
3.8–4.4x slower in the NumPy reference implementation. The experiment is
therefore rejected as a frozen-head replacement, not as a proof that joint
end-to-end geometry can never work. Full details are in
[`CURVED_TOKEN_GEOMETRY.md`](CURVED_TOKEN_GEOMETRY.md).

## Interdiffusion v1 / pure spectral zeroth-order updates

The original blockwise two-probe/spectral method saved memory but failed the
quality-time trade-off:

- smoothed spectral Interdiffusion v1 reached cycle CE `0.232207` and recall CE
  `1.149540`, versus AdamW `0.010614` and `0.789099` in the historical test;
- long-gap recall was only `50%`;
- time to the cycle target was about `4.8x` AdamW;
- spectral smoothing was worse than plain blockwise zeroth-order updates on
  both controlled tasks.

The v1 implementation remains useful as a diagnostic baseline and is not the
candidate for the main training path.

## Interdiffusion v2 as a universal AdamW replacement

V2 is materially better than v1 and is worth keeping as the memory-saving
optimizer, but it did not meet the bar for an unconditional replacement:

- it owns about `44.23%` less numeric optimizer storage in the current CLI-shape
  audit;
- it is slower per scheduled training token (`13,043` versus AdamW's `17,925`
  on the five-seed byte-text audit);
- on the four-study Tiny Shakespeare run, Interdiffusion test CE was `2.5372`
  versus AdamW `2.3763`, a `+0.1609` nats regression;
- it is CPU-only and its regular checkpoint/resume path restores readout state
  but rebuilds bounded eligibility scratch and optimizer state;
- its strongest wins are task-dependent: controlled cycle/recall tests and the
  small byte-text diagnostic, not a corpus-scale language-model result.

The useful core is retained: fused readout statistics, streaming eligibility,
local readout gradients, bounded probes, exact rollback, and low-storage
training. An explicit `--optimizer interdiffusion` mode now exports a regular
checkpoint, while AdamW remains the comparison baseline.

### Adaptive input-trace heuristic

An attempted optimization enabled input eligibility only when a chunk had enough
distinct token IDs and otherwise used the recurrent/local path. It was removed
after a fresh byte-text worker: two seeds became severely unstable on the
held-out split despite acceptable development CE (test CE reached approximately
`7.6e2` and `6.0e1`). Token diversity was too weak a proxy for input-trace
usefulness; reducing work per update did not preserve optimization stability.

## Sparse inference on diffuse or held-out banks

Geometric sparse retrieval did not provide a universal speedup:

- diffuse CSR was `0.984x` dense and dual mode was `0.970x`, with `100%`
  fallback;
- held-out CSR was `0.973x` build-inclusive;
- the useful speedups occurred on clustered/replay-like banks, not arbitrary
  banks.

The sparse path remains opt-in and must report fallback and index-build cost.

## Dream replay as a general forgetting solution

Replay slightly improved old-task retention in the controlled study but also
slightly worsened new-task CE. Frozen adapters showed no replay effect. This is
a bounded adapter-replay trade-off, not evidence that dream replay generally
solves catastrophic forgetting.

## What did succeed

The fused Interdiffusion readout-statistics pass preserved selected curves while
reducing measured short-run training time. It is an implementation optimization,
not a new learning-rule result, and remains in the main source path.

## Promotion rule

An optimizer becomes a default only after it has:

1. a checkpoint/resume path with explicit state semantics;
2. matched held-out quality against AdamW;
3. measured throughput and resident-memory results on the intended backend;
4. rollback and non-finite-update tests;
5. a result that does not depend on selecting against the test split.
