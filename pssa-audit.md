# MSSA codebase audit

**Scope:** whole repository. 80k lines, 85 files in `src/`, 41 TUI modules, 5 subsystems.
**Method:** `cargo clippy --all-targets` (154 warnings) + full-file reads delegated across 4 subagents; every dead-code claim independently re-verified by grep against `src/ tests/ examples/` before inclusion.
**Date:** 2026-10-08 · baseline HEAD `6670654`; follow-up fixes are tracked below.

## Follow-up status

This document is an engineering audit and prioritization record. Findings are
not treated as permission to remove compatibility APIs, alter checkpoint wire
formats, or claim an optimization without measurement. The current follow-up
implements the high-confidence safety and dead-storage fixes listed below while
retaining historical `pssa` identifiers where they are part of the public API or
serialized format.

| Area | Status | Implemented follow-up |
|---|---|---|
| Remote dataset input | fixed | HTTPS-only, redirect-disabled, 16 MiB bounded response reads; Hugging Face repository validation and query encoding. |
| Checkpoint input | fixed | Bounded streaming file reads and vocabulary-count validation before allocation. |
| Transformer checkpoint arithmetic | fixed | Checked parameter-element and byte-count arithmetic. |
| Dream replay failure mode | fixed | Sampling errors return through the training command instead of aborting the process. |
| Raw confidence shape | fixed | Logit/output lengths are asserted equal before normalization. |
| CUDA packed launch bounds | fixed | Packed backward dimensions pass through the shared `u32` and byte-size checks. |
| Dead runtime storage | fixed | Removed unused CUDA forward memory retention, unused diagnostics export, unused WGSL pipelines, unused tensor wrappers, and unused raw dot-pointer helpers. |
| MSSA branding | fixed | Primary `mssa` binary, formal README, package metadata, attribution, and compatibility `pssa` binary are in place. |

The remaining performance and architecture findings require independent
benchmarks or larger refactors. They remain explicitly listed below rather than
being represented as completed work. In particular, the duplicated stacked CUDA
stages, persistent transformer tape, WebGPU staging/dispatch consolidation, and
event-loop I/O changes should be handled in focused patches with parity checks.

---

## Summary

| Category | Findings | Severity |
|---|---|---|
| Dead code | 15 | 3 ~560 lines of Rust + ~130 lines WGSL |
| Unoptimized | 19 | 6 hot-path (per-token / per-update) |
| Unsafe / untrusted input | 15 | 3 exploitable, 28 undocumented `unsafe` |
| Over-engineering | 15 | ~2900 duplicated lines |
| Blocking IO on event loop | 3 | contradicts a documented design claim |

**Genuinely solid — do not "fix":** child-process reaping (11 `Drop` impls), the checkpoint binary format's overflow/length handling, `token_cache` parsing, `tokenizer`/JSON bounds, the `panic = "abort"` terminal-restore hook, `loss_csv.rs`, `ab::decoded_prefix`, `kaggle::Tail`, all 17 PTX kernels (loaded *and* launched — no mismatch), `#[cfg(feature)]` gating.

---

## 1. Dead code

| # | Tag | What to cut | Replacement | Path |
|---|---|---|---|---|
| 1 | `delete` | `linalg::Vector` + `linalg::Matrix` — entire parallel matrix API, **zero production users**; live only from `tests/linalg.rs`. Duplicates `pssa::ParamMatrix`/`ParamVector` | keep `ParamMatrix`; delete `linalg.rs:30-572` | `src/linalg.rs` |
| 2 | `delete` | 3 `pub unsafe fn` `dot_256_raw`/`dot_128_raw`/`dot_32_raw` — **no callers, no `# Safety` section, no length parameter** (clippy confirms missing-Safety) | nothing | `src/linalg.rs:283,308,333` |
| 3 | `delete` | `src/diagnostics.rs` — whole module, zero refs; `pub` export is what hides it | nothing | `src/diagnostics.rs` |
| 4 | `delete` | 3 WGSL entry points `affine_rmsnorm_main`/`silu_main`/`adamw_main` — pipelines built at init, **never dispatched**. ~130 lines WGSL + 3 pipelines | CPU twins do these stages | `src/backend.rs:150,177,206` |
| 5 | `delete` | `TensorBuffer` + `ParamTensor` (79 lines) — 22 refs, all inside `backend.rs` | nothing | `src/backend.rs:1582-1660` |
| 6 | `delete` | `certified_greedy` engine branch. `certified_greedy: true` appears once, in a unit test. ~40 lines of unreachable generation control flow | `inference.rs` gates it on a constant | `src/inference.rs:338-476` |
| 7 | `delete` | `ForwardBuffers.memory` — written at `stages.rs:796`, **never read**; retains 3 device allocations (`capacity*(d_k+d_v)` floats) for the whole resident stage lifetime | drop field + 3 assignments | `src/cuda/stages.rs:61,685` |
| 8 | `delete` | `execute_into` strided-batched branch — all 5 callers pass `batch=1, stride_b=0, stride_c=0`. Forces `"cublasSgemmStridedBatched"` into `BLAS_SYMBOLS` preflight, so a cuBLAS lacking it is rejected at startup for no reason | drop branch + 3 params + symbol | `src/cuda.rs:544-557`, `:59` |
| 9 | `delete` | 4 event symbols (`cuEventCreate`/`Record`/`Synchronize`/`StreamWaitEvent`) required by preflight, never created → every host/device wait is a blocking `synchronize()` | implement events or drop from gate | `src/cuda.rs:43-46` |
| 10 | `delete` | `bwd_stage_ssm_parallel` GPU arm — sole caller passes `None` | drop param + branch | `src/gpu_batch.rs:2715,2833` |
| 11 | `delete` | `Tokenizer::unigram_table` — 100000-element `Vec` + 100k iterations, **allocated on every checkpoint load** (`from_serialized`) into a field nobody reads | drop field | `src/dataset.rs:64-65,102` |
| 12 | `delete` | `RunState::loss_average` + `current_offset` — `parse_kv`/`split` run on **every ingested log line**, never rendered | drop both + 2 parse sites | `src/tui.rs:170,217` |
| 13 | `delete` | `reinforce_stable` → the only ctor of `UpdateOutcome::Stable`, so `memory.rs:260`'s match arm is unreachable. All 4 payload fields never read | drop fn + variant + arm | `src/defense.rs:25`, `src/memory.rs:260` |
| 14 | `delete` | 5 dead `pub fn` in `src/ui.rs`: `magenta`, `panel_blank`, `rule`, `warn`, and `yellow` (only used by dead `warn`). `pub mod` on a lib crate = dead-code lint can't fire | nothing | `src/ui.rs:61,67,135,164,190` |
| 15 | `delete` | Zero-caller `pub fn`: `checkpoint::load_model_v6`, `CLIHandler::load_model_v2`, `DatasetManager::{load_dataset_or_default, extract_clean_text, contradiction_stream, mqar_stream, spam_attack_stream}`, `Tokenizer::are_synonyms`, `Schedule::new`, `inference::generate_chat_turn`, `ScanExecutor` `Default` + `shutdown` flag (never set `true`), `adapter::forward_into`, `memory::{insert, insert_protected, set_value_cap}` CPU shims | nothing | see per-file grep |

**Verified *not* dead — do not chase:**

- `src/cuda/stage_bounds.rs` `token_m`/`token_s`/`weights`/`keys`/`values`/`w_query`/`w_gate`/`w_proj`. Clippy flags them, but only in a **test-local copy** of the file; production destructures and reads them at `src/cuda/stages.rs:325-330`.
- All 17 `stages.ptx` kernels — every one is `load_function`'d *and* launched; no compiled-but-unlaunched or launched-but-uncompiled mismatch.
- All 147 TUI enum variants across 22 enums — constructed and rendered. `keybindings.rs` even asserts `ALL.count_ones() == 28`.
- No CLI flag is accepted-and-ignored (walked every `allowed` vector in `cli.rs`).
- `src/tui/speech.rs` (`cfg(feature = "speech")`), all `cfg(cuda)` code, `max_storage_buffers_per_shader_stage: 18` (at the limit but justified).

---

## 2. Unoptimized code

| # | Tag | What to cut | Replacement | Path |
|---|---|---|---|---|
| 16 | `reuse` | `score`/`throughput`/`--resume` **load the whole model twice** — up to 1 GiB file read + 1 GiB tape/optimizer alloc per extra load. `LoadedCheckpoint.format` is already there, first result discarded | keep one `load_for_inference` | `src/cli.rs:1470-1474`, `:1614-1618`, `:521`+`:721` |
| 17 | `shrink` | `CLIHandler::finite` walks `data`/`grad`/`m`/`v` of every tensor **every optimizer update** — ~6M `is_finite()` + ~24 MB traffic/step on top of the Adam pass that already touches those arrays. CUDA path checks kernel-side; CPU doesn't | fuse into the Adam loop | `src/cli.rs:1188`, `:606-669` |
| 18 | `shrink` | `logits_for_context` allocates a fresh **27-Vec `TransformerTape` (~2 MB) per generated token**, then drops it (~200 GB churn at `--max-new-tokens 100000`). Also re-runs full O(L²) attention per token | one persistent tape in the engine | `src/transformer_inference.rs:52`, `src/transformer.rs:772` |
| 19 | `native` | WebGPU readback allocates a **new staging buffer + full `Maintain::Wait` per output buffer** — 8× per memory stage, 4× per SSM forward. `WgpuWorkspace.staging` already exists for exactly this | reuse it; 1 encoder per stage | `src/backend.rs:602-631` |
| 20 | `native` | WebGPU stage buffers reallocated every call: 12 (`ssm_forward`), 16 (`memory_dispatch`), 19 (`ssm_backward`) `create_buffer` calls per stage. High-water `WgpuWorkspace` exists but is only wired into GEMM | cache by (label, high-water) | `src/wgpu_stages.rs:495-528` |
| 21 | `native` | `dispatch()` per dispatch: new encoder + submit + **two `pollster::block_on(pop_error_scope())`**. 3+ per stage, +1 per recursion level | 1 encoder per stage, scopes once | `src/wgpu_stages.rs:530-559` |
| 22 | `native` | `stream.synchronize()` after **every** GEMM (6+ per `bwd_stage_memory` chunk). Comment justifies the *last* readback, not each one | defer to end-of-stage | `src/cuda.rs:561-568` |
| 23 | `native` | `softplus` elementwise op = full PCIe round-trip (`clone_htod` + launch + `memcpy_dtoh` + sync) per chunk. The packed path already fuses this into the `ssm_prepare` prologue; the non-packed path just doesn't | fuse, as packed does | `src/gpu_batch.rs:498` |
| 24 | `reuse` | `Vec::remove(0)` in 6 TUI buffers (`metric_series` 600, `loss_series` 600, `raw_lines` 400) = ~19 KB memmove **per ingested log line**. `VecDeque` already used in 7 sibling modules | `VecDeque` | `src/tui.rs:267,426,443,456,469` |
| 25 | `shrink` | `moving_loss_at` allocates a `Vec` per graph point (~8 `f64`s) → up to **1 200 heap allocs/frame** | running `(sum,count)` — allocation-free | `src/tui.rs:2285-2303` |
| 26 | `shrink` | `neuron_network()` rebuilds 80 nodes with O(n²) nearest-neighbour (≈13k `powi`) **every frame**, but topology depends only on `frame.cycle` — changes once per 9 s | `OnceLock` keyed on cycle | `src/tui.rs:1563-1621` |
| 27 | `native` | Full-vocab sort per generated token to take top-24 (`d_v`=50k BPE) | `select_nth_unstable_by` + truncate, O(n) | `src/inference.rs:147-166` |
| 28 | `reuse` | Stable softmax implemented **4×**; the one in `linalg.rs:584` is the only uncalled one | call it | `linalg.rs:584` vs `inference.rs:130,563` + `gpu_batch.rs:1090` |
| 29 | `reuse` | Embedding-gradient RMSNorm adjoint copy-pasted **7×** (same 5-line chain, different variable names) | one helper | `pssa.rs:1602`, `gpu_batch.rs:2429,2559,2655,2872,2904`, `sequence_batch.rs:525` |
| 30 | `reuse` | `rms_norm_slice` exists at `linalg.rs:145`; two live sites open-code it | call it | `gpu_batch.rs:462,1151` |
| 31 | `shrink` | `GpuDispatch::clone()` — `Device::gpu()` returns owned and clones **11 `Arc`s**, then the match discards a variant and rebuilds the identical value. 3× per `SequenceBatch` call | `matches!(..., Some(GpuDispatch::Wgpu(_)))` | `sequence_batch.rs:301,435,491` |
| 32 | `shrink` | `interdiffusion_adaptive::base_pass` computes the same softmax **3× per token** (max + other + exp, then `cross_entropy_f64` recomputes, then the error loop again) | return `(loss, max, other, sum)` | `src/interdiffusion_adaptive.rs:689-711` |
| 33 | `reuse` | `evaluation::documents` re-tokenizes the entire corpus for a small window; `token_cache::tokenize_until_window` already implements the optimization | share it | `src/evaluation.rs:43-54` |
| 34 | `shrink` | `score --max-tokens 100 huge.txt` tokenizes all of `huge.txt` | same | same |

---

## 3. Unsafe / untrusted input

| # | Tag | What to cut | Replacement | Path |
|---|---|---|---|---|
| 35 | **unsafe** | `download_url_raw`: `into_string()` with **no byte cap** (60 s window ⇒ unbounded alloc), and an explicit `http://` branch → cleartext corpus fetch. Redirects are followed, so `https://` can be downgraded. The TUI's own HTTP path caps at 16 MiB — the dataset path doesn't | `take(16 MiB + 1)`, reject non-https | `src/dataset.rs:800`, reachable from `:708` |
| 36 | **unsafe** | `read_vocab`'s `Vec::with_capacity(d_vocab)` runs **before** `ensure_backed_by_file` at both loaders. `d_vocab` isn't individually capped, so a ~110-byte crafted header requests ~800 MB before the "backed by this file" check rejects it. **~10⁷ amplification** | reorder; cap `d_vocab` | `src/checkpoint.rs:1331` and `:1049` |
| 37 | **unsafe** | `read_file_capped` checks `metadata().len()` then calls unbounded `fs::read`. For a FIFO/`/dev/zero` the length is 0 → check passes, read runs to EOF | `open` + bounded `read` | `src/checkpoint.rs:1182` |
| 38 | **unsafe** | `download_huggingface_dataset` interpolates `repo` raw into the query string and never calls the existing `validate_huggingface_dataset_name` — `'name&limit=100000'` injects params | reuse the validator + `encode_query_component` (as `dataset.rs:1066` does) | `src/dataset.rs:1175` |
| 39 | **unsafe** | `dream::sample_token` `assert!(max.is_finite())`. Under `panic = "abort"` this is **SIGABRT** on the first NaN — while every sibling path in the same training loop returns a recoverable `Err` and deliberately *continues* (loss guard `cli.rs:1083`, clip-skip `:1152` tolerating 20 bad gradients). Checkpoint is written only at end ⇒ **whole run lost** | `Result<usize, String>` | `src/dream.rs:52,60`, called `cli.rs:1172` |
| 40 | **unsafe** | **28 `unsafe` blocks with zero `// SAFETY:` comments** — `stages.rs` (17) and `packed.rs` (11). The invariants *are* enforced (`gemm_shape`, `lengths`, `elems` all run at fn top), so this is documentation debt, not unsoundness. `safeguards.rs` (4/4) is the correct in-repo style to copy | copy `safeguards.rs` | `src/cuda/stages.rs`, `src/cuda/packed.rs` |
| 41 | **unsafe** | `packed_gemm` / `packed_weight_grad`: `unsafe` wrappers with **no shape validation of their own** (unlike twin `gemm_device` which calls `gemm_shape?`). Sound only because cudarc's `CudaSlice::slice` panics rather than UB-ing | call `gemm_shape` | `src/cuda/packed.rs:817,850` |
| 42 | **unsafe** | `n as u32` unchecked in packed **backward** (`:551`) while the forward path calls `elems(n, …)?` at `:299` — inconsistent, and `n` is the grid bound for 5 launches | add `elems()` | `src/cuda/packed.rs:551` |
| 43 | **unsafe** | `.expect()` on GEMM **inside the documented CUDA-fallback path** — a cuBLAS failure after `ssm_backward` succeeded panics instead of falling back, and `panic = "abort"` = process death | propagate `Result` | `gpu_batch.rs:339,359,379,2396`; `sequence_batch.rs:498,504,519` |
| 44 | **unsafe** | 8 `.unwrap()`s on `HashMap` keys guarded by a **different statement block** — a state-machine drift turns into a panic, not an `Err` | return `Err` | `src/cuda/safeguards.rs:261,305,510` |
| 45 | **unsafe** | `scan_executor`'s `catch_unwind` is **inert in every shipped build** — `panic = "abort"` means it always returns `Ok`, so the `resume_unwind` branch is dead in release. The comment describing it describes dev-only behavior | note or drop | `src/scan_executor.rs:50-70` |
| 46 | **unsafe** | `set_loops()` after `SequenceBatch::new` passes `check_model` (loops isn't in `shape()`) but hits `replay: None` → `expect("stacked lane workspace")`. Unreachable today only because of caller ordering | add `m.loops()` to `shape()` | `src/sequence_batch.rs:80,269,567` |
| 47 | **unsafe** | `RawConfidence`: `max` over all of `logits`, `sum` over the zipped prefix — a short `out` silently yields a distribution that doesn't sum to 1. Latent (both callers pass `d_v`), but it's the one numeric entry point that doesn't assert its shape | assert equal lengths | `src/inference.rs:563` |
| 48 | **unsafe** | Benchmark aggregators `unwrap()` JSON from disk at 9 sites in a `report()` designed to run against **stale/hand-edited/partially-written** records. `panic = "abort"` ⇒ aborts instead of reporting | `Option`/`Result` | `src/interdiffusion_benchmark.rs:148-311`; `feature_benchmark.rs:882` |
| 49 | **unsafe** | `transformer_checkpoint.rs:134` — the only `*` in the checkpoint code that isn't `checked_mul`. Safe only because `allocation_bytes` in a *different function* already bounded it; in release it'd wrap silently | `checked_mul` | `src/transformer_checkpoint.rs:134` |

**Verified hardened — leave alone:**

- Checkpoint format overflow/length handling: `Reader::floats_into` rejects any length ≠ config-derived `out.len()`; every `count * dim` is `checked_mul`.
- `token_cache::read_cache`: every file-derived length validated against file size *before* allocating, contiguous offsets, FNV checksum, token-id range checks.
- `sequence_batch::new`: validates the whole batch before mutating any carry or tape.
- `inference::try_new`: `d_vocab >= 2` + vocab equality — this is what makes `sample()` safe.
- `ab::decoded_prefix`, `kaggle::Tail` — best parsers in the repo (proper `Utf8Error` handling, size caps, credential redaction after assembly).
- `linalg::dot_slice_avx2_fma`: runtime feature detection cached in `OnceLock`, equal-length assert before dispatch, loop bounds keep every 8-wide load in range.
- `scan_executor`: `unsafe impl Send` is in the right place to discharge the invariant; caller lock serializes submitters.

---

## 4. Over-engineering

| # | Tag | What to cut | Replacement | Path |
|---|---|---|---|---|
| 50 | `reuse` | `benchmark_replay::documents_from_encoded` is a **character-for-character copy** of `token_cache::select_documents` (verified by `diff` — only whitespace + `String` vs `&str` errors). A test already asserts they agree ⇒ the copy is pure liability | one calls the other | `src/tui/benchmark_replay.rs:12-58` |
| 51 | `reuse` | `train_documents` duplicates `train_corpus` (~60 lines of identical loop). The clincher: `benchmark_replay.rs:167` is a test that byte-compares the two trainers' output. When a test must prove two functions identical, merge them | one fn + batch param | `benchmark_replay.rs:63-144` vs `transformer_training.rs:12-225` |
| 52 | `shrink` | **~1800 of `gpu_batch.rs`'s 3653 lines** are `stacked_*` line-by-line copies of the base stages (`&mut PSSALayerV2`→`&mut PSSAContinuousBlockV2`, tape prefixes dropped). Justified by an f32 reduction-order contract, but a macro over the block type recovers most | macro | `gpu_batch.rs:1140-1458,2922-3460` |
| 53 | `shrink` | CPU-vs-GPU dispatch `if let Some(gpu) { gemm_nn_dev_into } else { dense_input_adjoint }` hand-written at **31 sites** — which is exactly how finding #10's dead GPU arm survived | fold selection into the helpers | `src/gpu_batch.rs` |
| 54 | `shrink` | `dense_weight_adjoint` / `dense_weight_adjoint_forward` — identical bodies except `.rev()` | one fn + flag | `gpu_batch.rs:200-264` |
| 55 | `stdlib` | FNV-1a implemented **5×** (2 inlined into adjacent fns in `training.rs`) | `checkpoint::fnv1a64` + `token_cache::hash_update` | `checkpoint.rs:89`, `token_cache.rs:221`, `comparison.rs:105`, `training.rs:77,95` |
| 56 | `stdlib` | `checked_mul`/`checked_add` duplicated with **divergent error types**; `add_mul` missing from the transformer copy entirely | one `Result<usize,String>` + `.map_err(invalid)` | `checkpoint.rs:97-108` vs `transformer.rs:91-98` |
| 57 | `delete` | 914-line `interdiffusion_benchmark.rs` (LR grids, bootstrap CIs, `/proc/self/status` RSS scraping, subprocess spawning) compiled into every binary as `pub(crate) use`. Precedent for the alternative is in the same `lib.rs`: `training_diagnostics`/`token_cache` are private `mod` | `#[cfg(feature)]` or private | `src/interdiffusion*.rs` (5 files, ~4300 lines) |
| 58 | `delete` | `ResourceLimits::batch_size`/`.max_tokens` parsed at `cli.rs:2489`, range-checked against a **different** max table than `common_options`, then **never read** — `append_args` (sole consumer) is called only from the TUI to render a display string. Parsed twice, applied once, applied zero times on the CLI | stop populating on the CLI path | `src/cli.rs:2489`, `src/cli/resource_limits.rs` |
| 59 | `delete` | 11 keys advertised in the help overlay reach **no handler** on most of the 7 tabs the shared `NETWORK` mask spans (`i`,`l`,`c`,`o`,`p`,`d`,`r`,`Up`,`Down`,`PgUp`,`PgDn`,`Home`,`End`) — accepted, routed, silently swallowed. The dedup test verifies the *registry*, never that a routed action reaches a live arm | narrow the masks | `src/tui/keybindings.rs:270-295`, `network.rs:470-497` |
| 60 | `delete` | `GpuDispatch::gemm_nn`/`gemm_tn` route the WebGPU arm to a **CPU call**; only production consumer is a test asserting the empty-input error path | delete or implement | `src/backend.rs:1118-1133` |
| 61 | `shrink` | `tui::Local` — ~90-line wrapper sequencing four `poll()`s, one caller | inline | `src/tui/local.rs:19-126` |
| 62 | `delete` | `.expect("running benchmark has output snapshot")` — invariant enforced across two methods with no assertion | `let Some(..) else { return }` | `src/tui/benchmark.rs:129` |
| 63 | `unwrap_or_default` | `Tokenizer::encode`/`decode` convert a genuine integrity failure (`"BPE emitted <unk> despite byte fallback"`) into an **empty vec / empty string** — silently skipped lines, empty completions | keep `try_*` | `src/dataset.rs:438,466` |
| 64 | `delete` | `tokenizer.rs:10` unreachable `else if` (6 warnings), `examples/stage_profile_tmp.rs` has 11 warnings incl. 6 broken `else if` chains | fix/remove | as named |

**Verified justified — keep:**

- `network::Http` / `hf_backup::HubTransport` — production impl + test-only fixture. These are the injectable boundaries that let `github.rs:470-563` and `hf_backup.rs:763-887` test rate limits, redirects, untrusted-origin refusal, and the streaming-upload commit protocol without a network.
- `charts::Shape` (`ThinLine`, `PointMarker`) — correct use of the ratatui abstraction.
- Panic hook (`session.rs:23-31`) correctly restores the terminal *before* delegating — the right behavior given `Drop` doesn't run under `panic = "abort"`. Verified against all 3 exit paths in subprocesses (`session.rs:91-118`).
- `interdiffusion` 2× cost in `validate_*` then `apply` — legitimate all-or-nothing commit design. Worth a note in `docs/INTERDIFFUSION.md`, not a code change.
- The 5-duplicate SSM recurrence implementations — a real f32 reduction-order contract, but #52 shows most of the 1800 lines are recoverable.

---

## 5. Blocking I/O on the event loop

The README's claim "blocking reads were deliberately moved off the event loop" is **substantially true and well executed** — ~20 subsystems do it right with `try_recv` polling and bounded caps (`ComparisonLog`, `hardware::snapshot`, `runs`, `library`, `inspector`, `timeline`, `mixer`, `eval`, `github`, `update`, `hf::Login::verify`, `log_stream`, `notify`, `hf_backup`, `kaggle`). Three holdouts remain, all on `run_app`'s main loop:

| # | Finding | Path |
|---|---|---|
| 65 | `RunState::refresh_chain()` — synchronous `read_dir` + `read_to_string` per checkpoint, every 5 s, unbounded in file count. Every sibling scanner (`runs.rs:231`, `timeline.rs:451`, `library.rs:454`) is threaded | `src/tui.rs:971-974`, `:596-611` |
| 66 | `benchmark::poll` — blocking file read + JSON parse on the render thread on completion | `src/tui/benchmark.rs:132-140` |
| 67 | `runs::poll` → `save_score` — synchronous write + rename on the render thread | `src/tui/runs.rs:295` |

Also verified **correct**, no action: every `Command::spawn` is reaped (`Drop` impls on `process::Job`, `preview::Process`, `eval::Job`, `library::Records`, `kaggle::Worker`, `ab::Worker`, `chat::Job`, `hf_backup::Backup`, `notify::Notify`); the deliberately-unreaped `setup::TrainingRun` is `process_group`'d and documented (`q detaches`); all 9 `VecDeque` caps correct, no off-by-one, no capacity-0 case.

---

## Priority if you only fix a few

1. **#39** `dream::sample_token` — the only finding where a recoverable NaN kills a whole training run.
2. **#16 / #17 / #18** — 2× model load, per-update full-param scan, 2 MB/token alloc. All on the hot path, all trivial diffs.
3. **#35 / #36 / #37** — unbounded download, 800 MB-from-110-bytes, FIFO bypass.
4. **#1 / #2 / #3 / #5** — ~500 lines of provably dead code including 3 undocumented `pub unsafe fn`s.
5. **#40** — add `// SAFETY:` to 28 blocks; copy the pattern already in `safeguards.rs`.
6. **#50 / #51** — 105 duplicated lines whose own tests prove the copies are unnecessary.

---

## Clippy triage

154 warnings. 137 in the lib. Worth noting by class:

- **`too_many_arguments` (23 sites)** — `wgpu_stages.rs` has functions at 25 and 28 args. These are GPU kernel-launch shims; a small `StageArgs` struct would be the honest fix.
- **`needless_range_loop` (~40 sites)** — nearly all are the same pattern: `for i in 0..n { xs[i] ... }` where `i` indexes a *second* slice too. Mostly legitimate in numeric kernels (zip would obscure the index math), but a few in `tui/limits.rs:138`, `tui.rs:3573`, `tui.rs:4545` (`for tab in 0..TABS { TABS[tab] }`) are just indexing a constant array and should be `for tab in TABS`.
- **`needless_return` / collapsible `if` (17 sites)** — pure style, `cargo clippy --fix` handles them.
- **`is_multiple_of` (6 sites)** — `transformer.rs:57,422`, `tui.rs:2245`, `ui.rs:207`, `tui/library.rs:379`, `interdiffusion_benchmark.rs:275`. Stdlib now ships it; repo is on Rust 1.88+.
- **`manual_div_ceil` (1 site)** — `tui/mixer.rs:94`.
- **`clone` on `Copy` type (2 sites)** — `gpu_batch.rs:611,2714` clone a `ScanExecutor`; fallout from #4 in §4 (scheduler object holding a config int).
- **Real ones buried in the noise:** `src/dataset.rs:50` large enum variant spread (1480 bytes), `src/tui/runs.rs:172` (2136 bytes) — both `Box` the big variant.
