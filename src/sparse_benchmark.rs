//! Measured certificate cost on frozen, paired CPU models. Bank fixtures are
//! identified explicitly; coordinate counts never stand in for elapsed time.

use crate::{
    dataset::DatasetManager,
    interdiffusion::{BENCHMARK_SEEDS, cross_entropy_f64},
    linalg::SimpleRng,
    memory::HyperbolicEpisodicBankV2,
    pssa::{PSSAConfigV2, PSSALayerV2},
    sparse_inference::{CertifiedMemoryIndex, CertifiedVocabularyIndex},
};
use serde_json::json;
use std::{fs, hint::black_box, path::Path, time::Instant};

const MODES: [&str; 4] = ["dense", "csr", "cvp", "dual"];
const WORKLOADS: [&str; 4] = [
    "trained_bank",
    "heldout_bank",
    "diffuse_fixture",
    "separated_fixture",
];
const REPEATS: usize = 5;
const TOKENS: usize = 256;
const EPSILON: f32 = 0.01;

fn greedy(logits: &[f32]) -> usize {
    (1..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}

fn install(model: &mut PSSALayerV2, mode: &str) -> Result<f64, String> {
    model.disable_certified_inference();
    let start = Instant::now();
    if mode == "csr" || mode == "dual" {
        model.block.certified_memory_index = Some(CertifiedMemoryIndex::build(&model.memory, 4)?);
        model.block.certified_memory_epsilon = Some(EPSILON);
    }
    if mode == "cvp" || mode == "dual" {
        model.certified_vocabulary_index =
            Some(CertifiedVocabularyIndex::build(&model.unembed_w, 32)?);
    }
    Ok(start.elapsed().as_secs_f64())
}

fn batch(
    model: &mut PSSALayerV2,
    mode: &str,
    tokens: &[usize],
    logits: &mut [f32],
) -> Result<usize, String> {
    let mut checksum = 0usize;
    for (i, &input) in tokens.iter().enumerate() {
        if i % 32 == 0 {
            model.reset_recurrent_state();
        }
        let id = if mode == "cvp" || mode == "dual" {
            model.forward_inference_certified_greedy(input)?.0
        } else {
            model.forward_inference(input, logits);
            greedy(logits)
        };
        checksum = checksum.wrapping_add(id);
    }
    Ok(black_box(checksum))
}

fn fixture(seed: u64, workload: &str) -> Result<PSSALayerV2, String> {
    let cfg = PSSAConfigV2 {
        d_vocab: 2048,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut model = PSSALayerV2::new(cfg, seed);
    let training_text = if workload == "heldout_bank" {
        DatasetManager::SCIENCE_REFERENCE_CORPUS
            .lines()
            .take(5)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        DatasetManager::SCIENCE_REFERENCE_CORPUS.to_owned()
    };
    let bytes: Vec<_> = training_text.bytes().map(|b| b as usize + 1).collect();
    if workload == "trained_bank" || workload == "heldout_bank" {
        // The same byte IDs, fixed schedule and exposure for all frozen modes.
        for update in 0..32 {
            let start = update * 16 % (bytes.len() - 17);
            model.reset_recurrent_state();
            model.forward_train_chunk(&bytes[start..start + 16], &bytes[start + 1..start + 17]);
            model.zero_gradients();
            model.backward_chunk(16, 1.0);
            if !matches!(
                model.apply_adamw_with_grad_clip(0.003, 1.0),
                crate::pssa::GradientClipOutcome::Applied { .. }
            ) {
                return Err("non-finite sparse benchmark training update".into());
            }
        }
        // Freeze 512 naturally generated keys/values; no index exists yet.
        let mut logits = vec![0.0; 2048];
        for i in 0..512 {
            if i % 32 == 0 {
                model.reset_recurrent_state();
            }
            model.forward_inference(bytes[i % bytes.len()], &mut logits);
            let key = model.inf_q_pnc.clone();
            let value = model.inf_z_final.clone();
            model.memory.insert(&key, &value);
        }
    } else {
        let mut rng = SimpleRng::new(seed ^ 0x4241_4e4b);
        if workload == "separated_fixture" {
            // An engineered best case, not trained-model evidence: query zero,
            // one near cluster, distant remaining clusters, separated head rows.
            model.block.w_qx.data.fill(0.0);
            model.block.w_qh.data.fill(0.0);
            model.block.cfg.tau_mem = 0.05;
            for row in 0..2048 {
                for col in 0..256 {
                    model.unembed_w.data[row * 256 + col] = (row / 64) as f32 * 0.03;
                }
            }
        }
        for i in 0..512 {
            let mut key = vec![0.0; 32];
            if workload == "separated_fixture" {
                key[0] = if i < 16 { 0.0 } else { 0.85 };
            } else {
                let raw: Vec<_> = (0..32).map(|_| rng.gen_range_f32(-0.3, 0.3)).collect();
                HyperbolicEpisodicBankV2::diffeomorphic_project(&raw, &mut key);
            }
            let value: Vec<_> = (0..256).map(|_| rng.gen_range_f32(-0.1, 0.1)).collect();
            model.memory.insert(&key, &value);
        }
    }
    model.reset_recurrent_state();
    Ok(model)
}

pub(super) fn run(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for workload in WORKLOADS {
        let evaluation_text = if workload == "heldout_bank" {
            DatasetManager::SCIENCE_REFERENCE_CORPUS
                .lines()
                .skip(5)
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            DatasetManager::SCIENCE_REFERENCE_CORPUS.to_owned()
        };
        let stream: Vec<_> = evaluation_text
            .bytes()
            .cycle()
            .take(TOKENS + 1)
            .map(|b| b as usize + 1)
            .collect();
        for seed in BENCHMARK_SEEDS {
            println!("Certified inference: {workload}, paired seed {seed}");
            let mut model = fixture(seed, workload)?;
            let mut logits = vec![0.0; 2048];
            let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut build_seconds = [0.0; 4];
            // Rotate mode order within each seed to reduce thermal/order bias.
            for repeat in 0..REPEATS {
                for offset in 0..4 {
                    let mode_id = (repeat + offset + seed as usize) % 4;
                    let mode = MODES[mode_id];
                    let built = install(&mut model, mode)?;
                    if repeat == 0 {
                        build_seconds[mode_id] = built;
                    }
                    batch(&mut model, mode, &stream[..32], &mut logits)?;
                    let start = Instant::now();
                    batch(&mut model, mode, &stream[..TOKENS], &mut logits)?;
                    samples[mode_id].push(start.elapsed().as_secs_f64());
                }
            }
            let mut dense_logits = Vec::with_capacity(TOKENS * 2048);
            let mut dense_ids = Vec::with_capacity(TOKENS);
            install(&mut model, "dense")?;
            let mut dense_loss = 0.0;
            for (i, &input) in stream[..TOKENS].iter().enumerate() {
                if i % 32 == 0 {
                    model.reset_recurrent_state();
                }
                model.forward_inference(input, &mut logits);
                dense_loss += cross_entropy_f64(&logits, stream[i + 1]);
                dense_ids.push(greedy(&logits));
                dense_logits.extend_from_slice(&logits);
            }
            dense_loss /= TOKENS as f64;
            let dense_seconds = median(&samples[0]);
            for (mode_id, mode) in MODES.iter().enumerate() {
                install(&mut model, mode)?;
                let mut mem_cert = 0;
                let mut mem_fallback = 0;
                let mut vocab_cert = 0;
                let mut vocab_fallback = 0;
                let (mut slots, mut centers, mut mixes, mut exact_rows, mut centroids) =
                    (0, 0, 0, 0, 0);
                let (mut bounds, mut bypasses, mut budget_fallbacks) = (0, 0, 0);
                let mut vocab_bypasses = 0;
                let mut max_omitted_mass = 0.0f64;
                let mut max_mass_bound = 0.0f32;
                let mut reference_val = vec![0.0; model.cfg.d_latent];
                let mut reference_weights = vec![0.0; model.cfg.mem_capacity];
                let mut queries = Vec::with_capacity(TOKENS * model.cfg.d_mem_key);
                // Match the timed scheduler's warmup, including its history.
                batch(&mut model, mode, &stream[..32], &mut logits)?;
                let (mut loss, mut agree, mut max_logit_error) = (0.0, 0, 0.0f64);
                for (i, &input) in stream[..TOKENS].iter().enumerate() {
                    if i % 32 == 0 {
                        model.reset_recurrent_state();
                    }
                    model.forward_inference(input, &mut logits);
                    queries.extend_from_slice(&model.block.inf_q_pnc);
                    let m = &model.block.last_sparse_read_stats;
                    slots += m.scanned_slots;
                    centers += m.center_distance_evaluations;
                    bounds += m.cluster_bound_evaluations;
                    bypasses += usize::from(m.routing_bypassed);
                    budget_fallbacks += usize::from(m.budget_fallback);
                    mixes += m.value_rows_mixed;
                    mem_cert += usize::from(m.certified);
                    mem_fallback += usize::from(m.exact_fallback);
                    if m.certified {
                        model.block.memory.retrieve_soft_into(
                            &model.block.inf_q_pnc,
                            model.block.cfg.tau_mem,
                            &mut reference_val,
                            &mut reference_weights,
                        );
                        let omitted: f64 = reference_weights
                            .iter()
                            .zip(&model.block.inf_mem_weights)
                            .filter(|(_, sparse)| **sparse == 0.0)
                            .map(|(&full, _)| full as f64)
                            .sum();
                        if omitted > m.omitted_mass_bound as f64 + 1e-7 {
                            return Err(format!(
                                "CSR omitted mass {omitted} exceeds certificate {}",
                                m.omitted_mass_bound
                            ));
                        }
                        max_omitted_mass = max_omitted_mass.max(omitted);
                        max_mass_bound = max_mass_bound.max(m.omitted_mass_bound);
                    }
                    let id = if *mode == "cvp" || *mode == "dual" {
                        let (id, stats) = model.certified_greedy_current()?;
                        vocab_cert += usize::from(stats.certified());
                        vocab_fallback += usize::from(stats.exact_fallback());
                        vocab_bypasses += usize::from(stats.routing_bypassed);
                        exact_rows += stats.exact_rows;
                        centroids += stats.centroid_rows;
                        // CVP fidelity is checked against the identical CSR/dense features.
                        if id != greedy(&logits) {
                            return Err("CVP disagrees with its exhaustive head".into());
                        }
                        id
                    } else {
                        exact_rows += 2047;
                        greedy(&logits)
                    };
                    agree += usize::from(id == dense_ids[i]);
                    loss += cross_entropy_f64(&logits, stream[i + 1]);
                    for (&a, &b) in logits.iter().zip(&dense_logits[i * 2048..(i + 1) * 2048]) {
                        max_logit_error = max_logit_error.max((a as f64 - b as f64).abs());
                    }
                }
                let index_bytes = model
                    .block
                    .certified_memory_index
                    .as_ref()
                    .map_or(0, CertifiedMemoryIndex::numeric_storage_bytes)
                    + model
                        .certified_vocabulary_index
                        .as_ref()
                        .map_or(0, CertifiedVocabularyIndex::numeric_storage_bytes);
                let seconds = median(&samples[mode_id]);
                // Cold routing state, varied actual recurrent queries; no
                // repeated-query cache can inflate the retrieval measurement.
                install(&mut model, mode)?;
                let memory_start = Instant::now();
                for query in queries.chunks_exact(model.cfg.d_mem_key) {
                    if let Some(index) = model.block.certified_memory_index.as_mut() {
                        index.retrieve_soft_into(
                            &model.block.memory,
                            query,
                            model.block.cfg.tau_mem,
                            EPSILON,
                            &mut model.block.inf_m_val,
                            &mut model.block.inf_mem_weights,
                        )?;
                    } else {
                        model.block.memory.retrieve_soft_into(
                            query,
                            model.block.cfg.tau_mem,
                            &mut model.block.inf_m_val,
                            &mut model.block.inf_mem_weights,
                        );
                    }
                    black_box(&model.block.inf_m_val);
                }
                let retrieval_seconds = memory_start.elapsed().as_secs_f64();
                let work = 495_616.0
                    + (slots + centers + bounds) as f64 / TOKENS as f64 * 32.0
                    + mixes as f64 / TOKENS as f64 * 256.0
                    + (exact_rows + centroids) as f64 / TOKENS as f64 * 256.0;
                rows.push(json!({"workload": workload, "seed": seed, "mode": mode,
                    "timing_repeats": REPEATS, "tokens_per_repeat": TOKENS, "wall_seconds_samples": samples[mode_id],
                    "median_wall_seconds": seconds, "tokens_per_second": TOKENS as f64 / seconds,
                    "paired_wall_clock_speedup": dense_seconds / seconds,
                    "index_build_seconds": build_seconds[mode_id], "additional_index_numeric_bytes": index_bytes,
                    "single_repeat_speedup_including_build": dense_seconds / (seconds + build_seconds[mode_id]),
                    "memory_certified_reads": mem_cert, "memory_exact_fallback_reads": mem_fallback,
                    "memory_routing_bypassed_reads": bypasses, "memory_budget_fallback_reads": budget_fallbacks,
                    "mean_memory_cluster_bound_evaluations": bounds as f64 / TOKENS as f64,
                    "max_actual_omitted_mass": max_omitted_mass, "max_omitted_mass_bound": max_mass_bound,
                    "memory_fallback_rate": if *mode == "csr" || *mode == "dual" {Some(mem_fallback as f64 / TOKENS as f64)} else {None},
                    "vocabulary_certified_tokens": vocab_cert, "vocabulary_full_scan_tokens": vocab_fallback,
                    "vocabulary_routing_bypassed_reads": vocab_bypasses,
                    "vocabulary_fallback_rate": if *mode == "cvp" || *mode == "dual" {Some(vocab_fallback as f64 / TOKENS as f64)} else {None},
                    "mean_scanned_slots": slots as f64 / TOKENS as f64, "mean_exact_vocabulary_rows": exact_rows as f64 / TOKENS as f64,
                    "mean_value_rows_mixed": mixes as f64 / TOKENS as f64,
                    "retrieval_seconds_varied_queries": retrieval_seconds,
                    "retrieval_queries_per_second_varied_queries": TOKENS as f64 / retrieval_seconds,
                    "full_head_teacher_forced_cross_entropy": loss / TOKENS as f64,
                    "dense_cross_entropy": dense_loss, "cross_entropy_delta": loss / TOKENS as f64 - dense_loss,
                    "greedy_agreement": agree as f64 / TOKENS as f64, "max_abs_logit_error": max_logit_error,
                    "coordinate_work_proxy_speedup": 1_167_104.0 / work}));
            }
        }
    }
    let mut aggregate = Vec::new();
    for workload in WORKLOADS {
        for mode in MODES {
            let group: Vec<_> = rows
                .iter()
                .filter(|r| r["workload"] == workload && r["mode"] == mode)
                .collect();
            let values = |key: &str| {
                group
                    .iter()
                    .filter_map(|r| r[key].as_f64())
                    .collect::<Vec<_>>()
            };
            let mean = |key: &str| {
                let v = values(key);
                if v.is_empty() {
                    None
                } else {
                    Some(v.iter().sum::<f64>() / v.len() as f64)
                }
            };
            aggregate.push(json!({"workload": workload, "mode": mode, "paired_seeds": group.len(),
                "median_tokens_per_second": median(&values("tokens_per_second")),
                "median_paired_wall_clock_speedup": median(&values("paired_wall_clock_speedup")),
                "min_paired_wall_clock_speedup": values("paired_wall_clock_speedup").into_iter().fold(f64::INFINITY, f64::min),
                "max_paired_wall_clock_speedup": values("paired_wall_clock_speedup").into_iter().fold(0.0, f64::max),
                "mean_memory_fallback_rate": mean("memory_fallback_rate"), "mean_vocabulary_fallback_rate": mean("vocabulary_fallback_rate"),
                "mean_vocabulary_routing_bypassed_reads": mean("vocabulary_routing_bypassed_reads"),
                "mean_scanned_slots": mean("mean_scanned_slots"),
                "mean_value_rows_mixed": mean("mean_value_rows_mixed"),
                "mean_memory_routing_bypassed_reads": mean("memory_routing_bypassed_reads"),
                "median_retrieval_queries_per_second": median(&values("retrieval_queries_per_second_varied_queries")),
                "median_build_inclusive_speedup": median(&values("single_repeat_speedup_including_build")),
                "min_build_inclusive_speedup": values("single_repeat_speedup_including_build").into_iter().fold(f64::INFINITY, f64::min),
                "max_actual_omitted_mass": values("max_actual_omitted_mass").into_iter().fold(0.0, f64::max),
                "max_omitted_mass_bound": values("max_omitted_mass_bound").into_iter().fold(0.0, f64::max),
                "max_abs_logit_error": values("max_abs_logit_error").into_iter().fold(0.0, f64::max),
                "max_abs_cross_entropy_delta": values("cross_entropy_delta").into_iter().map(f64::abs).fold(0.0, f64::max),
                "min_greedy_agreement": values("greedy_agreement").into_iter().fold(1.0, f64::min),
                "mean_cross_entropy_delta": mean("cross_entropy_delta"), "mean_greedy_agreement": mean("greedy_agreement"),
                "median_index_build_seconds": median(&values("index_build_seconds")),
                "mean_coordinate_work_proxy_speedup": mean("coordinate_work_proxy_speedup")}));
        }
    }
    let record = json!({"schema_version": 2, "experiment": "certified_sparse_inference_wall_clock", "backend": "cpu", "seeds": BENCHMARK_SEEDS,
        "reproduce_command": "cargo run --release -- benchmark --feature sparse --out __agent__/sparse_results",
        "protocol": {"memory_method": "GCSR: four-slot geometric groups, interval lower bounds, suffix mass certificate, half-bank work budget and bounded exact-path backoff",
            "shape": {"vocab": 2048, "latent": 256, "state": 16, "key": 32, "populated_memory": 512},
            "tokenizer": "UTF-8 byte+1 IDs, ID zero reserved; 2048 head rows for paper shape, unused byte-vocabulary rows retained",
            "training": "trained_bank: 32 AdamW updates x16 targets on built-in science corpus, constant lr0.003; same frozen weights for all inference modes",
            "heldout": "heldout_bank: train and populate512 slots from first5 science lines only; evaluate last4 lines, never inserted or used for updates; same32 updates x16 targets and paired frozen weights",
            "fixtures": "diffuse_fixture and separated_fixture are untrained diagnostic stress cases; separated is engineered for pruning and cannot establish trained-model speed",
            "timing": "five rotated paired batches per seed, 256 forced input tokens, resets every32; includes model forward, retrieval, head/greedy and certificate routing; no index build or quality scoring in timed batch; build-inclusive metric separate",
            "fallback": "CSR failure to terminate before exhaustive scan; CVP failure to prune any vocabulary rows; empty banks excluded (all banks have512 slots)",
            "quality": "teacher-forced exact full-head CE, including CSR changes; same routing warmup as timed runs; actual omitted mass checked against the full reader at every certified query; CVP is exact greedy only; all IDs checked against the same exhaustive head",
            "retrieval_timing": "one cold-routing replay of256 varied actual recurrent queries captured during quality scoring; excludes SSM/head and index build; secondary diagnostic, not end-to-end speed",
            "limitation": "five paired synthetic/local-corpus CPU seeds; no large-corpus quality, sampled/top-K CVP, GPU or energy claim; certificate inference adds index storage rather than reducing trained parameter memory"},
        "aggregate": aggregate, "measurements": rows});
    fs::write(
        dir.join("sparse_inference.json"),
        serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
