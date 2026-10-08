//! Isolated-process, from-scratch CPU experiments. Each method gets the same
//! seeds/documents and two learning-rate trials, with separate development and
//! longer-sequence test sets. This is a synthetic learning probe, not an LLM
//! pretraining or general-knowledge result.

use super::*;
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command, time::Instant};

const METHODS: [&str; 9] = [
    "adamw",
    "sgd",
    "plain",
    "spectral_uniform",
    "interdiffusion_v1",
    "readout_only",
    "interdiffusion_probes",
    "interdiffusion_recurrent_only",
    "interdiffusion",
];
// 7301..7305 were development/tuning seeds. Freeze changes before these fresh
// evaluation pairs; report every pair, including failures.
pub(crate) const SEEDS: [u64; 5] = [7401, 7402, 7403, 7404, 7405];
const STEPS: usize = 2048;
const EVAL_EVERY: usize = 128;
const TARGET_LOSS: f64 = 0.75;
const QUALITY_LOSS_TOLERANCE: f64 = 0.05;
const WARMUP_UPDATES: usize = 32;

fn schedule(update: usize) -> f32 {
    if update <= WARMUP_UPDATES {
        return update as f32 / WARMUP_UPDATES as f32;
    }
    let progress = (update - WARMUP_UPDATES) as f64 / (STEPS - WARMUP_UPDATES) as f64;
    (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
}

fn hash_tokens(hash: &mut u64, inputs: &[usize], targets: &[usize], reset: bool) {
    for value in [inputs.len(), usize::from(reset)]
        .into_iter()
        .chain(inputs.iter().copied())
        .chain(targets.iter().copied())
    {
        for byte in (value as u64).to_le_bytes() {
            *hash = (*hash ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    }
}

fn write(dir: &Path, value: &Value) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    fs::write(
        dir.join("interdiffusion.json"),
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

pub(crate) fn run(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    for method in METHODS {
        let destination = dir.join(method);
        println!("Interdiffusion: isolated {method} worker");
        let child = Command::new(&exe)
            .args([
                "benchmark",
                "--feature",
                &format!("interdiffusion_{method}"),
                "--out",
            ])
            .arg(&destination)
            .output()
            .map_err(|e| format!("start {method}: {e}"))?;
        if !child.status.success() {
            return Err(format!(
                "{method} worker failed: {}",
                String::from_utf8_lossy(&child.stderr)
            ));
        }
    }
    report(dir)
}

/// Regenerate the aggregate after interrupted runs whose completed worker
/// records are retained. Fairness checks reject incompatible token streams.
pub(crate) fn report(dir: &Path) -> Result<(), String> {
    let mut workers = Vec::new();
    for method in METHODS {
        let raw =
            fs::read(dir.join(method).join("interdiffusion.json")).map_err(|e| e.to_string())?;
        let worker = serde_json::from_slice::<Value>(&raw).map_err(|e| e.to_string())?;
        if worker["method"] != method
            || worker["seeds"] != json!(SEEDS)
            || (!worker["schema"].is_null() && worker["schema"] != 2)
        {
            return Err(format!(
                "incompatible {method} worker record; rerun the full benchmark"
            ));
        }
        workers.push(worker);
    }
    let token_comparisons = matched_tokens(&workers)?;
    let matched = matched_quality(&workers);
    let aggregates = aggregate_quality(&workers, &matched);
    let record = json!({
        "experiment": "interdiffusion_forward_only", "backend": "cpu",
        "reproduce_command": "cargo run --release -- benchmark --feature interdiffusion --out __agent__/interdiffusion_results",
        "protocol": {
            "initialization": "same stable MSSA initialization and seeds for every method; trained from scratch",
            "selection": "AdamW and full Interdiffusion share four peak-rate candidates; ablations use two; select lowest final development loss only; held-out curves never select a trial",
            "tokenizer": "fixed synthetic token IDs shared by every method; byte-text uses UTF-8 byte+1 IDs with ID 0 reserved, no vocabulary fitting",
            "schedule": {"kind": "linear warmup then cosine", "warmup_updates": WARMUP_UPDATES, "final_rate_fraction": 0.1, "common_adamw_and_interdiffusion_peak_rates": [0.001, 0.003, 0.01, 0.08]},
            "paired_seeds": SEEDS,
            "development_tuning_seeds": [7301, 7302, 7303, 7304, 7305],
            "confirmatory_policy": "optimizer implementation frozen before fresh 7401..7405 pairs; common rate grid expanded to include low rates after development overfitting was detected; no test-based reranking; exploratory evidence, not a preregistered trial",
            "fairness_check": "every selected method/task/seed must match AdamW's target-token count, stream digest and schedule; all five seeds retained",
            "cycle_lengths": {"training_document": 32, "training_chunk": 16, "development": 24, "test": 48},
            "recall_gaps": {"training_min": 2, "training_max": 10, "development": [8, 10], "test": [11, 12, 13, 14]},
            "development_generalization": "development exercises long training contexts; held-out test contexts extend further; test data never select learning rates",
            "state_policy": "cycle carries detached state across two 16-token updates per document; recall resets each document; frozen empty episodic bank; writes and consolidation disabled for every method",
            "v1_update": "one tensor per step in shared cyclic order; one RMS-normalized direction; two probes plus one original-weight carry replay",
            "v2_update": "local readout/MLP/adapter gradients, streaming cell-A and cosine-row Delta/B eligibility, and forward input/norm tangents (up to eight input rows); no reverse-time tape",
            "v2_readout": "local Adam on uncertain chunks; clipped diagonal CE-curvature step on chunks with CE <0.1; Adam moments maintained in both paths; unit-L2 cosine adaptive steps scaled by sqrt(width) to unit RMS",
            "recurrent_only_ablation": "same recurrent/local eligibility but sparse embedding/norm finite-difference probes every 16 steps",
            "probes_ablation": "local readout AdamW plus generic spectral body probes every four steps; no eligibility traces",
            "readout_only": "same local readout AdamW with the randomly initialized body frozen; ablation control",
            "spectral_basis": "separable orthonormal DCT-II real Fourier modes, sampled without replacement across the full frequency band",
            "memory_metric": "owned numeric Vec capacities; includes active-tensor backup/direction and cosine workspace; excludes headers/allocator/code/datasets",
            "rss_metric": "each method runs in a fresh process; VmHWM includes one CLI-shape allocation/update followed by all tiny training trials",
            "matched_quality": "for each task/seed, first sampled development point with loss <= 1.05 * selected AdamW final development loss AND last-token accuracy >= AdamW final development accuracy; no test-based tuning; misses remain null",
            "matched_quality_timing": "128-update resolution; includes periodic development evaluation; excludes initialization and LR search; separate two-candidate total times are reported; speedup is a paired per-seed ratio",
            "gradient_clipping": "backprop uses global norm 1; v2 clips readout, eligibility body, and finite-difference tensor separately to norm 1",
            "byte_text": {"training_lines": [1,2,3,4,5], "development_lines": [6,7], "test_lines": [8,9],
                "corpus": "built-in SCIENCE_REFERENCE_CORPUS, nine sentences only", "vocab": 257, "latent": 24, "state": 4},
            "limitation": "synthetic cycle/recall and tiny nine-sentence byte-text diagnostic; no corpus-scale, CUDA or broad-domain superiority claim",
            "sgd_storage": "SGD baseline retains legacy allocated Adam arrays; it tests learning dynamics, not an optimized SGD memory layout",
        },
        "matched_adamw_quality": matched,
        "cross_entropy_at_matched_tokens": token_comparisons,
        "aggregate_quality": aggregates,
        "workers": workers,
    });
    write(dir, &record)
}

fn matched_tokens(workers: &[Value]) -> Result<Vec<Value>, String> {
    let reference = workers.iter().find(|w| w["method"] == "adamw").unwrap();
    let mut rows = Vec::new();
    for worker in workers {
        for trial in worker["selected_by_development_loss"].as_array().unwrap() {
            let adam = reference["selected_by_development_loss"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["task"] == trial["task"] && a["seed"] == trial["seed"])
                .unwrap();
            for key in [
                "unique_training_target_tokens",
                "training_stream_digest",
                "schedule",
                "tokenizer",
                "config",
            ] {
                if trial[key] != adam[key] {
                    return Err(format!(
                        "{} {} seed {} mismatched {key}",
                        trial["method"], trial["task"], trial["seed"]
                    ));
                }
            }
            for point in trial["curve"].as_array().unwrap() {
                let paired = adam["curve"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|p| {
                        p["update"] == point["update"]
                            && p["training_target_tokens"] == point["training_target_tokens"]
                    })
                    .ok_or("unmatched target-token checkpoint")?;
                if point["training_stream_digest"] != paired["training_stream_digest"] {
                    return Err("checkpoint token stream differs from AdamW".into());
                }
                if point["learning_rate_multiplier"] != paired["learning_rate_multiplier"] {
                    return Err("checkpoint learning-rate schedule differs from AdamW".into());
                }
                rows.push(json!({"method": trial["method"], "task": trial["task"], "seed": trial["seed"],
                    "training_target_tokens": point["training_target_tokens"], "update": point["update"],
                    "test_loss": point["test_loss"], "adamw_test_loss": paired["test_loss"],
                    "paired_test_loss_delta": point["test_loss"].as_f64().unwrap() - paired["test_loss"].as_f64().unwrap(),
                    "training_seconds": point["elapsed_training_seconds"], "wall_seconds": point["elapsed_seconds"]}));
            }
        }
    }
    Ok(rows)
}

fn paired_interval(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut rng = SimpleRng::new(0x5041_4952_4544);
    let mut samples = Vec::with_capacity(4096);
    for _ in 0..4096 {
        samples.push(
            (0..values.len())
                .map(|_| values[rng.next_u32() as usize % values.len()])
                .sum::<f64>()
                / values.len() as f64,
        );
    }
    samples.sort_by(f64::total_cmp);
    json!({"mean": values.iter().sum::<f64>() / values.len() as f64,
        "bootstrap_95_percent_interval": [samples[102], samples[3993]], "pairs": values.len()})
}

fn meets_quality(point: &Value, loss: f64, accuracy: f64) -> bool {
    point["development_loss"].as_f64().unwrap() <= loss
        && point["development_last_token_accuracy"].as_f64().unwrap() >= accuracy
}

fn matched_quality(workers: &[Value]) -> Vec<Value> {
    let reference = workers.iter().find(|w| w["method"] == "adamw").unwrap();
    let mut rows = Vec::new();
    for worker in workers {
        for trial in worker["selected_by_development_loss"].as_array().unwrap() {
            let adam = reference["selected_by_development_loss"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["task"] == trial["task"] && a["seed"] == trial["seed"])
                .unwrap();
            let target =
                adam["development_loss"].as_f64().unwrap() * (1.0 + QUALITY_LOSS_TOLERANCE);
            let accuracy = adam["development_last_token_accuracy"].as_f64().unwrap();
            let first_match = |t: &Value| {
                t["curve"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|p| meets_quality(p, target, accuracy))
                    .map(|p| (p["update"].clone(), p["elapsed_seconds"].as_f64().unwrap()))
            };
            let found = first_match(trial);
            let adam_found = first_match(adam).unwrap();
            let speedup = found
                .as_ref()
                .filter(|p| p.1 > 0.0)
                .map(|p| adam_found.1 / p.1);
            rows.push(json!({
                "method": worker["method"], "task": trial["task"], "seed": trial["seed"],
                "target_development_loss": target, "target_development_last_token_accuracy": accuracy,
                "updates_to_adamw_quality": found.as_ref().map(|p| &p.0),
                "seconds_to_adamw_quality": found.as_ref().map(|p| p.1),
                "adamw_seconds_to_same_quality": adam_found.1,
                "paired_speedup_over_adamw": speedup,
                "final_development_meets_adamw_quality": meets_quality(trial, target, accuracy),
                "target_already_met_at_initialization": trial["curve"].as_array().unwrap().first()
                    .is_some_and(|p| meets_quality(p, target, accuracy)),
                "final_test_meets_adamw_quality": trial["test_loss"].as_f64().unwrap() <= adam["test_loss"].as_f64().unwrap() * (1.0 + QUALITY_LOSS_TOLERANCE)
                    && trial["test_last_token_accuracy"].as_f64().unwrap() >= adam["test_last_token_accuracy"].as_f64().unwrap(),
            }));
        }
    }
    rows
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

fn aggregate_quality(workers: &[Value], matched: &[Value]) -> Vec<Value> {
    let mut rows = Vec::new();
    for worker in workers {
        for task in ["cycle", "delayed_recall", "byte_text"] {
            let trials: Vec<_> = worker["selected_by_development_loss"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["task"] == task)
                .collect();
            if trials.is_empty() {
                continue;
            }
            let comparisons: Vec<_> = matched
                .iter()
                .filter(|r| r["method"] == worker["method"] && r["task"] == task)
                .collect();
            let mean = |key: &str| {
                trials.iter().map(|t| t[key].as_f64().unwrap()).sum::<f64>() / trials.len() as f64
            };
            let adam = workers.iter().find(|w| w["method"] == "adamw").unwrap();
            let deltas: Vec<_> = trials
                .iter()
                .map(|t| {
                    let paired = adam["selected_by_development_loss"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|a| a["seed"] == t["seed"] && a["task"] == t["task"])
                        .unwrap();
                    t["test_loss"].as_f64().unwrap() - paired["test_loss"].as_f64().unwrap()
                })
                .collect();
            let times: Vec<_> = comparisons
                .iter()
                .filter_map(|r| r["seconds_to_adamw_quality"].as_f64())
                .collect();
            let speedups: Vec<_> = comparisons
                .iter()
                .filter_map(|r| r["paired_speedup_over_adamw"].as_f64())
                .collect();
            let tuning_times = trials
                .iter()
                .map(|selected| {
                    worker["trials"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|t| t["task"] == task && t["seed"] == selected["seed"])
                        .map(|t| {
                            t["elapsed_training_and_development_seconds"]
                                .as_f64()
                                .unwrap()
                        })
                        .sum()
                })
                .collect();
            rows.push(json!({
                "method": worker["method"], "task": task, "seeds": trials.len(),
                "mean_test_loss": mean("test_loss"), "mean_test_last_token_accuracy": mean("test_last_token_accuracy"),
                "mean_final_development_loss": mean("development_loss"),
                "matched_development_seeds": times.len(),
                "final_development_matching_seeds": comparisons.iter().filter(|r| r["final_development_meets_adamw_quality"] == true).count(),
                "final_test_matching_seeds": comparisons.iter().filter(|r| r["final_test_meets_adamw_quality"] == true).count(),
                "median_seconds_to_adamw_quality_successful_seeds": median(times),
                "median_paired_speedup_over_adamw_all_seeds": if speedups.len() == trials.len() { median(speedups) } else { None },
                "median_selected_training_seconds": median(trials.iter().map(|t| t["elapsed_training_and_development_seconds"].as_f64().unwrap()).collect()),
                "median_learning_rate_search_seconds": median(tuning_times),
                "median_training_only_seconds": median(trials.iter().map(|t| t["elapsed_training_seconds"].as_f64().unwrap()).collect()),
                "median_training_target_tokens_per_second": median(trials.iter().map(|t| t["training_target_tokens_per_second"].as_f64().unwrap()).collect()),
                "scheduled_target_tokens_per_seed": trials.iter().map(|t| json!({"seed": t["seed"], "tokens": t["unique_training_target_tokens"]})).collect::<Vec<_>>(),
                "paired_test_loss_delta": paired_interval(&deltas),
                "test_loss_better_than_adamw_seeds": deltas.iter().filter(|&&x| x < 0.0).count(),
            }));
        }
    }
    rows
}

pub(crate) fn worker(dir: &Path, method: &str) -> Result<(), String> {
    if !METHODS.contains(&method) {
        return Err(format!("unknown Interdiffusion method '{method}'"));
    }
    // A common realistic shape makes RSS/model-storage comparisons meaningful
    // despite the small learning task. It is dropped before the tiny trials.
    let cli_cfg = PSSAConfigV2 {
        d_vocab: 2048,
        weight_decay: 0.0,
        ..Default::default()
    };
    let profile = {
        let mut learner = Learner::new(method, cli_cfg.clone(), SEEDS[0], rates(method)[0])?;
        let (inputs, targets) = cycle(0, 8);
        learner.step(&inputs, &targets, true)?;
        json!({"config": config_json(&cli_cfg), "parameters": learner.parameters(),
            "numeric_storage_bytes": learner.storage_bytes(), "warmup_target_tokens": inputs.len()})
    };
    let cfg = PSSAConfigV2 {
        d_vocab: 8,
        d_latent: 12,
        d_state: 3,
        d_mem_key: 4,
        mem_capacity: 8,
        chunk_len: 16,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut trials = Vec::new();
    let mut selected = Vec::new();
    for task in ["cycle", "delayed_recall", "byte_text"] {
        if task == "byte_text" && method != "adamw" && method != "interdiffusion" {
            continue;
        }
        let task_cfg = if task == "byte_text" {
            PSSAConfigV2 {
                d_vocab: 257,
                d_latent: 24,
                d_state: 4,
                d_mem_key: 4,
                mem_capacity: 8,
                chunk_len: 16,
                weight_decay: 0.0,
                ..Default::default()
            }
        } else {
            cfg.clone()
        };
        for seed in SEEDS {
            let mut candidates = Vec::new();
            for &lr in rates(method) {
                let result = trial(method, task, seed, lr, &task_cfg)?;
                candidates.push(result.clone());
                trials.push(result);
            }
            let best = candidates
                .into_iter()
                .min_by(|a, b| {
                    a["development_loss"]
                        .as_f64()
                        .unwrap()
                        .total_cmp(&b["development_loss"].as_f64().unwrap())
                })
                .unwrap();
            println!(
                "{method}/{task} seed={seed}: dev CE {:.3e}, test CE {:.3e}, last accuracy {:.3}, train {:.0} target tokens/s",
                best["development_loss"].as_f64().unwrap(),
                best["test_loss"].as_f64().unwrap(),
                best["test_last_token_accuracy"].as_f64().unwrap(),
                best["training_target_tokens_per_second"].as_f64().unwrap()
            );
            selected.push(best);
        }
    }
    let rss = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|raw| {
            raw.lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        });
    let cpu = fs::read_to_string("/proc/cpuinfo").ok().and_then(|raw| {
        raw.lines()
            .find(|line| line.starts_with("model name"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, name)| name.trim().to_owned())
    });
    write(
        dir,
        &json!({
            "method": method, "backend": "cpu", "hardware": cpu,
            "schema": 2,
            "process_peak_rss_kib": rss, "cli_shape_storage": profile,
            "config": config_json(&cfg), "steps": STEPS, "seeds": SEEDS,
            "learning_rate_candidates": rates(method), "epsilon": 0.001,
            "modes": 8, "smoothing": if method == "interdiffusion_v1" {4.0} else {0.0},
             "body_every": match method {"interdiffusion" | "interdiffusion_recurrent_only" => Some(16), "interdiffusion_probes" => Some(4), "readout_only" => Some(0), _ => None},
             "body_learning_rate": if method.starts_with("interdiffusion") && method != "interdiffusion_v1" {"head_learning_rate"} else {"see method"},
             "streaming_eligibility": method == "interdiffusion" || method == "interdiffusion_recurrent_only",
             "input_eligibility": method == "interdiffusion",
            "target_development_loss": TARGET_LOSS, "evaluation_every_updates": EVAL_EVERY,
            "trials": trials, "selected_by_development_loss": selected,
        }),
    )
}

fn config_json(c: &PSSAConfigV2) -> Value {
    json!({"depth": c.depth, "loops": 1, "d_vocab": c.d_vocab, "d_latent": c.d_latent,
        "d_state": c.d_state, "d_mem_key": c.d_mem_key, "mem_capacity": c.mem_capacity,
        "chunk_len": c.chunk_len, "weight_decay": c.weight_decay})
}

fn rates(method: &str) -> &'static [f32] {
    if method == "adamw" || method == "interdiffusion" {
        &[0.001, 0.003, 0.01, 0.08]
    } else if [
        "interdiffusion",
        "readout_only",
        "interdiffusion_probes",
        "interdiffusion_recurrent_only",
    ]
    .contains(&method)
    {
        &[0.01, 0.08]
    } else {
        &[0.02, 0.08]
    }
}

enum Learner {
    Backprop { model: Box<PSSALayerV2>, adam: bool },
    Forward(Box<ZerothOrderTrainer>),
    Adaptive(Box<InterdiffusionTrainer>),
}

impl Learner {
    fn set_rate(&mut self, rate: f32) -> Result<(), String> {
        match self {
            Self::Backprop { model, .. } => model.cfg.lr = rate,
            Self::Forward(trainer) => trainer.config.learning_rate = rate,
            Self::Adaptive(trainer) => trainer.set_learning_rates(rate, rate)?,
        }
        Ok(())
    }
    fn new(method: &str, cfg: PSSAConfigV2, seed: u64, lr: f32) -> Result<Self, String> {
        if method == "adamw" || method == "sgd" {
            let mut cfg = cfg;
            cfg.lr = lr;
            Ok(Self::Backprop {
                model: Box::new(PSSALayerV2::new(cfg, seed)),
                adam: method == "adamw",
            })
        } else if [
            "interdiffusion",
            "readout_only",
            "interdiffusion_probes",
            "interdiffusion_recurrent_only",
        ]
        .contains(&method)
        {
            Ok(Self::Adaptive(Box::new(InterdiffusionTrainer::new(
                cfg,
                seed,
                InterdiffusionConfig {
                    head_learning_rate: lr,
                    body_learning_rate: lr,
                    body_every: match method {
                        "readout_only" => 0,
                        "interdiffusion_probes" => 4,
                        _ => 16,
                    },
                    eligibility: method == "interdiffusion"
                        || method == "interdiffusion_recurrent_only",
                    input_eligibility: method == "interdiffusion",
                    curvature_readout: method == "interdiffusion",
                    ..Default::default()
                },
            )?)))
        } else {
            let config = ZerothOrderConfig {
                kind: if method == "plain" {
                    DirectionKind::Plain
                } else {
                    DirectionKind::Spectral
                },
                learning_rate: lr,
                smoothing: if method == "interdiffusion_v1" {
                    4.0
                } else {
                    0.0
                },
                ..Default::default()
            };
            Ok(Self::Forward(Box::new(ZerothOrderTrainer::new(
                cfg, seed, config,
            )?)))
        }
    }

    fn model(&mut self) -> &mut PSSALayerV2 {
        match self {
            Self::Backprop { model, .. } => model,
            Self::Forward(trainer) => &mut trainer.forward.model,
            Self::Adaptive(trainer) => &mut trainer.forward.model,
        }
    }

    fn parameters(&mut self) -> usize {
        self.model().parameter_count()
    }

    fn storage_bytes(&self) -> usize {
        match self {
            Self::Backprop { model, .. } => {
                crate::checkpoint::allocation_bytes(&model.cfg).unwrap()
                    + model.bitnet_activation_q.capacity()
            }
            Self::Forward(trainer) => trainer.numeric_storage_bytes(),
            Self::Adaptive(trainer) => trainer.numeric_storage_bytes(),
        }
    }

    fn step(&mut self, inputs: &[usize], targets: &[usize], reset: bool) -> Result<usize, String> {
        match self {
            Self::Adaptive(trainer) => Ok(trainer
                .train_step(inputs, targets, reset, false)?
                .forward_evaluations),
            Self::Forward(trainer) => Ok(trainer
                .train_step(inputs, targets, reset, false)?
                .forward_evaluations),
            Self::Backprop { model, adam } => {
                if reset {
                    model.reset_recurrent_state();
                }
                let loss = model.forward_train_chunk(inputs, targets);
                if !loss.is_finite() {
                    return Err("non-finite backprop loss".into());
                }
                model.zero_gradients();
                model.backward_chunk(inputs.len(), 1.0);
                if *adam {
                    if !matches!(
                        model.apply_adamw_with_grad_clip(model.cfg.lr, 1.0),
                        crate::pssa::GradientClipOutcome::Applied { .. }
                    ) {
                        return Err("non-finite AdamW gradient".into());
                    }
                } else {
                    let norm = (0..tensor_count(model))
                        .map(|id| {
                            tensor(model, id)
                                .grad
                                .iter()
                                .map(|&g| (g as f64).powi(2))
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                        .sqrt();
                    if !norm.is_finite() {
                        return Err("non-finite SGD gradient".into());
                    }
                    let rate = model.cfg.lr as f64 / norm.max(1.0);
                    for id in 0..tensor_count(model) {
                        let t = tensor(model, id);
                        for (w, &g) in t.data.iter_mut().zip(t.grad.iter()) {
                            *w = (*w as f64 - rate * g as f64) as f32;
                        }
                    }
                    model.step_counter += 1;
                }
                Ok(1)
            }
        }
    }

    fn score(&mut self, docs: &[(Vec<usize>, Vec<usize>)]) -> Result<(f64, f64), String> {
        let model = self.model();
        let mut carry = vec![0.0; model.recurrent_state_len()];
        model.copy_recurrent_state_to(&mut carry);
        let mut logits = vec![0.0; model.cfg.d_vocab];
        let (mut loss, mut tokens, mut last_correct) = (0.0, 0, 0);
        let result = (|| {
            for (inputs, targets) in docs {
                model.reset_recurrent_state();
                for (index, (&input, &target)) in inputs.iter().zip(targets).enumerate() {
                    model.try_forward_inference(input, &mut logits)?;
                    if logits.iter().any(|x| !x.is_finite()) {
                        return Err("non-finite scoring logits".into());
                    }
                    loss += cross_entropy_f64(&logits, target);
                    tokens += 1;
                    if index + 1 == inputs.len() {
                        let guess = logits
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                            .unwrap()
                            .0;
                        last_correct += usize::from(guess == target);
                    }
                }
            }
            Ok((
                loss / tokens as f64,
                last_correct as f64 / docs.len() as f64,
            ))
        })();
        model.copy_recurrent_state_from(&carry);
        result
    }
}

fn cycle(phase: usize, len: usize) -> (Vec<usize>, Vec<usize>) {
    (
        (0..len).map(|i| 1 + (phase + i) % 6).collect(),
        (0..len).map(|i| 1 + (phase + i + 1) % 6).collect(),
    )
}

fn recall(key: usize, gap: usize) -> (Vec<usize>, Vec<usize>) {
    let mut doc = vec![key];
    doc.extend(std::iter::repeat_n(3, gap));
    doc.extend([5, key]);
    (doc[..doc.len() - 1].to_vec(), doc[1..].to_vec())
}

fn trial(
    method: &str,
    task: &str,
    seed: u64,
    lr: f32,
    cfg: &PSSAConfigV2,
) -> Result<Value, String> {
    let mut learner = Learner::new(method, cfg.clone(), seed, lr)?;
    let text_docs: Vec<Vec<usize>> = crate::dataset::DatasetManager::SCIENCE_REFERENCE_CORPUS
        .lines()
        .map(|line| line.bytes().map(|b| b as usize + 1).collect())
        .collect();
    let transitions = |docs: &[Vec<usize>]| {
        docs.iter()
            .map(|d| (d[..d.len() - 1].to_vec(), d[1..].to_vec()))
            .collect::<Vec<_>>()
    };
    let mut data_rng = SimpleRng::new(seed ^ 0x4441_5441);
    let dev = if task == "cycle" {
        (0..4).map(|p| cycle(p, 24)).collect::<Vec<_>>()
    } else if task == "delayed_recall" {
        [1, 2]
            .into_iter()
            .flat_map(|key| [8, 10].map(|gap| recall(key, gap)))
            .collect()
    } else {
        transitions(&text_docs[5..7])
    };
    let test = if task == "cycle" {
        (4..6).map(|p| cycle(p, 48)).collect::<Vec<_>>()
    } else if task == "delayed_recall" {
        [1, 2]
            .into_iter()
            .flat_map(|key| [11, 12, 13, 14].map(|gap| recall(key, gap)))
            .collect()
    } else {
        transitions(&text_docs[7..9])
    };
    let (initial, initial_accuracy) = learner.score(&dev)?;
    let (initial_test, initial_test_accuracy) = learner.score(&test)?;
    let start = Instant::now();
    let (mut seen, mut forward_tokens, mut reached, mut time_to_quality) = (0, 0, None, None);
    let mut curve = vec![
        json!({"update": 0, "elapsed_seconds": 0.0, "development_loss": initial,
        "development_last_token_accuracy": initial_accuracy, "training_target_tokens": 0,
        "test_loss": initial_test, "test_last_token_accuracy": initial_test_accuracy,
        "elapsed_training_seconds": 0.0}),
    ];
    let mut dev_loss = initial;
    let mut dev_accuracy = initial_accuracy;
    let mut phase = 0;
    let mut train_seconds = 0.0;
    let mut stream_hash = 0xcbf29ce484222325u64;
    let (mut text_doc, mut text_cursor) = (0, 0);
    for update in 1..=STEPS {
        let reset = if task == "byte_text" {
            text_cursor == 0
        } else {
            task != "cycle" || update % 2 == 1
        };
        let (inputs, targets) = if task == "cycle" {
            if reset {
                phase = data_rng.next_u32() as usize % 6;
            }
            let doc = cycle(phase, 16);
            phase = (phase + 16) % 6;
            doc
        } else if task == "delayed_recall" {
            let key = 1 + data_rng.next_u32() as usize % 2;
            let gap = 2 + data_rng.next_u32() as usize % 9;
            recall(key, gap)
        } else {
            if reset {
                text_doc = data_rng.next_u32() as usize % 5;
            }
            let doc = &text_docs[text_doc];
            let n = cfg.chunk_len.min(doc.len() - 1 - text_cursor);
            let pair = (
                doc[text_cursor..text_cursor + n].to_vec(),
                doc[text_cursor + 1..text_cursor + n + 1].to_vec(),
            );
            text_cursor += n;
            if text_cursor + 1 == doc.len() {
                text_cursor = 0;
            }
            pair
        };
        hash_tokens(&mut stream_hash, &inputs, &targets, reset);
        learner.set_rate(lr * schedule(update))?;
        let step_start = Instant::now();
        let forwards = learner.step(&inputs, &targets, reset)?;
        train_seconds += step_start.elapsed().as_secs_f64();
        seen += inputs.len();
        forward_tokens += inputs.len() * forwards;
        if update % EVAL_EVERY == 0 {
            (dev_loss, dev_accuracy) = learner.score(&dev)?;
            let (test_loss, test_accuracy) = learner.score(&test)?;
            if !dev_loss.is_finite() {
                return Err(format!("{method}/{task}/{seed} diverged"));
            }
            let seconds = start.elapsed().as_secs_f64();
            if reached.is_none() && dev_loss <= TARGET_LOSS {
                reached = Some(update);
                time_to_quality = Some(seconds);
            }
            curve.push(
                json!({"update": update, "elapsed_seconds": seconds, "development_loss": dev_loss,
                    "development_last_token_accuracy": dev_accuracy, "training_target_tokens": seen,
                    "training_stream_digest": format!("{stream_hash:016x}"),
                    "test_loss": test_loss, "test_last_token_accuracy": test_accuracy,
                    "elapsed_training_seconds": train_seconds, "learning_rate_multiplier": schedule(update)}),
            );
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let (test_loss, last_accuracy) = learner.score(&test)?;
    Ok(json!({
        "method": method, "task": task, "seed": seed, "learning_rate": lr,
        "config": config_json(cfg), "tokenizer": if task == "byte_text" {"utf8_byte_plus_one_v257"} else {"synthetic_ids_v8"},
        "parameters": learner.parameters(), "numeric_storage_bytes": learner.storage_bytes(),
        "initial_development_loss": initial, "development_loss": dev_loss,
        "development_last_token_accuracy": dev_accuracy,
        "test_loss": test_loss, "test_last_token_accuracy": last_accuracy,
        "updates": STEPS, "unique_training_target_tokens": seen,
        "training_forward_target_tokens_including_probes": forward_tokens,
        "backward_target_tokens": if method == "adamw" || method == "sgd" {seen} else {0},
        "elapsed_training_and_development_seconds": seconds,
        "elapsed_training_and_evaluation_seconds": seconds,
        "elapsed_training_seconds": train_seconds,
        "training_target_tokens_per_second": seen as f64 / train_seconds,
        "end_to_end_training_target_tokens_per_second": seen as f64 / seconds,
        "training_stream_digest": format!("{stream_hash:016x}"),
        "schedule": "warmup32_cosine_to_0.1",
        "updates_to_target_development_loss": reached, "seconds_to_target_development_loss": time_to_quality,
        "curve": curve,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_confidence_loss_and_common_schedule_remain_measurable() {
        let loss = cross_entropy_f64(&[100.0, 0.0], 0);
        assert!(loss > 0.0 && (loss / (-100.0f64).exp() - 1.0).abs() < 1e-12);
        assert_eq!(cross_entropy_f64(&[100.0, 0.0], 1), 100.0);
        assert_eq!(schedule(32), 1.0);
        assert!((schedule(STEPS) - 0.1).abs() < 1e-7);
        assert_eq!(rates("adamw"), rates("interdiffusion"));
        let mut a = 1;
        let mut b = 1;
        hash_tokens(&mut a, &[1, 2], &[2, 3], true);
        hash_tokens(&mut b, &[1, 2], &[2, 3], false);
        assert_ne!(a, b);
    }

    #[test]
    fn matched_token_report_rejects_stream_or_exposure_drift() {
        let trial = json!({"task": "cycle", "seed": 7401, "unique_training_target_tokens": 16,
            "training_stream_digest": "same", "schedule": "same", "tokenizer": "same", "config": {},
            "curve": [{"update": 1, "training_target_tokens": 16, "training_stream_digest": "same",
                "learning_rate_multiplier": 0.5, "test_loss": 0.4, "elapsed_seconds": 0.2, "elapsed_training_seconds": 0.1}]});
        let worker = |method: &str| json!({"method": method, "selected_by_development_loss": [trial.clone()]});
        let mut workers = vec![worker("adamw"), worker("interdiffusion")];
        assert_eq!(matched_tokens(&workers).unwrap().len(), 2);
        workers[1]["selected_by_development_loss"][0]["unique_training_target_tokens"] = json!(15);
        assert!(
            matched_tokens(&workers)
                .unwrap_err()
                .contains("unique_training_target_tokens")
        );
        workers[1] = worker("interdiffusion");
        workers[1]["selected_by_development_loss"][0]["curve"][0]["training_stream_digest"] =
            json!("different");
        assert!(
            matched_tokens(&workers)
                .unwrap_err()
                .contains("token stream")
        );
    }

    #[test]
    fn quality_matching_requires_accuracy_and_reports_unreached_targets() {
        let point = |update, seconds, loss, accuracy| {
            json!({
                "update": update, "elapsed_seconds": seconds, "development_loss": loss,
                "development_last_token_accuracy": accuracy,
            })
        };
        let trial = |curve, loss, accuracy| {
            json!({
                "seed": 7301, "task": "delayed_recall", "curve": curve,
                "development_loss": loss, "development_last_token_accuracy": accuracy,
                "test_loss": 0.4, "test_last_token_accuracy": accuracy,
            })
        };
        let worker = |method, t| json!({"method": method, "selected_by_development_loss": [t]});
        let rows = matched_quality(&[
            worker(
                "adamw",
                trial(
                    vec![point(128, 0.3, 0.5, 1.0), point(256, 0.6, 0.4, 1.0)],
                    0.4,
                    1.0,
                ),
            ),
            worker(
                "better",
                trial(
                    vec![point(128, 0.1, 0.2, 0.5), point(256, 0.2, 0.41, 1.0)],
                    0.41,
                    1.0,
                ),
            ),
            worker("miss", trial(vec![point(128, 0.1, 0.2, 0.5)], 0.2, 0.5)),
        ]);
        assert_eq!(rows[1]["updates_to_adamw_quality"], 256);
        assert_eq!(rows[1]["seconds_to_adamw_quality"], 0.2);
        assert!((rows[1]["paired_speedup_over_adamw"].as_f64().unwrap() - 3.0).abs() < 1e-12);
        assert_eq!(rows[1]["final_test_meets_adamw_quality"], true);
        assert!(rows[2]["seconds_to_adamw_quality"].is_null());
        assert!(rows[2]["paired_speedup_over_adamw"].is_null());
        assert_eq!(rows[2]["final_development_meets_adamw_quality"], false);
        assert_eq!(rows[2]["final_test_meets_adamw_quality"], false);
    }
}
