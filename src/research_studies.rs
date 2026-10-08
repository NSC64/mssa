//! Small isolated workers sharing the existing scalar/reference learning paths.
use super::*;
use std::hint::black_box;

#[path = "sparse_crossover.rs"]
mod sparse;

type Documents = Vec<(Vec<usize>, Vec<usize>)>;

fn positive(value: &str) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("expected positive integer: {value}"))
}

pub fn run(args: &[String]) -> Result<(), String> {
    let record = match args.first().map(String::as_str) {
        Some("language") if args.len() == 9 => language(
            &args[1], &args[2], &args[3], &args[4], positive(&args[5])?,
            args[6].parse().map_err(|_| "invalid seed")?,
            args[7].parse().map_err(|_| "invalid rate")?,
        )?,
        Some("timing") if args.len() == 4 => timing(
            args[1].parse().map_err(|_| "invalid seed")?, positive(&args[2])?,
        )?,
        Some("sparse") if args.len() == 3 => sparse::run(args[1].parse().map_err(|_| "invalid seed")?)?,
        Some("dream") if args.len() == 4 => dream(
            args[1].parse().map_err(|_| "invalid seed")?, positive(&args[2])?,
        )?,
        _ => return Err("usage: language METHOD TRAIN DEV TEST UPDATES SEED RATE OUTPUT | timing SEED UPDATES OUTPUT | sparse SEED OUTPUT | dream SEED UPDATES OUTPUT".into()),
    };
    let output = Path::new(args.last().unwrap());
    if output.exists() {
        return Err(format!("refusing to overwrite {}", output.display()));
    }
    fs::write(
        output,
        serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn rss() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|s| s.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

fn byte_config() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 257,
        d_latent: 32,
        d_state: 4,
        d_mem_key: 8,
        mem_capacity: 32,
        chunk_len: 32,
        weight_decay: 0.0,
        ..Default::default()
    }
}

fn documents(bytes: &[u8], limit: usize) -> Result<Documents, String> {
    if bytes.len() < 2 {
        return Err("scoring split needs at least two bytes".into());
    }
    Ok(bytes[..bytes.len().min(limit)]
        .chunks(257)
        .filter(|d| d.len() > 1)
        .map(|d| {
            (
                d[..d.len() - 1].iter().map(|&b| b as usize + 1).collect(),
                d[1..].iter().map(|&b| b as usize + 1).collect(),
            )
        })
        .collect())
}

// Score every target, rather than the synthetic benchmark's last-token metric.
fn score(learner: &mut Learner, docs: &Documents) -> Result<Value, String> {
    let model = learner.model();
    let mut carry = vec![0.0; model.recurrent_state_len()];
    model.copy_recurrent_state_to(&mut carry);
    let mut logits = vec![0.0; model.cfg.d_vocab];
    let (mut loss, mut correct, mut count) = (0.0, 0, 0);
    let result = (|| {
        for (inputs, targets) in docs {
            model.reset_recurrent_state();
            for (&input, &target) in inputs.iter().zip(targets) {
                model.try_forward_inference(input, &mut logits)?;
                if logits.iter().any(|x| !x.is_finite()) {
                    return Err("non-finite scoring logits".into());
                }
                loss += cross_entropy_f64(&logits, target);
                let id = (0..logits.len())
                    .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
                    .unwrap();
                correct += usize::from(id == target);
                count += 1;
            }
        }
        let ce = loss / count as f64;
        Ok(json!({"cross_entropy": ce, "perplexity": ce.exp(),
            "next_token_accuracy": correct as f64 / count as f64, "targets": count}))
    })();
    model.copy_recurrent_state_from(&carry);
    result
}

fn multiplier(step: usize, steps: usize) -> f32 {
    let warmup = 32.min(steps);
    if step <= warmup {
        return step as f32 / warmup as f32;
    }
    let phase = (step - warmup) as f64 / (steps - warmup) as f64;
    (0.1 + 0.45 * (1.0 + (std::f64::consts::PI * phase).cos())) as f32
}

fn parameter_digest(learner: &mut Learner) -> String {
    let model = learner.model();
    let mut hash = 0xcbf29ce484222325u64;
    for id in 0..tensor_count(model) {
        for &x in tensor(model, id).data.iter() {
            for b in x.to_bits().to_le_bytes() {
                hash = (hash ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
    }
    format!("{hash:016x}")
}

fn language(
    method: &str,
    train: &str,
    dev: &str,
    test: &str,
    updates: usize,
    seed: u64,
    rate: f32,
) -> Result<Value, String> {
    if ![
        "adamw",
        "interdiffusion",
        "interdiffusion_recurrent_only",
        "readout_only",
    ]
    .contains(&method)
    {
        return Err("invalid language method".into());
    }
    if !(rate.is_finite() && rate > 0.0) {
        return Err("rate must be finite and positive".into());
    }
    let training = fs::read(train).map_err(|e| e.to_string())?;
    let dev = documents(&fs::read(dev).map_err(|e| e.to_string())?, 8192)?;
    let test = documents(&fs::read(test).map_err(|e| e.to_string())?, 8192)?;
    if training.len() < 258 {
        return Err("training split needs at least 258 bytes".into());
    }
    let cfg = byte_config();
    let construction = Instant::now();
    let mut learner = Learner::new(method, cfg.clone(), seed, rate)?;
    let construction_seconds = construction.elapsed().as_secs_f64();
    let initial_digest = parameter_digest(&mut learner);
    let mut rng = SimpleRng::new(seed ^ 0x4441_5441);
    let mut digest = 0xcbf29ce484222325;
    let mut curve =
        vec![json!({"update": 0, "targets": 0, "development": score(&mut learner, &dev)?})];
    let (mut training_seconds, mut forward_tokens) = (0.0, 0);
    let mut start = 0;
    let mut failure = None;
    let mut completed = 0;
    let wall = Instant::now();
    for step in 1..=updates {
        let offset = (step - 1) % 8 * 32;
        if offset == 0 {
            start = rng.next_u32() as usize % (training.len() - 257);
        }
        let inputs: Vec<_> = training[start + offset..start + offset + 32]
            .iter()
            .map(|&b| b as usize + 1)
            .collect();
        let targets: Vec<_> = training[start + offset + 1..start + offset + 33]
            .iter()
            .map(|&b| b as usize + 1)
            .collect();
        hash_tokens(&mut digest, &inputs, &targets, offset == 0);
        learner.set_rate(rate * multiplier(step, updates))?;
        let at = Instant::now();
        match learner.step(&inputs, &targets, offset == 0) {
            Ok(passes) => forward_tokens += passes * inputs.len(),
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
        training_seconds += at.elapsed().as_secs_f64();
        completed = step;
        if step % 512 == 0 || step == updates {
            curve.push(json!({"update": step, "targets": step * 32,
                "training_stream_digest": format!("{digest:016x}"),
                "training_seconds": training_seconds, "wall_seconds": wall.elapsed().as_secs_f64(),
                "development": score(&mut learner, &dev)?}));
        }
    }
    let training_and_dev_seconds = wall.elapsed().as_secs_f64();
    let final_test = if failure.is_none() {
        Some(score(&mut learner, &test)?)
    } else {
        None
    };
    let storage = learner.storage_bytes();
    Ok(
        json!({"study": "language", "method": method, "seed": seed, "peak_rate": rate,
        "config": config_json(&cfg), "parameters": learner.parameters(), "initial_parameter_digest": initial_digest,
        "updates": completed, "target_tokens": completed * 32, "forward_tokens_including_probes": forward_tokens,
        "training_stream_digest": format!("{digest:016x}"), "numeric_storage_bytes": storage,
        "process_peak_rss_kib": rss(), "construction_seconds": construction_seconds,
        "training_seconds": training_seconds, "training_and_development_seconds": training_and_dev_seconds,
        "total_wall_seconds_including_test": wall.elapsed().as_secs_f64(),
        "target_tokens_per_second": completed as f64 * 32.0 / training_seconds,
        "failure": failure, "curve": curve, "test": final_test,
        "final_parameter_digest": parameter_digest(&mut learner)}),
    )
}

fn timing(seed: u64, updates: usize) -> Result<Value, String> {
    let cfg = byte_config();
    let mut learner = Learner::new("interdiffusion", cfg, seed, 0.003)?;
    let inputs: Vec<_> = (0..32)
        .map(|i| 1 + (i * 17 + seed as usize) % 256)
        .collect();
    let targets: Vec<_> = (0..32)
        .map(|i| 1 + ((i + 1) * 17 + seed as usize) % 256)
        .collect();
    // Warm up code/data before timing, identically in both implementations.
    for _ in 0..32 {
        learner.step(&inputs, &targets, true)?;
    }
    let begin = Instant::now();
    for _ in 0..updates {
        black_box(learner.step(&inputs, &targets, true)?);
    }
    let seconds = begin.elapsed().as_secs_f64();
    let accuracy = score(&mut learner, &vec![(inputs, targets)])?;
    Ok(
        json!({"study": "timing", "seed": seed, "updates": updates, "seconds": seconds,
        "config": config_json(&byte_config()), "target_tokens": updates * 32,
        "target_tokens_per_second": updates as f64 * 32.0 / seconds,
        "final_score": accuracy, "final_parameter_digest": parameter_digest(&mut learner),
        "process_peak_rss_kib": rss()}),
    )
}

fn cycle_doc(tokens: &[usize], phase: usize, len: usize) -> (Vec<usize>, Vec<usize>) {
    (
        (0..len)
            .map(|i| tokens[(phase + i) % tokens.len()])
            .collect(),
        (0..len)
            .map(|i| tokens[(phase + i + 1) % tokens.len()])
            .collect(),
    )
}

fn memory_digest(model: &PSSALayerV2) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for &x in model
        .memory
        .keys
        .iter()
        .chain(&model.memory.values)
        .chain(&model.memory.norm_sq)
    {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

fn dream(seed: u64, updates: usize) -> Result<Value, String> {
    let mut rows = Vec::new();
    // Shared context/embeddings, contradictory transitions: interference is intentional.
    let tasks: [&[usize]; 2] = [&[1, 2, 3, 4], &[1, 4, 3, 2]];
    let dev_a: Documents = (0..4).map(|phase| cycle_doc(tasks[0], phase, 48)).collect();
    let dev_b: Documents = (0..4).map(|phase| cycle_doc(tasks[1], phase, 48)).collect();
    for frozen in [false, true] {
        for policy in ["none", "replay", "replay_consolidate", "consolidate_only"] {
            let cfg = PSSAConfigV2 {
                d_vocab: 8,
                d_latent: 16,
                d_state: 3,
                d_mem_key: 4,
                mem_capacity: 8,
                chunk_len: 16,
                weight_decay: 0.0,
                ema_alpha: 0.05,
                ..Default::default()
            };
            let mut learner = Learner::new("adamw", cfg.clone(), seed, 0.003)?;
            let mut times = [0.0; 2];
            for step in 0..updates {
                let doc = cycle_doc(tasks[0], (step + seed as usize) % 4, 16);
                let at = Instant::now();
                learner.step(&doc.0, &doc.1, true)?;
                times[0] += at.elapsed().as_secs_f64();
            }
            let after_a = score(&mut learner, &dev_a)?;
            // Store genuine A query/value pairs for all controls. The insertion
            // is explicit to isolate replay from the production surprise gate.
            {
                let model = learner.model();
                let mut logits = vec![0.0; cfg.d_vocab];
                for token in 1..=4 {
                    model.reset_recurrent_state();
                    model.forward_inference(token, &mut logits);
                    let key = model.inf_q_pnc.clone();
                    let value = model.inf_z_final.clone();
                    model.memory.insert(&key, &value);
                }
            }
            let before_b_a = score(&mut learner, &dev_a)?;
            let before_b_b = score(&mut learner, &dev_b)?;
            let memory_before = memory_digest(learner.model());
            let mut replay_seconds = 0.0;
            let mut entries = 0;
            let mut adapter_delta = 0.0;
            let mut rng = SimpleRng::new(seed ^ 0x4452_4541_4d);
            let mut curve = Vec::new();
            for step in 0..updates {
                let doc = cycle_doc(tasks[1], (step + seed as usize) % 4, 16);
                let model = learner.model();
                let frozen_adapter = frozen.then(|| model.adapters[0].clone());
                model.reset_recurrent_state();
                let at = Instant::now();
                let loss = model.forward_train_chunk(&doc.0, &doc.1);
                if !loss.is_finite() {
                    return Err("dream study diverged".into());
                }
                model.zero_gradients();
                model.backward_chunk(16, 1.0);
                if frozen {
                    model.adapters[0].zero_grad();
                }
                if !matches!(
                    model.apply_adamw_with_grad_clip(0.003, 1.0),
                    crate::pssa::GradientClipOutcome::Applied { .. }
                ) {
                    return Err("dream study non-finite gradient".into());
                }
                if let Some(adapter) = frozen_adapter {
                    model.adapters[0] = adapter;
                }
                times[1] += at.elapsed().as_secs_f64();
                if (step + 1) % 8 == 0 {
                    let model = learner.model();
                    let saved_fast = model.adapters[0].up_proj.data.clone();
                    let saved_slow = model.adapters[0].consolidated_up.clone();
                    let replay_at = Instant::now();
                    match policy {
                        "replay" | "replay_consolidate" => {
                            let summary = model.dream_replay(
                                crate::dream::DreamMode::Memory,
                                4,
                                0,
                                0.8,
                                &mut rng,
                            )?;
                            entries += summary.entries_replayed;
                            if policy == "replay" {
                                // Production replay always consolidates. Undo only
                                // the coefficient transfer for this controlled arm,
                                // preserving its effective fast+slow weights.
                                let adapter = &mut model.adapters[0];
                                for (i, &slow) in saved_slow.iter().enumerate() {
                                    adapter.up_proj.data[i] += adapter.consolidated_up[i] - slow;
                                    adapter.consolidated_up[i] = slow;
                                }
                            }
                        }
                        "consolidate_only" => model.ema_consolidate_plasticity(),
                        _ => (),
                    }
                    if frozen {
                        model.adapters[0].up_proj.data.copy_from_slice(&saved_fast);
                        model.adapters[0]
                            .consolidated_up
                            .copy_from_slice(&saved_slow);
                    }
                    replay_seconds += replay_at.elapsed().as_secs_f64();
                    adapter_delta += model.adapters[0]
                        .up_proj
                        .data
                        .iter()
                        .zip(&model.adapters[0].consolidated_up)
                        .zip(saved_fast.iter().zip(&saved_slow))
                        .map(|((&a, &b), (&c, &d))| {
                            ((a as f64 + b as f64) - (c as f64 + d as f64)).powi(2)
                        })
                        .sum::<f64>()
                        .sqrt();
                }
                if (step + 1) % 32 == 0 || step + 1 == updates {
                    curve.push(
                        json!({"update": step+1, "old_task": score(&mut learner, &dev_a)?,
                        "new_task": score(&mut learner, &dev_b)?}),
                    );
                }
            }
            let memory_after = memory_digest(learner.model());
            if memory_before != memory_after {
                return Err("dream study altered frozen bank".into());
            }
            rows.push(
                json!({"policy": policy, "frozen_adapters": frozen, "seed": seed,
                "after_a_before_memory": after_a, "before_b_old_task": before_b_a,
                "before_b_new_task": before_b_b, "after_b_old_task": score(&mut learner, &dev_a)?,
                "after_b_new_task": score(&mut learner, &dev_b)?, "curve": curve,
                "training_seconds_a": times[0], "training_seconds_b": times[1],
                "replay_seconds": replay_seconds, "entries_replayed": entries,
                "sum_effective_adapter_replay_delta_norm": adapter_delta,
                "memory_before": memory_before, "memory_after": memory_after}),
            );
        }
    }
    Ok(
        json!({"study": "dream", "seed": seed, "updates_per_task": updates,
        "targets_per_task": updates * 16, "test_length": 48, "rows": rows,
        "process_peak_rss_kib": rss()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_preserves_carry_and_counts_all_targets() {
        let mut learner = Learner::new("adamw", byte_config(), 42, 0.003).unwrap();
        learner.step(&[1, 2], &[2, 3], true).unwrap();
        let before = learner.model().block.h_persistent.clone();
        let docs = documents(&vec![b'a'; 600], 600).unwrap();
        let result = score(&mut learner, &docs).unwrap();
        assert_eq!(result["targets"], 597);
        assert_eq!(before, learner.model().block.h_persistent);
        assert!(documents(&[1], 10).is_err());
    }
}
