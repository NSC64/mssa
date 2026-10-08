//! Bounded factorial GCSR/CVP crossover study; quality auditing is untimed.
use super::*;
use crate::memory::HyperbolicEpisodicBankV2;
use crate::sparse_inference::{CertifiedMemoryIndex, CertifiedVocabularyIndex};

const MODES: [&str; 4] = ["dense", "csr", "cvp", "dual"];

fn install(model: &mut PSSALayerV2, mode: &str, group: usize, epsilon: f32) -> Result<f64, String> {
    model.disable_certified_inference();
    let at = Instant::now();
    if mode == "csr" || mode == "dual" {
        model.block.certified_memory_index =
            Some(CertifiedMemoryIndex::build(&model.memory, group)?);
        model.block.certified_memory_epsilon = Some(epsilon);
    }
    if mode == "cvp" || mode == "dual" {
        model.certified_vocabulary_index =
            Some(CertifiedVocabularyIndex::build(&model.unembed_w, 32)?);
    }
    Ok(at.elapsed().as_secs_f64())
}

fn greedy(logits: &[f32]) -> usize {
    (1..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}

fn batch(
    model: &mut PSSALayerV2,
    mode: &str,
    stream: &[usize],
    logits: &mut [f32],
) -> Result<(), String> {
    for (i, &token) in stream.iter().enumerate() {
        if i % 32 == 0 {
            model.reset_recurrent_state();
        }
        let id = if mode == "cvp" || mode == "dual" {
            model.forward_inference_certified_greedy(token)?.0
        } else {
            model.forward_inference(token, logits);
            greedy(logits)
        };
        black_box(id);
    }
    Ok(())
}

fn fixture(seed: u64, capacity: usize, vocab: usize, bank: &str) -> Result<PSSALayerV2, String> {
    let cfg = PSSAConfigV2 {
        d_vocab: vocab,
        d_latent: 64,
        d_state: 4,
        d_mem_key: 8,
        mem_capacity: capacity,
        chunk_len: 16,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut model = PSSALayerV2::new(cfg, seed);
    let text = crate::dataset::DatasetManager::SCIENCE_REFERENCE_CORPUS;
    let stream: Vec<_> = text.bytes().map(|b| b as usize + 1).collect();
    if bank == "trained" {
        for step in 0..32 {
            let offset = step * 16 % (stream.len() - 17);
            model.reset_recurrent_state();
            model.forward_train_chunk(
                &stream[offset..offset + 16],
                &stream[offset + 1..offset + 17],
            );
            model.zero_gradients();
            model.backward_chunk(16, 1.0);
            if !matches!(
                model.apply_adamw_with_grad_clip(0.003, 1.0),
                crate::pssa::GradientClipOutcome::Applied { .. }
            ) {
                return Err("sparse fixture training diverged".into());
            }
        }
    }
    let mut rng = SimpleRng::new(seed ^ 0x4241_4e4b);
    let mut logits = vec![0.0; vocab];
    if bank == "separated" {
        model.block.w_qx.data.fill(0.0);
        model.block.w_qh.data.fill(0.0);
        model.block.cfg.tau_mem = 0.05;
        for row in 0..vocab {
            for col in 0..64 {
                model.unembed_w.data[row * 64 + col] = (row / 32) as f32 * 0.03;
            }
        }
    }
    for i in 0..capacity {
        let (key, value) = if bank == "trained" {
            if i % 32 == 0 {
                model.reset_recurrent_state();
            }
            model.forward_inference(stream[i % stream.len()], &mut logits);
            (model.inf_q_pnc.clone(), model.inf_z_final.clone())
        } else {
            let mut key = vec![0.0; 8];
            if bank == "separated" {
                key[0] = if i < capacity / 16 { 0.0 } else { 0.85 };
            } else {
                let raw: Vec<_> = (0..8).map(|_| rng.gen_range_f32(-0.3, 0.3)).collect();
                HyperbolicEpisodicBankV2::diffeomorphic_project(&raw, &mut key);
            }
            (key, (0..64).map(|_| rng.gen_range_f32(-0.1, 0.1)).collect())
        };
        model.memory.insert(&key, &value);
    }
    Ok(model)
}

pub(super) fn run(seed: u64) -> Result<Value, String> {
    let mut rows = Vec::new();
    for capacity in [64, 512] {
        for vocab in [257, 2048] {
            for bank in ["trained", "diffuse", "separated"] {
                let mut model = fixture(seed, capacity, vocab, bank)?;
                let text: Vec<_> = crate::dataset::DatasetManager::SCIENCE_REFERENCE_CORPUS
                    .bytes()
                    .map(|b| b as usize + 1)
                    .collect();
                for queries in ["replay", "shifted"] {
                    let stream: Vec<_> = (0..257)
                        .map(|i| {
                            if queries == "replay" {
                                text[i % text.len()]
                            } else {
                                1 + (i * 67 + seed as usize) % 256
                            }
                        })
                        .collect();
                    let mut logits = vec![0.0; vocab];
                    install(&mut model, "dense", 4, 0.01)?;
                    let mut dense_ids = Vec::new();
                    let mut dense_logits = Vec::new();
                    for (i, &input) in stream[..256].iter().enumerate() {
                        if i % 32 == 0 {
                            model.reset_recurrent_state();
                        }
                        model.forward_inference(input, &mut logits);
                        dense_ids.push(greedy(&logits));
                        dense_logits.extend_from_slice(&logits);
                    }
                    for group in [4, 16] {
                        for epsilon in [0.0, 0.01, 0.05] {
                            for length in [32, 256] {
                                let mut samples: [Vec<f64>; 4] =
                                    std::array::from_fn(|_| Vec::new());
                                let mut builds: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
                                for round in 0..3 {
                                    for offset in 0..4 {
                                        let id = (round + offset + seed as usize) % 4;
                                        let mode = MODES[id];
                                        builds[id].push(install(&mut model, mode, group, epsilon)?);
                                        batch(&mut model, mode, &stream[..32], &mut logits)?;
                                        let at = Instant::now();
                                        batch(&mut model, mode, &stream[..length], &mut logits)?;
                                        samples[id].push(at.elapsed().as_secs_f64());
                                    }
                                }
                                let dense_seconds =
                                    super::super::median(samples[0].clone()).unwrap();
                                for (id, &mode) in MODES.iter().enumerate() {
                                    install(&mut model, mode, group, epsilon)?;
                                    batch(&mut model, mode, &stream[..32], &mut logits)?;
                                    let (
                                        mut memory_fallback,
                                        mut vocab_fallback,
                                        mut agree,
                                        mut scans,
                                    ) = (0, 0, 0, 0);
                                    let (mut mass, mut logit_error, mut ce_delta) =
                                        (0.0f64, 0.0f64, 0.0f64);
                                    let mut ref_values = vec![0.0; 64];
                                    let mut ref_weights = vec![0.0; capacity];
                                    for (i, &input) in stream[..length].iter().enumerate() {
                                        if i % 32 == 0 {
                                            model.reset_recurrent_state();
                                        }
                                        model.forward_inference(input, &mut logits);
                                        let stats = &model.block.last_sparse_read_stats;
                                        memory_fallback += usize::from(stats.exact_fallback);
                                        scans += stats.scanned_slots;
                                        if stats.certified {
                                            model.memory.retrieve_soft_into(
                                                &model.inf_q_pnc,
                                                model.block.cfg.tau_mem,
                                                &mut ref_values,
                                                &mut ref_weights,
                                            );
                                            let omitted: f64 = ref_weights
                                                .iter()
                                                .zip(&model.inf_mem_weights)
                                                .filter(|(_, w)| **w == 0.0)
                                                .map(|(&w, _)| w as f64)
                                                .sum();
                                            if omitted > stats.omitted_mass_bound as f64 + 1e-7 {
                                                return Err("invalid mass certificate".into());
                                            }
                                            mass = mass.max(omitted);
                                        }
                                        let selected = if mode == "cvp" || mode == "dual" {
                                            let (selected, vstats) =
                                                model.certified_greedy_current()?;
                                            if selected != greedy(&logits) {
                                                return Err("CVP exact-greedy mismatch".into());
                                            }
                                            vocab_fallback += usize::from(vstats.exact_fallback());
                                            selected
                                        } else {
                                            greedy(&logits)
                                        };
                                        agree += usize::from(selected == dense_ids[i]);
                                        let reference = &dense_logits[i * vocab..(i + 1) * vocab];
                                        logit_error = logit_error.max(
                                            logits
                                                .iter()
                                                .zip(reference)
                                                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                                                .fold(0.0, f64::max),
                                        );
                                        ce_delta += cross_entropy_f64(&logits, stream[i + 1])
                                            - cross_entropy_f64(reference, stream[i + 1]);
                                    }
                                    let seconds =
                                        super::super::median(samples[id].clone()).unwrap();
                                    let build = super::super::median(builds[id].clone()).unwrap();
                                    let index_bytes = model
                                        .block
                                        .certified_memory_index
                                        .as_ref()
                                        .map_or(0, CertifiedMemoryIndex::numeric_storage_bytes)
                                        + model.certified_vocabulary_index.as_ref().map_or(
                                            0,
                                            CertifiedVocabularyIndex::numeric_storage_bytes,
                                        );
                                    rows.push(json!({"seed":seed,"capacity":capacity,"vocab":vocab,"bank":bank,"queries":queries,
                            "group":group,"epsilon":epsilon,"length":length,"mode":mode,"seconds_samples":samples[id],
                            "build_seconds_samples":builds[id],"median_seconds":seconds,"median_build_seconds":build,
                            "paired_speedup":dense_seconds/seconds,"build_inclusive_speedup":dense_seconds/(seconds+build),
                            "break_even_tokens_estimate":if seconds<dense_seconds {Some(build/(dense_seconds-seconds)*length as f64)} else {None},
                            "index_numeric_bytes":index_bytes,"memory_fallback_rate":memory_fallback as f64/length as f64,
                            "vocab_full_scan_rate":vocab_fallback as f64/length as f64,"mean_scanned_slots":scans as f64/length as f64,
                            "greedy_agreement":agree as f64/length as f64,"max_omitted_mass":mass,"max_abs_logit_error":logit_error,
                            "mean_ce_delta":ce_delta/length as f64}));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(
        json!({"study":"sparse","seed":seed,"latent":64,"state":4,"key":8,
        "timing_rounds":3,"rows":rows}),
    )
}
