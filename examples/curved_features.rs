//! Frozen-context exporter for the opt-in curved-head experiment.
//! cargo run --release --example curved_features -- CORPUS OUTPUT_DIRECTORY SEED
use pssa::{
    linalg::SimpleRng,
    pssa::{GradientClipOutcome, PSSAConfigV2, PSSALayerV2},
};
use serde_json::json;
use std::{fs, io::Write, path::Path, time::Instant};

const WIDTH: usize = 32;
const UPDATES: usize = 1024;

fn schedule(step: usize) -> f32 {
    if step <= 32 {
        step as f32 / 32.0
    } else {
        let phase = (step - 32) as f64 / (UPDATES - 32) as f64;
        (0.1 + 0.45 * (1.0 + (std::f64::consts::PI * phase).cos())) as f32
    }
}

fn export(
    model: &mut PSSALayerV2,
    bytes: &[u8],
    out: &Path,
    split: &str,
) -> Result<serde_json::Value, String> {
    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut logits = vec![0.0; 257];
    let (mut ce, mut correct) = (0.0, 0);
    for doc in bytes.chunks(257).filter(|doc| doc.len() > 1) {
        model.reset_recurrent_state();
        for pair in doc.windows(2) {
            let target = pair[1] as usize + 1;
            model.forward_inference(pair[0] as usize + 1, &mut logits);
            if logits
                .iter()
                .chain(&model.inf_features)
                .any(|x| !x.is_finite())
            {
                return Err("non-finite feature or logit".into());
            }
            features.extend_from_slice(&model.inf_features);
            targets.push(target as u32);
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
            ce += max - logits[target] as f64 + sum.ln();
            let guess = (0..257)
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            correct += usize::from(guess == target);
        }
    }
    let write = |name: &str, bytes: &[u8]| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(name))
            .map_err(|e| e.to_string())?;
        file.write_all(bytes).map_err(|e| e.to_string())
    };
    let raw_features: Vec<_> = features.iter().flat_map(|x| x.to_le_bytes()).collect();
    let raw_targets: Vec<_> = targets.iter().flat_map(|x| x.to_le_bytes()).collect();
    write(&format!("{split}.f32"), &raw_features)?;
    write(&format!("{split}.u32"), &raw_targets)?;
    Ok(
        json!({"examples": targets.len(), "source_bytes": bytes.len(),
        "native_readout_ce": ce / targets.len() as f64,
        "native_readout_accuracy": correct as f64 / targets.len() as f64}),
    )
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: curved_features CORPUS OUTPUT_DIRECTORY SEED".into());
    }
    let seed: u64 = args[2].parse().map_err(|_| "invalid seed")?;
    let out = Path::new(&args[1]);
    if out.exists() {
        return Err("output directory already exists".into());
    }
    let corpus = fs::read(&args[0]).map_err(|e| e.to_string())?;
    if corpus.len() < 4096 {
        return Err("corpus must contain at least 4096 bytes".into());
    }
    let cut1 = corpus.len() * 8 / 10;
    let cut2 = corpus.len() * 9 / 10;
    let train = &corpus[..cut1];
    let dev = &corpus[cut1..cut2];
    let test = &corpus[cut2..];
    let cfg = PSSAConfigV2 {
        d_vocab: 257,
        d_latent: WIDTH,
        d_state: 4,
        d_mem_key: 8,
        mem_capacity: 32,
        chunk_len: 32,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut model = PSSALayerV2::new(cfg, seed);
    let mut rng = SimpleRng::new(seed ^ 0x4441_5441);
    let mut digest = 0xcbf29ce484222325u64;
    let mut start = 0;
    let at = Instant::now();
    for step in 1..=UPDATES {
        let offset = (step - 1) % 8 * 32;
        if offset == 0 {
            start = rng.next_u32() as usize % (train.len() - 256);
            model.reset_recurrent_state();
        }
        let input: Vec<_> = train[start + offset..start + offset + 32]
            .iter()
            .map(|&b| b as usize + 1)
            .collect();
        let target: Vec<_> = train[start + offset + 1..start + offset + 33]
            .iter()
            .map(|&b| b as usize + 1)
            .collect();
        for value in input.iter().chain(&target) {
            for byte in (*value as u64).to_le_bytes() {
                digest = (digest ^ byte as u64).wrapping_mul(0x100000001b3);
            }
        }
        let loss = model.forward_train_chunk(&input, &target);
        if !loss.is_finite() {
            return Err("backbone loss diverged".into());
        }
        model.zero_gradients();
        model.backward_chunk(32, 1.0);
        if !matches!(
            model.apply_adamw_with_grad_clip(0.01 * schedule(step), 1.0),
            GradientClipOutcome::Applied { .. }
        ) {
            return Err("backbone update failed".into());
        }
    }
    let backbone_seconds = at.elapsed().as_secs_f64();
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let exported = json!({
        "train": export(&mut model, &train[..train.len().min(32768)], out, "train")?,
        "dev": export(&mut model, &dev[..dev.len().min(8192)], out, "dev")?,
        "test": export(&mut model, &test[..test.len().min(8192)], out, "test")?,
    });
    let metadata = json!({"seed":seed,"feature_width": WIDTH,"vocab":257,
        "backbone_updates":UPDATES,"backbone_target_tokens":UPDATES*32,
        "backbone_peak_rate":0.01,"backbone_training_seconds":backbone_seconds,
        "backbone_stream_digest":format!("{digest:016x}"),
        "split_offsets":[0,cut1,cut2,corpus.len()],"features":exported,
        "context_policy":"disjoint contiguous 80/10/10 splits, reset every 256 targets; frozen empty memory bank",
        "binary_layout":"row-major little-endian f32 features and little-endian u32 byte+1 targets"});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("curved features: {error}");
        std::process::exit(1);
    }
}
