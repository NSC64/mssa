//! Opt-in CPU research runtime for learning without a reverse-time tape.
//!
//! `InterdiffusionTrainer` combines local analytic derivatives and Adam state
//! with streaming, cosine-projected eligibility and intermittent spectral probes.
//! `ZerothOrderTrainer` retains the original pure tensor-block two-probe optimizer
//! and original-weight commit replay, with no gradients or Adam moments.
//!
//! The private model is deliberately unavailable to legacy backward/checkpoint
//! APIs, which require optimizer arrays. This first prototype supports CPU and
//! one temporal loop; independently weighted depth blocks are supported.

use crate::linalg::SimpleRng;
use crate::pssa::{PSSAConfigV2, PSSALayerV2, ParamMatrix, ParamVector};
use std::f64::consts::PI;

/// Stable CE including high-confidence tails that `ln(1 + tail)` rounds away.
pub(crate) fn cross_entropy_f64(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let other: f64 = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target)
        .map(|(_, &x)| (x as f64 - max).exp())
        .sum();
    if logits[target] as f64 == max {
        other.ln_1p()
    } else {
        max - logits[target] as f64 + (other + (logits[target] as f64 - max).exp()).ln()
    }
}

#[path = "interdiffusion_adaptive.rs"]
mod adaptive;
pub use adaptive::{InterdiffusionConfig, InterdiffusionReport, InterdiffusionTrainer};

#[path = "interdiffusion_benchmark.rs"]
mod benchmark;

pub(crate) use benchmark::SEEDS as BENCHMARK_SEEDS;
pub(crate) use benchmark::report as benchmark_report;
pub(crate) use benchmark::{run as run_benchmark, worker as benchmark_worker};

pub struct ForwardModel {
    model: PSSALayerV2,
    logits: Vec<f32>,
    saved_state: Vec<f32>,
}

impl ForwardModel {
    pub fn new(cfg: PSSAConfigV2, seed: u64) -> Result<Self, String> {
        crate::checkpoint::validate_model_config(&cfg)?;
        let logits = vec![0.0; cfg.d_vocab];
        let saved_state = vec![0.0; cfg.depth * cfg.d_latent * cfg.d_state];
        Ok(Self {
            model: PSSALayerV2::new_forward_only(cfg, seed),
            logits,
            saved_state,
        })
    }

    pub fn config(&self) -> &PSSAConfigV2 {
        &self.model.cfg
    }

    pub fn parameter_count(&self) -> usize {
        self.model.parameter_count()
    }

    fn validate_tokens(&self, inputs: &[usize], targets: &[usize]) -> Result<(), String> {
        if inputs.is_empty() || inputs.len() != targets.len() {
            return Err("forward loss needs matching nonempty input/target slices".into());
        }
        if inputs
            .iter()
            .chain(targets)
            .any(|&id| id >= self.model.cfg.d_vocab)
        {
            return Err("forward loss token ID is outside the vocabulary".into());
        }
        Ok(())
    }

    /// Stream exact next-token cross entropy, advancing carry without writing
    /// memory. Length is independent of the historical chunk/tape capacity.
    pub fn loss(
        &mut self,
        inputs: &[usize],
        targets: &[usize],
        reset: bool,
    ) -> Result<f64, String> {
        self.validate_tokens(inputs, targets)?;
        if reset {
            self.model.reset_recurrent_state();
        }
        let mut total = 0.0;
        for (&input, &target) in inputs.iter().zip(targets) {
            self.model.try_forward_inference(input, &mut self.logits)?;
            if self.logits.iter().any(|x| !x.is_finite()) {
                return Err("non-finite forward-only logits".into());
            }
            total += cross_entropy_f64(&self.logits, target);
        }
        Ok(total / inputs.len() as f64)
    }

    /// Frozen scoring, including restoration on a non-finite-logit error.
    pub fn evaluate(&mut self, inputs: &[usize], targets: &[usize]) -> Result<f64, String> {
        self.validate_tokens(inputs, targets)?;
        self.save_state();
        let result = self.loss(inputs, targets, true);
        self.restore_state();
        result
    }

    fn save_state(&mut self) {
        self.model.copy_recurrent_state_to(&mut self.saved_state);
    }

    fn restore_state(&mut self) {
        self.model.copy_recurrent_state_from(&self.saved_state);
    }

    fn refresh_rates(&mut self) {
        self.model.block.refresh_ssm_rates();
        for block in &mut self.model.extra_blocks {
            block.refresh_ssm_rates();
        }
    }

    fn commit_memory(&mut self, loss: f64) {
        if loss <= 3.5 {
            return;
        }
        let step = self.model.step_counter;
        for block in std::iter::once(&mut self.model.block).chain(&mut self.model.extra_blocks) {
            block.memory.insert_protected_with_device(
                &block.inf_q_pnc,
                &block.inf_z_final,
                loss as f32,
                step,
                &crate::backend::Device::Cpu,
            );
        }
    }

    /// Owned numeric Vec capacity, excluding headers, allocator overhead and code.
    /// Counts the streaming logits and carry snapshot as well as model storage.
    pub fn numeric_storage_bytes(&self) -> usize {
        let mut floats = self.parameter_count()
            + self.logits.capacity()
            + self.saved_state.capacity()
            + self.model.inf_features.capacity()
            + self.model.inf_block_out.capacity()
            + self.model.residual_scales.capacity();
        let mut words = 0;
        for block in std::iter::once(&self.model.block).chain(&self.model.extra_blocks) {
            floats += [
                &block.h_persistent,
                &block.ssm_raw_snapshot,
                &block.ssm_rates,
                &block.memory.keys,
                &block.memory.values,
                &block.memory.norm_sq,
                &block.memory.confidence,
                &block.adapters[0].consolidated_up,
                &block.inf_x_norm,
                &block.inf_delta,
                &block.inf_b,
                &block.inf_c,
                &block.inf_y_ssm,
                &block.inf_q_euc,
                &block.inf_q_pnc,
                &block.inf_mem_weights,
                &block.inf_m_val,
                &block.inf_g_mem,
                &block.inf_m_proj,
                &block.inf_ad_act,
                &block.inf_ad_out,
                &block.inf_z_raw,
                &block.inf_mlp_act,
                &block.inf_mlp_out,
                &block.inf_z_final,
            ]
            .iter()
            .map(|v| v.capacity())
            .sum::<usize>();
            words += block.memory.last_seen_step.capacity();
        }
        floats * std::mem::size_of::<f32>() + words * std::mem::size_of::<usize>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionKind {
    Plain,
    Spectral,
}

#[derive(Clone, Copy, Debug)]
pub struct ZerothOrderConfig {
    pub kind: DirectionKind,
    pub learning_rate: f32,
    /// Absolute perturbation scale after direction normalization to unit RMS.
    pub epsilon: f32,
    /// Number of randomly selected cosine modes in the active tensor.
    pub modes: usize,
    /// Nonnegative frequency attenuation; zero samples the full band uniformly.
    pub smoothing: f32,
    /// L2 clipping of the estimated active-tensor gradient, without storing it.
    pub max_gradient_norm: f32,
}

impl Default for ZerothOrderConfig {
    fn default() -> Self {
        Self {
            kind: DirectionKind::Spectral,
            learning_rate: 0.05,
            epsilon: 0.001,
            modes: 8,
            smoothing: 4.0,
            max_gradient_norm: 1.0,
        }
    }
}

impl ZerothOrderConfig {
    fn validate(&self) -> Result<(), String> {
        if [self.learning_rate, self.epsilon, self.max_gradient_norm]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
            || !self.smoothing.is_finite()
            || self.smoothing < 0.0
            || !(1..=256).contains(&self.modes)
        {
            return Err("zeroth-order lr, epsilon and clip must be finite >0; smoothing finite >=0; modes 1..256".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepReport {
    pub loss: f64,
    pub positive_loss: f64,
    pub negative_loss: f64,
    pub directional_derivative: f64,
    pub estimated_gradient_norm: f64,
    pub parameter_tensor: usize,
    pub forward_evaluations: usize,
}

/// Block-coordinate SPSA. Every tensor is visited once per deterministic sweep,
/// identically for plain and spectral modes. Sparse frequencies are resampled.
pub struct ZerothOrderTrainer {
    pub forward: ForwardModel,
    config: ZerothOrderConfig,
    rng: SimpleRng,
    next_tensor: usize,
    original: Vec<f32>,
    direction: Vec<f32>,
    row_basis: Vec<f64>,
    col_basis: Vec<f64>,
}

impl ZerothOrderTrainer {
    pub fn new(cfg: PSSAConfigV2, seed: u64, config: ZerothOrderConfig) -> Result<Self, String> {
        config.validate()?;
        let mut forward = ForwardModel::new(cfg, seed)?;
        let mut largest = 0;
        let mut largest_row_basis = 0;
        let mut largest_col_basis = 0;
        for id in 0..tensor_count(&forward.model) {
            let t = tensor(&mut forward.model, id);
            largest = largest.max(t.data.len());
            let modes = config.modes.min(t.data.len());
            largest_row_basis = largest_row_basis.max(t.rows * modes);
            largest_col_basis = largest_col_basis.max(t.cols * modes);
        }
        Ok(Self {
            forward,
            config,
            rng: SimpleRng::new(seed ^ 0x494e_5445_5244_4946),
            next_tensor: 0,
            original: vec![0.0; largest],
            direction: vec![0.0; largest],
            row_basis: if config.kind == DirectionKind::Spectral {
                vec![0.0; largest_row_basis]
            } else {
                Vec::new()
            },
            col_basis: if config.kind == DirectionKind::Spectral {
                vec![0.0; largest_col_basis]
            } else {
                Vec::new()
            },
        })
    }

    pub fn numeric_storage_bytes(&self) -> usize {
        self.forward.numeric_storage_bytes()
            + (self.original.capacity() + self.direction.capacity()) * std::mem::size_of::<f32>()
            + (self.row_basis.capacity() + self.col_basis.capacity()) * std::mem::size_of::<f64>()
    }

    /// Uses pre-update, unperturbed carry/memory, matching the scalar trainer's
    /// detached-state ordering. Failed probes restore weights bit-for-bit and
    /// restore incoming carry, without publishing memory or an optimizer step.
    pub fn train_step(
        &mut self,
        inputs: &[usize],
        targets: &[usize],
        reset: bool,
        write_memory: bool,
    ) -> Result<StepReport, String> {
        self.forward.validate_tokens(inputs, targets)?;
        let next_step = self
            .forward
            .model
            .step_counter
            .checked_add(1)
            .ok_or("optimizer step overflow")?;
        self.forward.save_state();
        let id = self.next_tensor;
        let t = tensor(&mut self.forward.model, id);
        let n = t.data.len();
        self.original[..n].copy_from_slice(t.data);
        let seed = ((self.rng.next_u32() as u64) << 32) | self.rng.next_u32() as u64;
        fill_direction(
            &mut self.direction[..n],
            t.rows,
            t.cols,
            seed,
            self.config,
            &mut self.row_basis,
            &mut self.col_basis,
        );

        let result = (|| {
            self.set_probe(id, n, self.config.epsilon)?;
            let positive = self.forward.loss(inputs, targets, reset);
            self.forward.restore_state();
            let positive = positive?;
            self.set_probe(id, n, -self.config.epsilon)?;
            let negative = self.forward.loss(inputs, targets, reset);
            self.forward.restore_state();
            let negative = negative?;
            let slope = (positive - negative) / (2.0 * self.config.epsilon as f64);
            let norm = slope.abs() * (n as f64).sqrt();
            let clip = if norm > self.config.max_gradient_norm as f64 {
                self.config.max_gradient_norm as f64 / norm
            } else {
                1.0
            };
            let step = self.config.learning_rate as f64 * slope * clip;
            if !slope.is_finite()
                || self.original[..n]
                    .iter()
                    .zip(&self.direction[..n])
                    .any(|(&w, &u)| !((w as f64 - step * u as f64) as f32).is_finite())
            {
                return Err("non-finite zeroth-order update".into());
            }
            tensor(&mut self.forward.model, id)
                .data
                .copy_from_slice(&self.original[..n]);
            let loss = self.forward.loss(inputs, targets, reset)?;
            if write_memory {
                self.forward.commit_memory(loss);
            }
            let t = tensor(&mut self.forward.model, id);
            for ((w, &original), &u) in t
                .data
                .iter_mut()
                .zip(&self.original[..n])
                .zip(&self.direction[..n])
            {
                *w = (original as f64 - step * u as f64) as f32;
            }
            self.forward.model.step_counter = next_step;
            self.next_tensor = (id + 1) % tensor_count(&self.forward.model);
            Ok(StepReport {
                loss,
                positive_loss: positive,
                negative_loss: negative,
                directional_derivative: slope,
                estimated_gradient_norm: norm,
                parameter_tensor: id,
                forward_evaluations: 3,
            })
        })();
        if result.is_err() {
            tensor(&mut self.forward.model, id)
                .data
                .copy_from_slice(&self.original[..n]);
            self.forward.restore_state();
        }
        self.forward.refresh_rates();
        result
    }

    fn set_probe(&mut self, id: usize, n: usize, epsilon: f32) -> Result<(), String> {
        let t = tensor(&mut self.forward.model, id);
        for ((w, &original), &u) in t
            .data
            .iter_mut()
            .zip(&self.original[..n])
            .zip(&self.direction[..n])
        {
            *w = (original as f64 + epsilon as f64 * u as f64) as f32;
            if !w.is_finite() {
                return Err("non-finite zeroth-order perturbation".into());
            }
        }
        Ok(())
    }
}

fn normal(rng: &mut SimpleRng) -> f64 {
    let a = (rng.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0);
    let b = (rng.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0);
    (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
}

fn cosine_basis(out: &mut [f64], frequency: usize) {
    let n = out.len() as f64;
    let scale = (if frequency == 0 { 1.0 } else { 2.0 } / n).sqrt();
    for (i, value) in out.iter_mut().enumerate() {
        *value = scale * (PI * (i as f64 + 0.5) * frequency as f64 / n).cos();
    }
}

fn fill_direction(
    out: &mut [f32],
    rows: usize,
    cols: usize,
    seed: u64,
    config: ZerothOrderConfig,
    row_basis: &mut [f64],
    col_basis: &mut [f64],
) {
    let mut rng = SimpleRng::new(seed);
    if config.kind == DirectionKind::Plain {
        for x in out.iter_mut() {
            *x = normal(&mut rng) as f32;
        }
    } else {
        out.fill(0.0);
        let modes = config.modes.min(out.len());
        // Bounded stack storage: duplicate frequencies would change covariance.
        let mut frequencies = [usize::MAX; 256];
        for mode in 0..modes {
            let mut index = rng.next_u32() as usize % out.len();
            while frequencies[..mode].contains(&index) {
                index = rng.next_u32() as usize % out.len();
            }
            frequencies[mode] = index;
            let (r, c) = (index / cols, index % cols);
            let rb = &mut row_basis[mode * rows..(mode + 1) * rows];
            let cb = &mut col_basis[mode * cols..(mode + 1) * cols];
            cosine_basis(rb, r);
            cosine_basis(cb, c);
            let omega = (r as f64 / rows.max(2).saturating_sub(1) as f64).powi(2)
                + (c as f64 / cols.max(2).saturating_sub(1) as f64).powi(2);
            let amplitude = normal(&mut rng) / (1.0 + config.smoothing as f64 * omega).sqrt();
            for (i, &a) in rb.iter().enumerate() {
                for (j, &b) in cb.iter().enumerate() {
                    out[i * cols + j] += (amplitude * a * b) as f32;
                }
            }
        }
    }
    // Match per-direction perturbation size in both methods. This makes the
    // estimator normalized-direction SPSA, rather than unnormalized Gaussian MeZO.
    let rms = (out.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / out.len() as f64).sqrt();
    for x in out {
        *x = (*x as f64 / rms.max(f64::MIN_POSITIVE)) as f32;
    }
}

fn tensor_count(model: &PSSALayerV2) -> usize {
    2 + 14 * model.depth()
}

struct Tensor<'a> {
    rows: usize,
    cols: usize,
    data: &'a mut [f32],
    grad: &'a mut [f32],
}

fn matrix(p: &mut ParamMatrix) -> Tensor<'_> {
    Tensor {
        rows: p.rows,
        cols: p.cols,
        data: &mut p.data,
        grad: &mut p.grad,
    }
}

fn vector(p: &mut ParamVector) -> Tensor<'_> {
    Tensor {
        rows: p.data.len(),
        cols: 1,
        data: &mut p.data,
        grad: &mut p.grad,
    }
}

fn tensor(model: &mut PSSALayerV2, id: usize) -> Tensor<'_> {
    match id {
        0 => return matrix(&mut model.embed_w),
        1 => return matrix(&mut model.unembed_w),
        _ => (),
    }
    let block_id = (id - 2) / 14;
    let block = if block_id == 0 {
        &mut model.block
    } else {
        &mut model.extra_blocks[block_id - 1]
    };
    match (id - 2) % 14 {
        0 => vector(&mut block.norm_gamma),
        1 => vector(&mut block.norm_beta),
        2 => matrix(&mut block.a_mat),
        3 => matrix(&mut block.w_delta),
        4 => matrix(&mut block.w_b),
        5 => matrix(&mut block.w_c),
        6 => matrix(&mut block.w_qx),
        7 => matrix(&mut block.w_qh),
        8 => matrix(&mut block.w_gate),
        9 => matrix(&mut block.w_proj),
        10 => matrix(&mut block.adapters[0].down_proj),
        11 => matrix(&mut block.adapters[0].up_proj),
        12 => matrix(&mut block.mlp_w1),
        13 => matrix(&mut block.mlp_w2),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(depth: usize) -> PSSAConfigV2 {
        PSSAConfigV2 {
            depth,
            d_vocab: 7,
            d_latent: 4,
            d_state: 3,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 4,
            weight_decay: 0.0,
            ..Default::default()
        }
    }

    fn populate(model: &mut PSSALayerV2) {
        for block in std::iter::once(&mut model.block).chain(&mut model.extra_blocks) {
            block.memory.insert(&[0.1, -0.05], &[0.2, -0.1, 0.3, 0.4]);
        }
    }

    #[test]
    fn weight_only_initialization_and_streaming_loss_match_full_training_runtime() {
        for depth in [1, 2, 4] {
            let cfg = config(depth);
            let mut full = PSSALayerV2::new(cfg.clone(), 41);
            let mut light = ForwardModel::new(cfg, 41).unwrap();
            for id in 0..tensor_count(&full) {
                assert_eq!(
                    tensor(&mut full, id).data,
                    tensor(&mut light.model, id).data
                );
            }
            for t in light.model.adam_tensors() {
                assert!(t.grad.is_empty() && t.m.is_empty() && t.v.is_empty());
            }
            for block in std::iter::once(&light.model.block).chain(&light.model.extra_blocks) {
                assert!(block.tape.x_raw.is_empty() && block.tape.h_states.is_empty());
                assert!(block.tape.logits.is_empty() && block.bwd_g_logits.is_empty());
                assert!(block.ssm_scan_a.is_empty() && block.ssm_rate_derivatives.is_empty());
            }
            populate(&mut full);
            populate(&mut light.model);
            let inputs = [1, 2, 3, 4, 2, 1, 5, 3];
            let targets = [2, 3, 4, 2, 1, 5, 3, 2];
            let a = full.forward_train_chunk(&inputs[..4], &targets[..4]);
            let b = full.forward_train_chunk(&inputs[4..], &targets[4..]);
            let actual = light.loss(&inputs, &targets, true).unwrap();
            assert!((actual - (a as f64 + b as f64) / 2.0).abs() < 2e-6);
            let mut carry = vec![0.0; full.recurrent_state_len()];
            full.copy_recurrent_state_to(&mut carry);
            let mut light_carry = vec![0.0; carry.len()];
            light.model.copy_recurrent_state_to(&mut light_carry);
            assert_eq!(carry, light_carry);
            light.evaluate(&inputs, &targets).unwrap();
            light.model.copy_recurrent_state_to(&mut light_carry);
            assert_eq!(carry, light_carry);
        }
    }

    #[test]
    fn identical_state_finite_difference_agrees_with_backprop_directional_derivative() {
        for kind in [DirectionKind::Plain, DirectionKind::Spectral] {
            for id in [0, 1, 4, 7, 13, 15] {
                let cfg = config(2);
                let mut trainer = ZerothOrderTrainer::new(
                    cfg.clone(),
                    89,
                    ZerothOrderConfig {
                        kind,
                        learning_rate: 1e-6,
                        max_gradient_norm: 1e6,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut reference = PSSALayerV2::new(cfg, 89);
                populate(&mut trainer.forward.model);
                populate(&mut reference);
                trainer.forward.loss(&[1, 2], &[2, 3], true).unwrap();
                reference.forward_train_chunk(&[1, 2], &[2, 3]);
                trainer.next_tensor = id;
                let report = trainer
                    .train_step(&[3, 1, 4], &[1, 4, 2], false, false)
                    .unwrap();
                reference.forward_train_chunk(&[3, 1, 4], &[1, 4, 2]);
                reference.zero_gradients();
                reference.backward_chunk(3, 1.0);
                let t = tensor(&mut reference, id);
                let exact = t
                    .grad
                    .iter()
                    .zip(&trainer.direction)
                    .map(|(&g, &u)| g as f64 * u as f64)
                    .sum::<f64>();
                assert!(
                    (exact - report.directional_derivative).abs() < 2e-4 + exact.abs() * 0.01,
                    "{kind:?}/{id}: exact {exact}, probe {}",
                    report.directional_derivative
                );
                for (a, b) in std::iter::once(&reference.block)
                    .chain(&reference.extra_blocks)
                    .zip(
                        std::iter::once(&trainer.forward.model.block)
                            .chain(&trainer.forward.model.extra_blocks),
                    )
                {
                    assert_eq!(a.h_persistent, b.h_persistent);
                    assert_eq!(a.memory, b.memory);
                }
            }
        }
    }

    #[test]
    fn failed_probe_restores_exact_weights_incoming_carry_and_memory() {
        let mut trainer = ZerothOrderTrainer::new(
            config(2),
            91,
            ZerothOrderConfig {
                epsilon: f32::MAX,
                ..Default::default()
            },
        )
        .unwrap();
        populate(&mut trainer.forward.model);
        trainer.forward.loss(&[1, 2], &[2, 3], true).unwrap();
        let mut before = Vec::new();
        for id in 0..tensor_count(&trainer.forward.model) {
            before.push(tensor(&mut trainer.forward.model, id).data.to_vec());
        }
        let carries: Vec<_> = std::iter::once(&trainer.forward.model.block)
            .chain(&trainer.forward.model.extra_blocks)
            .map(|b| b.h_persistent.clone())
            .collect();
        let memories: Vec<_> = std::iter::once(&trainer.forward.model.block)
            .chain(&trainer.forward.model.extra_blocks)
            .map(|b| b.memory.clone())
            .collect();
        assert!(trainer.train_step(&[3, 1], &[1, 4], true, true).is_err());
        for (id, weights) in before.iter().enumerate() {
            assert_eq!(tensor(&mut trainer.forward.model, id).data, weights);
        }
        for ((block, carry), memory) in std::iter::once(&trainer.forward.model.block)
            .chain(&trainer.forward.model.extra_blocks)
            .zip(carries)
            .zip(memories)
        {
            assert_eq!(block.h_persistent, carry);
            assert_eq!(block.memory, memory);
        }
        assert_eq!(trainer.forward.model.step_counter, 0);
        assert_eq!(trainer.next_tensor, 0);
    }

    #[test]
    fn memory_write_uses_only_original_weight_replay_once() {
        let cfg = PSSAConfigV2 {
            d_vocab: 64,
            ..config(2)
        };
        let mut trainer = ZerothOrderTrainer::new(cfg.clone(), 41, Default::default()).unwrap();
        let mut reference = PSSALayerV2::new(cfg, 41);
        populate(&mut trainer.forward.model);
        populate(&mut reference);
        let loss = reference.forward_train_chunk(&[1, 2, 3], &[2, 3, 4]);
        assert!(loss > 3.5);
        reference.insert_training_memory(loss, 3);
        let report = trainer
            .train_step(&[1, 2, 3], &[2, 3, 4], true, true)
            .unwrap();
        assert_eq!(report.forward_evaluations, 3);
        for (a, b) in std::iter::once(&reference.block)
            .chain(&reference.extra_blocks)
            .zip(
                std::iter::once(&trainer.forward.model.block)
                    .chain(&trainer.forward.model.extra_blocks),
            )
        {
            assert_eq!(a.h_persistent, b.h_persistent);
            // Confidence contains an f32 loss; the streaming f64 sum can round
            // one ULP differently than the reference's f32 accumulation.
            assert_eq!(a.memory.count, b.memory.count);
            assert_eq!(a.memory.write_head, b.memory.write_head);
            assert_eq!(a.memory.keys, b.memory.keys);
            assert_eq!(a.memory.values, b.memory.values);
            assert_eq!(a.memory.last_seen_step, b.memory.last_seen_step);
        }
    }

    #[test]
    fn cosine_modes_are_orthonormal_and_both_directions_have_matched_rms() {
        for len in [1, 3, 8] {
            let basis: Vec<_> = (0..len)
                .map(|frequency| {
                    let mut row = vec![0.0; len];
                    cosine_basis(&mut row, frequency);
                    row
                })
                .collect();
            for (i, a) in basis.iter().enumerate() {
                for (j, b) in basis.iter().enumerate() {
                    let dot = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
                    assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
                }
            }
        }
        for kind in [DirectionKind::Plain, DirectionKind::Spectral] {
            let cfg = ZerothOrderConfig {
                kind,
                ..Default::default()
            };
            let mut direction = vec![0.0; 21];
            fill_direction(
                &mut direction,
                7,
                3,
                12,
                cfg,
                &mut [0.0; 56],
                &mut [0.0; 24],
            );
            assert!(
                (direction.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / 21.0 - 1.0).abs()
                    < 1e-6
            );
        }
    }
}
