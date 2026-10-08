//! Interdiffusion v2: streaming eligibility, local gradients, and sparse probes.
//! No reverse recurrence or activation-history tape is used. Local derivatives,
//! Adam moments, projected tangents, and probe scratch are counted in storage.

use super::*;

#[path = "interdiffusion_eligibility.rs"]
mod eligibility;
use eligibility::Eligibility;

#[derive(Clone, Copy, Debug)]
pub struct InterdiffusionConfig {
    pub head_learning_rate: f32,
    pub body_learning_rate: f32,
    /// Zero freezes the body (readout-only); otherwise probe at this cadence.
    /// Eligible recurrent/local body updates run every step, independently.
    pub body_every: usize,
    pub epsilon: f32,
    pub modes: usize,
    pub smoothing: f32,
    pub max_gradient_norm: f32,
    /// Projected forward eligibility on depth-one, empty-bank models; otherwise
    /// the generic spectral-probe body path remains available.
    pub eligibility: bool,
    /// Trace up to eight input rows and two norm directions on eligible models,
    /// replacing finite-difference probes with bounded forward tangents.
    pub input_eligibility: bool,
    /// Give a unit-L2 cosine update unit RMS per coordinate, matching the scale
    /// of dense coordinate-wise adaptive updates. Does not change trace bases.
    pub coordinate_rms_scaling: bool,
    /// Use clipped diagonal CE curvature on confident chunks (CE < 0.1).
    /// Ambiguous chunks retain local Adam; false preserves the Adam ablation.
    pub curvature_readout: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(depth: usize) -> PSSAConfigV2 {
        PSSAConfigV2 {
            depth,
            d_vocab: 7,
            d_latent: 4,
            d_state: 3,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 4,
            ..Default::default()
        }
    }

    fn populated(model: &mut PSSALayerV2) {
        for block in std::iter::once(&mut model.block).chain(&mut model.extra_blocks) {
            block.memory.insert(&[0.1, -0.05], &[0.2, -0.1, 0.3, 0.4]);
        }
    }

    #[test]
    fn local_readout_gradient_and_adam_match_backprop_reference_without_body_adjoints() {
        let cfg = fixture(2);
        let options = InterdiffusionConfig {
            body_every: 0,
            curvature_readout: false,
            max_gradient_norm: 1e6,
            ..Default::default()
        };
        let mut actual = InterdiffusionTrainer::new(cfg.clone(), 89, options).unwrap();
        let mut reference = PSSALayerV2::new(cfg.clone(), 89);
        populated(&mut actual.forward.model);
        populated(&mut reference);
        for step in 1..=3 {
            let (inputs, targets) = (&[1, 3, 2][..], &[3, 2, 4][..]);
            let report = actual
                .train_step(inputs, targets, step == 1, false)
                .unwrap();
            if step == 1 {
                reference.reset_recurrent_state();
            }
            reference.forward_train_chunk(inputs, targets);
            reference.zero_gradients();
            reference.backward_chunk(inputs.len(), 1.0);
            for (&a, &b) in actual.head_grad.iter().zip(&reference.unembed_w.grad) {
                assert!((a - b).abs() < 2e-6, "local {a}, reference {b}");
            }
            reference.unembed_w.step_adamw(
                options.head_learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.weight_decay,
                cfg.eps,
                step,
            );
            for (&a, &b) in actual
                .forward
                .model
                .unembed_w
                .data
                .iter()
                .zip(&reference.unembed_w.data)
            {
                assert!((a - b).abs() < 2e-5);
            }
            assert_eq!(report.forward_evaluations, 1);
            for id in 0..tensor_count(&reference) {
                if id != 1 {
                    assert_eq!(
                        tensor(&mut reference, id).data,
                        tensor(&mut actual.forward.model, id).data
                    );
                }
            }
        }
    }

    #[test]
    fn compact_embedding_probe_matches_backprop_and_preserves_untouched_rows_and_carry() {
        let cfg = fixture(2);
        let mut actual = InterdiffusionTrainer::new(cfg.clone(), 41, Default::default()).unwrap();
        let mut reference = PSSALayerV2::new(cfg, 41);
        populated(&mut actual.forward.model);
        populated(&mut reference);
        actual.forward.loss(&[2, 1], &[1, 3], true).unwrap();
        reference.forward_train_chunk(&[2, 1], &[1, 3]);
        let before = actual.forward.model.embed_w.data.clone();
        let report = actual
            .train_step(&[1, 3, 1], &[3, 1, 2], false, false)
            .unwrap();
        reference.forward_train_chunk(&[1, 3, 1], &[3, 1, 2]);
        reference.zero_gradients();
        reference.backward_chunk(3, 1.0);
        let d = reference.cfg.d_latent;
        let exact = actual
            .embedding_rows
            .iter()
            .enumerate()
            .map(|(index, &row)| {
                reference.embed_w.grad[row * d..(row + 1) * d]
                    .iter()
                    .zip(&actual.direction[index * d..(index + 1) * d])
                    .map(|(&g, &u)| g as f64 * u as f64)
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert!(
            (exact - report.body_directional_derivative.unwrap()).abs() < 2e-4 + exact.abs() * 0.01
        );
        for row in [0, 2, 4, 5, 6] {
            assert_eq!(
                &actual.forward.model.embed_w.data[row * d..(row + 1) * d],
                &before[row * d..(row + 1) * d]
            );
        }
        for (a, b) in std::iter::once(&reference.block)
            .chain(&reference.extra_blocks)
            .zip(
                std::iter::once(&actual.forward.model.block)
                    .chain(&actual.forward.model.extra_blocks),
            )
        {
            assert_eq!(a.h_persistent, b.h_persistent);
            assert_eq!(a.memory, b.memory);
        }
    }

    #[test]
    fn failed_body_probe_restores_readout_moments_weights_carry_and_memory() {
        let mut actual = InterdiffusionTrainer::new(
            fixture(2),
            41,
            InterdiffusionConfig {
                epsilon: f32::MAX,
                ..Default::default()
            },
        )
        .unwrap();
        populated(&mut actual.forward.model);
        actual.forward.loss(&[1, 2], &[2, 3], true).unwrap();
        let before: Vec<_> = (0..tensor_count(&actual.forward.model))
            .map(|id| tensor(&mut actual.forward.model, id).data.to_vec())
            .collect();
        let carry: Vec<_> = std::iter::once(&actual.forward.model.block)
            .chain(&actual.forward.model.extra_blocks)
            .map(|b| (b.h_persistent.clone(), b.memory.clone()))
            .collect();
        assert!(actual.train_step(&[1, 3], &[3, 2], true, true).is_err());
        for (id, weights) in before.iter().enumerate() {
            assert_eq!(tensor(&mut actual.forward.model, id).data, weights);
        }
        for (block, (state, memory)) in std::iter::once(&actual.forward.model.block)
            .chain(&actual.forward.model.extra_blocks)
            .zip(carry)
        {
            assert_eq!(block.h_persistent, state);
            assert_eq!(block.memory, memory);
        }
        assert!(
            actual
                .head_m
                .iter()
                .chain(&actual.head_v)
                .all(|&x| x == 0.0)
        );
        assert_eq!(actual.forward.model.step_counter, 0);
        assert_eq!(actual.next_tensor, 0);
    }

    #[test]
    fn cadence_and_activity_selection_do_not_probe_dead_routes() {
        let mut actual = InterdiffusionTrainer::new(
            fixture(1),
            41,
            InterdiffusionConfig {
                body_every: 4,
                eligibility: false,
                ..Default::default()
            },
        )
        .unwrap();
        actual.next_tensor = 8;
        assert_eq!(actual.select_body(), 13); // empty bank + zero adapter-up/MLP-up
        actual.next_tensor = 0;
        for expected in [3, 1, 1, 1, 3] {
            assert_eq!(
                actual
                    .train_step(&[1, 2], &[2, 3], true, false)
                    .unwrap()
                    .forward_evaluations,
                expected
            );
        }
    }

    #[test]
    fn readout_curvature_matches_finite_difference_and_bounds_the_update() {
        let mut actual = InterdiffusionTrainer::new(
            fixture(1),
            89,
            InterdiffusionConfig {
                body_every: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let (inputs, targets) = (&[1, 2, 3][..], &[2, 3, 4][..]);
        let base = actual.base_pass(inputs, targets, true).unwrap();
        for index in [0, 5, 13, 20] {
            let original = actual.forward.model.unembed_w.data[index];
            let eps = 0.05;
            actual.forward.model.unembed_w.data[index] = original + eps;
            let positive = actual.forward.evaluate(inputs, targets).unwrap();
            actual.forward.model.unembed_w.data[index] = original - eps;
            let negative = actual.forward.evaluate(inputs, targets).unwrap();
            actual.forward.model.unembed_w.data[index] = original;
            let exact = (positive + negative - 2.0 * base) / (eps as f64).powi(2);
            assert!(
                (actual.head_curvature[index] as f64 - exact).abs() < 2e-4 + exact.abs() * 0.02
            );
        }
        actual.use_curvature = true;
        actual.head_gradient().unwrap();
        actual.validate_head_update(1).unwrap();
        let before = actual.forward.model.unembed_w.data.clone();
        actual.forward.model.cfg.weight_decay = 0.0;
        actual.apply_head(1);
        assert!(
            before
                .iter()
                .zip(&actual.forward.model.unembed_w.data)
                .all(|(&a, &b)| (a - b).abs() <= actual.config.head_learning_rate + 1e-6)
        );
    }

    #[test]
    fn memory_commit_preserves_base_keys_and_activates_accounted_larger_probe_workspace() {
        let cfg = PSSAConfigV2 {
            d_vocab: 64,
            chunk_len: 2,
            weight_decay: 0.0,
            ..fixture(1)
        };
        let mut actual = InterdiffusionTrainer::new(
            cfg.clone(),
            41,
            InterdiffusionConfig {
                body_every: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let mut reference = ForwardModel::new(cfg, 41).unwrap();
        actual.forward.model.unembed_w.data.fill(0.0);
        reference.model.unembed_w.data.fill(0.0);
        let loss = reference.loss(&[1, 2], &[2, 3], true).unwrap();
        assert!(loss > 3.5);
        reference.commit_memory(loss);
        let first = actual.train_step(&[1, 2], &[2, 3], true, true).unwrap();
        assert!(first.eligibility_updates);
        assert_eq!(actual.forward.model.memory.count, 1);
        assert_eq!(actual.forward.model.memory, reference.model.memory);
        assert_eq!(
            actual.forward.model.h_persistent,
            reference.model.h_persistent
        );

        reference
            .model
            .unembed_w
            .data
            .clone_from(&actual.forward.model.unembed_w.data);
        reference.loss(&[2, 1], &[1, 3], false).unwrap();
        actual.next_tensor = 15; // MLP output matrix, larger than embedding-row scratch
        let before = actual.numeric_storage_bytes();
        let workspace_bytes = |t: &InterdiffusionTrainer| {
            (t.original.capacity() + t.direction.capacity()) * 4
                + (t.row_basis.capacity() + t.col_basis.capacity()) * 8
        };
        let old_workspace = workspace_bytes(&actual);
        let second = actual.train_step(&[2, 1], &[1, 3], false, false).unwrap();
        assert!(!second.eligibility_updates);
        assert_eq!(second.body_tensor, Some(15));
        assert_eq!(second.forward_evaluations, 3);
        assert!(actual.numeric_storage_bytes() > before);
        assert_eq!(
            actual.numeric_storage_bytes() - before,
            workspace_bytes(&actual) - old_workspace
        );
        assert_eq!(actual.forward.model.memory, reference.model.memory);
        assert_eq!(
            actual.forward.model.h_persistent,
            reference.model.h_persistent
        );
    }
}

impl Default for InterdiffusionConfig {
    fn default() -> Self {
        Self {
            head_learning_rate: 0.01,
            body_learning_rate: 0.01,
            body_every: 16,
            epsilon: 0.001,
            modes: 8,
            smoothing: 0.0,
            max_gradient_norm: 1.0,
            eligibility: true,
            input_eligibility: true,
            coordinate_rms_scaling: true,
            curvature_readout: true,
        }
    }
}

impl InterdiffusionConfig {
    fn validate(&self) -> Result<(), String> {
        ZerothOrderConfig {
            learning_rate: self.head_learning_rate,
            epsilon: self.epsilon,
            modes: self.modes,
            smoothing: self.smoothing,
            max_gradient_norm: self.max_gradient_norm,
            ..Default::default()
        }
        .validate()?;
        if !self.body_learning_rate.is_finite() || self.body_learning_rate < 0.0 {
            return Err("body learning rate must be finite and nonnegative".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InterdiffusionReport {
    pub loss: f64,
    pub head_gradient_norm: f64,
    pub body_tensor: Option<usize>,
    pub body_directional_derivative: Option<f64>,
    pub forward_evaluations: usize,
    pub eligibility_updates: bool,
}

pub struct InterdiffusionTrainer {
    pub forward: ForwardModel,
    config: InterdiffusionConfig,
    rng: SimpleRng,
    next_tensor: usize,
    head_grad: Vec<f32>,
    head_m: Vec<f32>,
    head_v: Vec<f32>,
    head_curvature: Vec<f32>,
    use_curvature: bool,
    errors: Vec<f32>,
    terminal_state: Vec<f32>,
    terminal_queries: Vec<f32>,
    terminal_values: Vec<f32>,
    embedding_rows: Vec<usize>,
    original: Vec<f32>,
    direction: Vec<f32>,
    row_basis: Vec<f64>,
    col_basis: Vec<f64>,
    eligibility: Option<Eligibility>,
}

impl InterdiffusionTrainer {
    /// Change rates for a caller-owned schedule without resetting optimizer state.
    pub fn set_learning_rates(&mut self, head: f32, body: f32) -> Result<(), String> {
        let mut options = self.config;
        options.head_learning_rate = head;
        options.body_learning_rate = body;
        options.validate()?;
        self.config = options;
        Ok(())
    }

    pub fn new(cfg: PSSAConfigV2, seed: u64, config: InterdiffusionConfig) -> Result<Self, String> {
        config.validate()?;
        let (l, d, v, k, depth) = (
            cfg.chunk_len,
            cfg.d_latent,
            cfg.d_vocab,
            cfg.d_mem_key,
            cfg.depth,
        );
        let mut forward = ForwardModel::new(cfg, seed)?;
        let eligibility = (config.eligibility && config.body_every > 0 && depth == 1)
            .then(|| Eligibility::new(forward.config(), seed, config.input_eligibility));
        let mut largest = 0;
        let (mut largest_rows, mut largest_cols) = (0, 0);
        if config.body_every > 0 {
            for id in 0..tensor_count(&forward.model) {
                if id == 1 {
                    continue;
                }
                if eligibility.is_some() && config.input_eligibility {
                    continue; // occupied-bank probe scratch grows on demand
                }
                if eligibility.is_some() && id != 0 && id != 2 && id != 3 {
                    continue;
                }
                let t = tensor(&mut forward.model, id);
                let rows = if id == 0 { l.min(v) } else { t.rows };
                let n = rows * t.cols;
                let modes = config.modes.min(n);
                largest = largest.max(n);
                largest_rows = largest_rows.max(rows * modes);
                largest_cols = largest_cols.max(t.cols * modes);
            }
        }
        Ok(Self {
            terminal_state: vec![0.0; forward.saved_state.len()],
            forward,
            config,
            rng: SimpleRng::new(seed ^ 0x494e_5445_5244_4946),
            next_tensor: 0,
            head_grad: vec![0.0; v * d],
            head_m: vec![0.0; v * d],
            head_v: vec![0.0; v * d],
            head_curvature: vec![0.0; if config.curvature_readout { v * d } else { 0 }],
            use_curvature: false,
            errors: vec![0.0; v],
            terminal_queries: vec![0.0; depth * k],
            terminal_values: vec![0.0; depth * d],
            embedding_rows: Vec::with_capacity(l.min(v)),
            original: vec![0.0; largest],
            direction: vec![0.0; largest],
            row_basis: vec![0.0; largest_rows],
            col_basis: vec![0.0; largest_cols],
            eligibility,
        })
    }

    pub fn numeric_storage_bytes(&self) -> usize {
        self.forward.numeric_storage_bytes()
            + [
                &self.head_grad,
                &self.head_m,
                &self.head_v,
                &self.head_curvature,
                &self.errors,
                &self.terminal_state,
                &self.terminal_queries,
                &self.terminal_values,
                &self.original,
                &self.direction,
            ]
            .iter()
            .map(|x| x.capacity())
            .sum::<usize>()
                * 4
            + (self.row_basis.capacity() + self.col_basis.capacity()) * 8
            + self.embedding_rows.capacity() * std::mem::size_of::<usize>()
            + self.eligibility.as_ref().map_or(0, Eligibility::bytes)
    }

    /// One base pass produces local CE gradients and eligible forward tangents.
    /// Fallback/ablation body probes reuse that pass's original terminal state.
    /// Every published carry/key/value is from the unperturbed, pre-update model.
    pub fn train_step(
        &mut self,
        inputs: &[usize],
        targets: &[usize],
        reset: bool,
        write_memory: bool,
    ) -> Result<InterdiffusionReport, String> {
        self.forward.validate_tokens(inputs, targets)?;
        if inputs.len() > self.forward.config().chunk_len {
            return Err(
                "Interdiffusion update exceeds its bounded probe workspace; increase chunk_len"
                    .into(),
            );
        }
        let step = self
            .forward
            .model
            .step_counter
            .checked_add(1)
            .ok_or("optimizer step overflow")?;
        self.forward.save_state();
        let use_eligibility = self.eligibility.is_some() && self.forward.model.memory.count == 0;
        if use_eligibility {
            self.eligibility
                .as_mut()
                .unwrap()
                .prepare(&self.forward.model.block, inputs);
        }
        let mut active = None;
        let result = (|| {
            let loss = self.base_pass(inputs, targets, reset)?;
            self.use_curvature = self.config.curvature_readout && loss < 0.1;
            self.forward
                .model
                .copy_recurrent_state_to(&mut self.terminal_state);
            for (id, block) in std::iter::once(&self.forward.model.block)
                .chain(&self.forward.model.extra_blocks)
                .enumerate()
            {
                let (k, d) = (block.cfg.d_mem_key, block.cfg.d_latent);
                self.terminal_queries[id * k..(id + 1) * k].copy_from_slice(&block.inf_q_pnc);
                self.terminal_values[id * d..(id + 1) * d].copy_from_slice(&block.inf_z_final);
            }
            let head_norm = self.head_gradient()?;
            let mut slope = None;
            let mut body_step = 0.0;
            let mut n = 0;
            if self.config.body_every > 0
                && !(use_eligibility && self.config.input_eligibility)
                && (step - 1) % self.config.body_every == 0
            {
                let id = self.select_body();
                active = Some(id);
                let (rows, cols) = self.prepare_body(id, inputs);
                n = rows * cols;
                let seed = ((self.rng.next_u32() as u64) << 32) | self.rng.next_u32() as u64;
                fill_direction(
                    &mut self.direction[..n],
                    rows,
                    cols,
                    seed,
                    ZerothOrderConfig {
                        kind: DirectionKind::Spectral,
                        learning_rate: self.config.head_learning_rate,
                        epsilon: self.config.epsilon,
                        modes: self.config.modes,
                        smoothing: self.config.smoothing,
                        max_gradient_norm: self.config.max_gradient_norm,
                    },
                    &mut self.row_basis,
                    &mut self.col_basis,
                );
                self.forward.restore_state();
                self.set_body(id, n, self.config.epsilon as f64)?;
                let positive = self.forward.loss(inputs, targets, reset)?;
                self.forward.restore_state();
                self.set_body(id, n, -(self.config.epsilon as f64))?;
                let negative = self.forward.loss(inputs, targets, reset)?;
                let derivative = (positive - negative) / (2.0 * self.config.epsilon as f64);
                let norm = derivative.abs() * (n as f64).sqrt();
                let clip =
                    (self.config.max_gradient_norm as f64 / norm.max(f64::MIN_POSITIVE)).min(1.0);
                body_step = -self.config.body_learning_rate as f64 * derivative * clip;
                if !derivative.is_finite()
                    || self.original[..n]
                        .iter()
                        .zip(&self.direction[..n])
                        .any(|(&w, &u)| !((w as f64 + body_step * u as f64) as f32).is_finite())
                {
                    return Err("non-finite Interdiffusion body update".into());
                }
                slope = Some(derivative);
                self.restore_body(id, n);
            }
            self.validate_head_update(step)?;
            if use_eligibility {
                self.eligibility
                    .as_ref()
                    .unwrap()
                    .validate_update(&mut self.forward.model, self.config)?;
            }
            if write_memory && !(loss as f32).is_finite() {
                return Err("Interdiffusion loss exceeds memory-confidence precision".into());
            }
            self.forward
                .model
                .copy_recurrent_state_from(&self.terminal_state);
            if write_memory {
                for (id, block) in std::iter::once(&mut self.forward.model.block)
                    .chain(&mut self.forward.model.extra_blocks)
                    .enumerate()
                {
                    let (k, d) = (block.cfg.d_mem_key, block.cfg.d_latent);
                    block
                        .inf_q_pnc
                        .copy_from_slice(&self.terminal_queries[id * k..(id + 1) * k]);
                    block
                        .inf_z_final
                        .copy_from_slice(&self.terminal_values[id * d..(id + 1) * d]);
                }
                self.forward.commit_memory(loss);
            }
            self.apply_head(step);
            if use_eligibility {
                self.eligibility
                    .as_mut()
                    .unwrap()
                    .apply(&mut self.forward.model, self.config);
            }
            if let Some(id) = active {
                self.set_body(id, n, body_step)?;
                self.next_tensor = (id + 1) % tensor_count(&self.forward.model);
            }
            self.forward.model.step_counter = step;
            Ok(InterdiffusionReport {
                loss,
                head_gradient_norm: head_norm,
                body_tensor: active,
                body_directional_derivative: slope,
                forward_evaluations: if active.is_some() { 3 } else { 1 },
                eligibility_updates: use_eligibility,
            })
        })();
        if result.is_err() {
            if let Some(id) = active {
                let n = if id == 0 {
                    self.embedding_rows.len() * self.forward.config().d_latent
                } else {
                    tensor(&mut self.forward.model, id).data.len()
                };
                self.restore_body(id, n);
            }
            self.forward.restore_state();
        }
        self.forward.refresh_rates();
        result
    }

    fn base_pass(
        &mut self,
        inputs: &[usize],
        targets: &[usize],
        reset: bool,
    ) -> Result<f64, String> {
        if reset {
            self.forward.model.reset_recurrent_state();
        }
        let d = self.forward.config().d_latent;
        let scale = 1.0 / (inputs.len() as f64 * (d as f64).sqrt());
        let mut loss = 0.0;
        self.head_grad.fill(0.0);
        self.head_curvature.fill(0.0);
        for (&input, &target) in inputs.iter().zip(targets) {
            let use_eligibility =
                self.eligibility.is_some() && self.forward.model.memory.count == 0;
            if use_eligibility {
                let eligibility = self.eligibility.as_mut().unwrap();
                eligibility.before_token(&self.forward.model.block);
                self.forward.model.try_forward_inference_observed(
                    input,
                    &mut self.forward.logits,
                    Some(&mut eligibility.observations),
                )?;
            } else {
                self.forward
                    .model
                    .try_forward_inference(input, &mut self.forward.logits)?;
            }
            if self.forward.logits.iter().any(|x| !x.is_finite()) {
                return Err("non-finite Interdiffusion logits".into());
            }
            let max = self
                .forward
                .logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max) as f64;
            let other = self
                .forward
                .logits
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != target)
                .map(|(_, &x)| (x as f64 - max).exp())
                .sum::<f64>();
            let sum = other + (self.forward.logits[target] as f64 - max).exp();
            loss += cross_entropy_f64(&self.forward.logits, target);
            for (r, &logit) in self.forward.logits.iter().enumerate() {
                let error = if r == target {
                    -other / sum
                } else {
                    (logit as f64 - max).exp() / sum
                };
                self.errors[r] = (error * scale) as f32;
                if self.config.curvature_readout {
                    let probability = (logit as f64 - max).exp() / sum;
                    let variance = probability
                        * if r == target {
                            other / sum
                        } else {
                            1.0 - probability
                        };
                    let curvature = (variance / (inputs.len() * d) as f64) as f32;
                    for (hessian, &h) in self.head_curvature[r * d..(r + 1) * d]
                        .iter_mut()
                        .zip(&self.forward.model.inf_features)
                    {
                        *hessian += curvature * h * h;
                    }
                }
                let e = self.errors[r];
                if e != 0.0 {
                    for (g, &h) in self.head_grad[r * d..(r + 1) * d]
                        .iter_mut()
                        .zip(&self.forward.model.inf_features)
                    {
                        *g += e * h;
                    }
                }
            }
            if use_eligibility {
                self.eligibility.as_mut().unwrap().observe(
                    &self.forward.model.block,
                    &self.forward.model.unembed_w,
                    &self.errors,
                    &self.forward.model.embed_w,
                    input,
                );
            }
        }
        Ok(loss / inputs.len() as f64)
    }

    fn head_gradient(&mut self) -> Result<f64, String> {
        let norm = self
            .head_grad
            .iter()
            .map(|&g| (g as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return Err("non-finite Interdiffusion readout gradient".into());
        }
        let clip = (self.config.max_gradient_norm as f64 / norm.max(f64::MIN_POSITIVE)).min(1.0);
        for g in &mut self.head_grad {
            *g = (*g as f64 * clip) as f32;
        }
        Ok(norm)
    }

    fn head_candidate(&self, index: usize, bias: (f32, f32)) -> (f32, f32, f32) {
        let c = self.forward.config();
        let g = self.head_grad[index];
        let m = c.beta1 * self.head_m[index] + (1.0 - c.beta1) * g;
        let v = c.beta2 * self.head_v[index] + (1.0 - c.beta2) * g * g;
        if self.use_curvature {
            let curvature = self.head_curvature[index];
            if !curvature.is_finite() || curvature < 0.0 {
                return (f32::NAN, 0.0, 0.0);
            }
            // A diagonal Newton direction with a per-coordinate trust radius;
            // no inverse softmax matrix or cross-token tape is constructed.
            let direction = (g / curvature.max(f32::MIN_POSITIVE)).clamp(-1.0, 1.0);
            let original = self.forward.model.unembed_w.data[index];
            return (
                original - self.config.head_learning_rate * (direction + c.weight_decay * original),
                m,
                v,
            );
        }
        let mh = m / bias.0;
        let vh = v / bias.1;
        let original = self.forward.model.unembed_w.data[index];
        let w = original
            - self.config.head_learning_rate * c.weight_decay * original
            - self.config.head_learning_rate * mh / (vh.sqrt() + c.eps);
        (w, m, v)
    }

    fn validate_head_update(&self, step: usize) -> Result<(), String> {
        let c = self.forward.config();
        let bias = (
            1.0 - c.beta1.powf(step as f32),
            1.0 - c.beta2.powf(step as f32),
        );
        for i in 0..self.head_grad.len() {
            let (w, m, v) = self.head_candidate(i, bias);
            if !w.is_finite() || !m.is_finite() || !v.is_finite() {
                return Err("non-finite Interdiffusion readout update".into());
            }
        }
        Ok(())
    }

    fn apply_head(&mut self, step: usize) {
        let c = self.forward.config();
        let bias = (
            1.0 - c.beta1.powf(step as f32),
            1.0 - c.beta2.powf(step as f32),
        );
        for i in 0..self.head_grad.len() {
            let (w, m, v) = self.head_candidate(i, bias);
            self.forward.model.unembed_w.data[i] = w;
            self.head_m[i] = m;
            self.head_v[i] = v;
        }
    }

    fn select_body(&self) -> usize {
        let count = tensor_count(&self.forward.model);
        for offset in 0..count {
            let id = (self.next_tensor + offset) % count;
            if id == 1 {
                continue;
            }
            if id >= 2 {
                let block_id = (id - 2) / 14;
                let block = if block_id == 0 {
                    &self.forward.model.block
                } else {
                    &self.forward.model.extra_blocks[block_id - 1]
                };
                let member = (id - 2) % 14;
                if self.eligibility.is_some()
                    && block.memory.count == 0
                    && ![0, 1].contains(&member)
                {
                    continue;
                }
                if (6..=9).contains(&member) && block.memory.count == 0 {
                    continue;
                }
                if member == 10
                    && block.adapters[0]
                        .up_proj
                        .data
                        .iter()
                        .zip(&block.adapters[0].consolidated_up)
                        .all(|(&a, &b)| a + b == 0.0)
                {
                    continue;
                }
                if member == 12 && block.mlp_w2.data.iter().all(|&x| x == 0.0) {
                    continue;
                }
            }
            return id;
        }
        unreachable!("input embeddings always provide an active body tensor")
    }

    fn prepare_body(&mut self, id: usize, inputs: &[usize]) -> (usize, usize) {
        if id == 0 {
            self.embedding_rows.clear();
            for &input in inputs {
                if !self.embedding_rows.contains(&input) {
                    self.embedding_rows.push(input);
                }
            }
            self.embedding_rows.sort_unstable();
            let d = self.forward.config().d_latent;
            self.ensure_body_workspace(self.embedding_rows.len(), d);
            for (index, &row) in self.embedding_rows.iter().enumerate() {
                self.original[index * d..(index + 1) * d]
                    .copy_from_slice(&self.forward.model.embed_w.data[row * d..(row + 1) * d]);
            }
            (self.embedding_rows.len(), d)
        } else {
            let (rows, cols) = {
                let t = tensor(&mut self.forward.model, id);
                (t.rows, t.cols)
            };
            self.ensure_body_workspace(rows, cols);
            self.original[..rows * cols].copy_from_slice(tensor(&mut self.forward.model, id).data);
            (rows, cols)
        }
    }

    fn ensure_body_workspace(&mut self, rows: usize, cols: usize) {
        let n = rows * cols;
        let modes = self.config.modes.min(n);
        // The empty-bank eligibility route only probes embeddings/norms. If
        // memory becomes occupied, the generic fallback grows bounded scratch
        // for its newly reachable tensor; actual capacities remain accounted.
        if self.original.len() < n {
            self.original.resize(n, 0.0);
        }
        if self.direction.len() < n {
            self.direction.resize(n, 0.0);
        }
        if self.row_basis.len() < rows * modes {
            self.row_basis.resize(rows * modes, 0.0);
        }
        if self.col_basis.len() < cols * modes {
            self.col_basis.resize(cols * modes, 0.0);
        }
    }

    fn set_body(&mut self, id: usize, n: usize, delta: f64) -> Result<(), String> {
        let d = self.forward.config().d_latent;
        let t = tensor(&mut self.forward.model, id);
        for i in 0..n {
            let index = if id == 0 {
                self.embedding_rows[i / d] * d + i % d
            } else {
                i
            };
            let value = (self.original[i] as f64 + delta * self.direction[i] as f64) as f32;
            if !value.is_finite() {
                return Err("non-finite Interdiffusion perturbation".into());
            }
            t.data[index] = value;
        }
        Ok(())
    }

    fn restore_body(&mut self, id: usize, n: usize) {
        let d = self.forward.config().d_latent;
        let t = tensor(&mut self.forward.model, id);
        if id == 0 {
            for (index, &row) in self.embedding_rows.iter().enumerate() {
                t.data[row * d..(row + 1) * d]
                    .copy_from_slice(&self.original[index * d..(index + 1) * d]);
            }
        } else {
            t.data.copy_from_slice(&self.original[..n]);
        }
    }
}
