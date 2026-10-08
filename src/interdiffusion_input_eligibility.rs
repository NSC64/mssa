//! Bounded forward tangents for input rows and affine RMSNorm. Up to eight
//! embedding rows share the chunk with two norm directions; incoming carry is
//! detached. Row-wise RMS preconditioning uses no dense embedding moments.

use super::*;

pub(super) struct InputEligibility {
    unique: Vec<usize>,
    pub(super) signals: Vec<usize>,
    pub(super) frequencies: Vec<usize>,
    pub(super) grad: Vec<f32>,
    phase: Vec<usize>,
    visits: Vec<usize>,
    mean_square: Vec<f32>,
    traces: Vec<f32>,
    x_dot: Vec<f32>,
    delta_dot: Vec<f32>,
    b_dot: Vec<f32>,
    c_dot: Vec<f32>,
    adapter_dot: Vec<f32>,
    out_dot: Vec<f32>,
    max_rows: usize,
    vocab: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PSSAConfigV2 {
        PSSAConfigV2 {
            d_vocab: 16,
            d_latent: 4,
            d_state: 3,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 12,
            weight_decay: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn row_budget_is_bounded_and_sparse_visits_cover_every_coordinate() {
        let cfg = fixture();
        let mut model = ForwardModel::new(cfg.clone(), 89).unwrap();
        let mut input = InputEligibility::new(&cfg, 89);
        let mut cosines = vec![0.0; 16];
        let mut row = vec![0.0; 4];
        for frequency in 0..4 {
            cosine_basis(&mut row, frequency);
            for (out, &x) in cosines[frequency * 4..(frequency + 1) * 4]
                .iter_mut()
                .zip(&row)
            {
                *out = x as f32;
            }
        }
        input.prepare(&(0..12).collect::<Vec<_>>(), 0, 4);
        assert_eq!(input.signals.len(), 10); // eight input rows plus two norms
        let bytes = input.bytes();
        let mut frequencies = Vec::new();
        for step in [0, 4, 8, 12] {
            input.prepare(&[3], step, 4);
            frequencies.push(input.frequencies[0]);
            input.apply(
                &mut model.model,
                InterdiffusionConfig::default(),
                1.0,
                &cosines,
            );
        }
        frequencies.sort_unstable();
        assert_eq!(frequencies, vec![0, 1, 2, 3]);
        assert_eq!(input.bytes(), bytes);
    }

    #[test]
    fn late_input_validation_failure_preserves_all_committed_optimizer_state() {
        let mut actual = InterdiffusionTrainer::new(fixture(), 89, Default::default()).unwrap();
        actual.train_step(&[1, 2], &[2, 3], true, false).unwrap();
        let vocab = actual.forward.config().d_vocab;
        actual
            .eligibility
            .as_mut()
            .unwrap()
            .input
            .as_mut()
            .unwrap()
            .visits[vocab] = usize::MAX;
        let weights: Vec<_> = (0..tensor_count(&actual.forward.model))
            .map(|id| tensor(&mut actual.forward.model, id).data.to_vec())
            .collect();
        let snapshot = |t: &InterdiffusionTrainer| {
            let e = t.eligibility.as_ref().unwrap();
            let input = e.input.as_ref().unwrap();
            let moments: Vec<_> = t
                .head_m
                .iter()
                .chain(&t.head_v)
                .chain(e.parts.iter().flat_map(|p| p.m.iter().chain(&p.v)))
                .chain(&e.delta_m)
                .chain(&e.delta_v)
                .chain(&e.b_m)
                .chain(&e.b_v)
                .chain(&input.mean_square)
                .copied()
                .collect();
            (
                moments,
                input.visits.clone(),
                e.steps,
                t.forward.model.step_counter,
            )
        };
        let before = snapshot(&actual);
        let carry = actual.forward.model.h_persistent.clone();
        let memory = actual.forward.model.memory.clone();
        assert!(
            actual
                .train_step(&[3, 1], &[1, 2], false, true)
                .unwrap_err()
                .contains("overflow")
        );
        assert_eq!(snapshot(&actual), before);
        for (id, expected) in weights.iter().enumerate() {
            assert_eq!(tensor(&mut actual.forward.model, id).data, expected);
        }
        assert_eq!(actual.forward.model.h_persistent, carry);
        assert_eq!(actual.forward.model.memory, memory);
        actual
            .eligibility
            .as_mut()
            .unwrap()
            .input
            .as_mut()
            .unwrap()
            .visits[vocab] = 1;
        actual.train_step(&[3, 1], &[1, 2], false, false).unwrap();
        assert_eq!(actual.forward.model.step_counter, 2);
    }
}

impl InputEligibility {
    pub fn new(c: &PSSAConfigV2, seed: u64) -> Self {
        let (v, d, s) = (c.d_vocab, c.d_latent, c.d_state);
        let max_rows = 8.min(c.chunk_len).min(v);
        let mut rng = SimpleRng::new(seed ^ 0x494e_5055_5445_4c49);
        Self {
            unique: Vec::with_capacity(c.chunk_len.min(v)),
            signals: Vec::with_capacity(max_rows + 2),
            frequencies: vec![0; max_rows + 2],
            grad: vec![0.0; max_rows + 2],
            phase: (0..v + 2).map(|_| rng.next_u32() as usize % d).collect(),
            visits: vec![0; v + 2],
            mean_square: vec![0.0; v + 2],
            traces: vec![0.0; (max_rows + 2) * d * s],
            x_dot: vec![0.0; d],
            delta_dot: vec![0.0; d],
            b_dot: vec![0.0; s],
            c_dot: vec![0.0; s],
            adapter_dot: vec![0.0; 16],
            out_dot: vec![0.0; d],
            max_rows,
            vocab: v,
        }
    }

    pub fn bytes(&self) -> usize {
        [
            &self.grad,
            &self.mean_square,
            &self.traces,
            &self.x_dot,
            &self.delta_dot,
            &self.b_dot,
            &self.c_dot,
            &self.adapter_dot,
            &self.out_dot,
        ]
        .iter()
        .map(|v| v.capacity())
        .sum::<usize>()
            * 4
            + [
                &self.unique,
                &self.signals,
                &self.frequencies,
                &self.phase,
                &self.visits,
            ]
            .iter()
            .map(|v| v.capacity())
            .sum::<usize>()
                * std::mem::size_of::<usize>()
    }

    pub fn prepare(&mut self, inputs: &[usize], step: usize, d: usize) {
        self.unique.clear();
        for &id in inputs {
            if !self.unique.contains(&id) {
                self.unique.push(id);
            }
        }
        self.unique.sort_unstable();
        self.signals.clear();
        for offset in 0..self.max_rows.min(self.unique.len()) {
            self.signals
                .push(self.unique[(step % self.unique.len() + offset) % self.unique.len()]);
        }
        self.signals.extend([self.vocab, self.vocab + 1]);
        for (i, &id) in self.signals.iter().enumerate() {
            // Per-row visits ensure coordinate coverage with periodic tokens.
            self.frequencies[i] = (self.visits[id] % d + self.phase[id]) % d;
        }
        self.grad.fill(0.0);
        self.traces.fill(0.0);
    }

    pub fn observe(
        &mut self,
        block: &PSSAContinuousBlockV2,
        embed: &ParamMatrix,
        input: usize,
        previous: &[f32],
        observations: &InferenceObservations,
        gz: &[f32],
        cosines: &[f32],
        delta_slope: &[f32],
        adapter_slope: &[f32],
    ) {
        let (d, s) = (block.cfg.d_latent, block.cfg.d_state);
        let raw = &embed.data[input * d..(input + 1) * d];
        let inv = 1.0 / (raw.iter().map(|x| x * x).sum::<f32>() / d as f32 + 1e-5).sqrt();
        let inv_cubed = inv.powi(3);
        let scale = 1.0 / (s as f32).sqrt();
        for index in 0..self.signals.len() {
            let id = self.signals[index];
            let frequency = self.frequencies[index];
            let u = &cosines[frequency * d..(frequency + 1) * d];
            let direct = id >= self.vocab || id == input;
            if direct {
                let mean_dot = if id < self.vocab {
                    crate::linalg::dot_slice(raw, u) / d as f32
                } else {
                    0.0
                };
                for i in 0..d {
                    self.x_dot[i] = if id == self.vocab {
                        raw[i] * inv * u[i]
                    } else if id == self.vocab + 1 {
                        u[i]
                    } else {
                        block.norm_gamma.data[i] * (inv * u[i] - raw[i] * inv_cubed * mean_dot)
                    };
                }
                block.w_delta.matvec(&self.x_dot, &mut self.delta_dot);
                for (g, &slope) in self.delta_dot.iter_mut().zip(delta_slope) {
                    *g *= slope;
                }
                block.w_b.matvec(&self.x_dot, &mut self.b_dot);
                block.w_c.matvec(&self.x_dot, &mut self.c_dot);
                block.adapters[0]
                    .down_proj
                    .matvec(&self.x_dot, &mut self.adapter_dot);
                for (g, &slope) in self.adapter_dot.iter_mut().zip(adapter_slope) {
                    *g *= slope;
                }
                block.adapters[0].total_up_matvec(&self.adapter_dot, &mut self.out_dot);
            }
            let trace = &mut self.traces[index * d * s..(index + 1) * d * s];
            for i in 0..d {
                let mut y_dot = 0.0;
                for j in 0..s {
                    let cell = i * s + j;
                    let a = observations.bar_a[cell];
                    trace[cell] *= a;
                    if direct {
                        trace[cell] += self.delta_dot[i]
                            * (block.ssm_rates[cell] * a * previous[cell]
                                + block.inf_b[j] * block.inf_x_norm[i])
                            + block.inf_delta[i]
                                * (self.b_dot[j] * block.inf_x_norm[i]
                                    + block.inf_b[j] * self.x_dot[i]);
                    }
                    y_dot += block.inf_c[j] * trace[cell];
                    if direct {
                        y_dot += self.c_dot[j] * block.h_persistent[cell];
                    }
                }
                let z_dot = y_dot * scale + if direct { self.out_dot[i] } else { 0.0 };
                self.grad[index] += gz[i] * z_dot;
            }
        }
    }

    pub fn norm_sq(&self) -> f64 {
        self.grad[..self.signals.len()]
            .iter()
            .map(|&g| (g as f64).powi(2))
            .sum()
    }

    fn candidate(
        &self,
        i: usize,
        c: &PSSAConfigV2,
        options: InterdiffusionConfig,
        clip: f32,
    ) -> (f32, f32) {
        let id = self.signals[i];
        let g = self.grad[i] * clip;
        let v = c.beta2 * self.mean_square[id] + (1.0 - c.beta2) * g * g;
        let bias = 1.0 - c.beta2.powf((self.visits[id] + 1) as f32);
        // Share variance, never momentum, across unrelated coordinates.
        let gain = if options.coordinate_rms_scaling {
            (c.d_latent as f32).sqrt()
        } else {
            1.0
        };
        let delta = -options.body_learning_rate * gain * g / ((v / bias).sqrt() + c.eps);
        (delta, v)
    }

    pub fn validate_update(
        &self,
        model: &PSSALayerV2,
        options: InterdiffusionConfig,
        clip: f32,
        cosines: &[f32],
    ) -> Result<(), String> {
        let d = model.cfg.d_latent;
        for (i, &id) in self.signals.iter().enumerate() {
            self.visits[id]
                .checked_add(1)
                .ok_or("input eligibility step overflow")?;
            let (delta, v) = self.candidate(i, &model.cfg, options, clip);
            let weights = if id == self.vocab {
                &model.block.norm_gamma.data[..]
            } else if id == self.vocab + 1 {
                &model.block.norm_beta.data[..]
            } else {
                &model.embed_w.data[id * d..(id + 1) * d]
            };
            let frequency = self.frequencies[i];
            if !delta.is_finite()
                || !v.is_finite()
                || weights
                    .iter()
                    .zip(&cosines[frequency * d..(frequency + 1) * d])
                    .any(|(&w, &u)| !(w + delta * u).is_finite())
            {
                return Err("non-finite input eligibility update".into());
            }
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        model: &mut PSSALayerV2,
        options: InterdiffusionConfig,
        clip: f32,
        cosines: &[f32],
    ) {
        let d = model.cfg.d_latent;
        for (i, &id) in self.signals.iter().enumerate() {
            let (delta, v) = self.candidate(i, &model.cfg, options, clip);
            let weights = if id == self.vocab {
                &mut model.block.norm_gamma.data[..]
            } else if id == self.vocab + 1 {
                &mut model.block.norm_beta.data[..]
            } else {
                &mut model.embed_w.data[id * d..(id + 1) * d]
            };
            let frequency = self.frequencies[i];
            for (w, &u) in weights
                .iter_mut()
                .zip(&cosines[frequency * d..(frequency + 1) * d])
            {
                *w += delta * u;
            }
            self.mean_square[id] = v;
            self.visits[id] += 1;
        }
    }
}
