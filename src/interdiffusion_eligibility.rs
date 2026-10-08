//! Streaming, projected forward eligibility for a depth-one empty-bank SSM.
//! A has independent cell traces. Delta/B use one cosine coordinate per row,
//! giving independent row-wise directional signals instead of one global slope.
//! Readout/MLP/adapter derivatives are local to the current token; no activation
//! history is traversed backwards and no reverse-time tape is allocated.

use super::*;
use crate::linalg::sigmoid;
use crate::pssa::{InferenceObservations, PSSAContinuousBlockV2};

#[path = "interdiffusion_input_eligibility.rs"]
mod input_eligibility;
use input_eligibility::InputEligibility;

struct LocalAdam {
    tensor: usize,
    grad: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl LocalAdam {
    fn new(tensor: usize, n: usize) -> Self {
        Self {
            tensor,
            grad: vec![0.0; n],
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
}

pub(super) struct Eligibility {
    pub observations: InferenceObservations,
    parts: Vec<LocalAdam>,
    cosines: Vec<f32>,
    phase: Vec<usize>,
    previous_h: Vec<f32>,
    rate_derivative: Vec<f32>,
    ea: Vec<f32>,
    ed: Vec<f32>,
    eb: Vec<f32>,
    delta_dot: Vec<f32>,
    delta_slope: Vec<f32>,
    adapter_slope: Vec<f32>,
    b_dot: Vec<f32>,
    delta_grad: Vec<f32>,
    b_grad: Vec<f32>,
    delta_m: Vec<f32>,
    delta_v: Vec<f32>,
    b_m: Vec<f32>,
    b_v: Vec<f32>,
    gz: Vec<f32>,
    gz_raw: Vec<f32>,
    mlp_grad: Vec<f32>,
    adapter_grad: Vec<f32>,
    c_grad: Vec<f32>,
    steps: usize,
    input: Option<InputEligibility>,
}

impl Eligibility {
    pub fn new(c: &PSSAConfigV2, seed: u64, input_eligibility: bool) -> Self {
        let (d, s, rank) = (c.d_latent, c.d_state, 16);
        let mut cosines = vec![0.0; d * d];
        let mut row = vec![0.0; d];
        for frequency in 0..d {
            cosine_basis(&mut row, frequency);
            for (out, &value) in cosines[frequency * d..(frequency + 1) * d]
                .iter_mut()
                .zip(&row)
            {
                *out = value as f32;
            }
        }
        let mut rng = SimpleRng::new(seed ^ 0x454c_4947_4942_4c45);
        Self {
            observations: InferenceObservations {
                bar_a: vec![0.0; d * s],
                delta_raw: vec![0.0; d],
                adapter_raw: vec![0.0; rank],
                mlp_raw: vec![0.0; 2 * d],
            },
            parts: [
                (4, d * s),
                (7, s * d),
                (12, rank * d),
                (13, d * rank),
                (14, 2 * d * d),
                (15, 2 * d * d),
            ]
            .into_iter()
            .map(|(id, n)| LocalAdam::new(id, n))
            .collect(),
            cosines,
            phase: (0..d + s).map(|_| rng.next_u32() as usize % d).collect(),
            previous_h: vec![0.0; d * s],
            rate_derivative: vec![0.0; d * s],
            ea: vec![0.0; d * s],
            ed: vec![0.0; d * s],
            eb: vec![0.0; d * s],
            delta_dot: vec![0.0; d],
            delta_slope: vec![0.0; d],
            adapter_slope: vec![0.0; rank],
            b_dot: vec![0.0; s],
            delta_grad: vec![0.0; d],
            b_grad: vec![0.0; s],
            delta_m: vec![0.0; d * d],
            delta_v: vec![0.0; d * d],
            b_m: vec![0.0; s * d],
            b_v: vec![0.0; s * d],
            gz: vec![0.0; d],
            gz_raw: vec![0.0; d],
            mlp_grad: vec![0.0; 2 * d],
            adapter_grad: vec![0.0; rank],
            c_grad: vec![0.0; s],
            steps: 0,
            input: input_eligibility.then(|| InputEligibility::new(c, seed)),
        }
    }

    pub fn bytes(&self) -> usize {
        let n = [
            &self.cosines,
            &self.previous_h,
            &self.rate_derivative,
            &self.ea,
            &self.ed,
            &self.eb,
            &self.delta_dot,
            &self.delta_slope,
            &self.adapter_slope,
            &self.b_dot,
            &self.delta_grad,
            &self.b_grad,
            &self.delta_m,
            &self.delta_v,
            &self.b_m,
            &self.b_v,
            &self.gz,
            &self.gz_raw,
            &self.mlp_grad,
            &self.adapter_grad,
            &self.c_grad,
            &self.observations.bar_a,
            &self.observations.delta_raw,
            &self.observations.adapter_raw,
            &self.observations.mlp_raw,
        ]
        .iter()
        .map(|v| v.capacity())
        .sum::<usize>()
            + self
                .parts
                .iter()
                .map(|p| p.grad.capacity() + p.m.capacity() + p.v.capacity())
                .sum::<usize>();
        n * 4
            + self.phase.capacity() * std::mem::size_of::<usize>()
            + self.input.as_ref().map_or(0, InputEligibility::bytes)
    }

    pub fn prepare(&mut self, block: &PSSAContinuousBlockV2, inputs: &[usize]) {
        // TBPTT's incoming carry is detached: all parameter traces start at zero.
        self.ea.fill(0.0);
        self.ed.fill(0.0);
        self.eb.fill(0.0);
        self.delta_grad.fill(0.0);
        self.b_grad.fill(0.0);
        for part in &mut self.parts {
            part.grad.fill(0.0);
        }
        for (out, &raw) in self.rate_derivative.iter_mut().zip(&block.a_mat.data) {
            *out = -sigmoid(raw);
        }
        if let Some(input) = &mut self.input {
            input.prepare(inputs, self.steps, block.cfg.d_latent);
        }
    }

    pub fn before_token(&mut self, block: &PSSAContinuousBlockV2) {
        self.previous_h.copy_from_slice(&block.h_persistent);
    }

    pub fn observe(
        &mut self,
        block: &PSSAContinuousBlockV2,
        head: &ParamMatrix,
        error: &[f32],
        embed: &ParamMatrix,
        token: usize,
    ) {
        let (d, s, rank) = (
            block.cfg.d_latent,
            block.cfg.d_state,
            block.adapters[0].rank,
        );
        for (out, &raw) in self
            .delta_slope
            .iter_mut()
            .zip(&self.observations.delta_raw)
        {
            *out = sigmoid(raw);
        }
        for (out, &raw) in self
            .adapter_slope
            .iter_mut()
            .zip(&self.observations.adapter_raw)
        {
            *out = silu_derivative(raw);
        }
        head.matvec_transpose(error, &mut self.gz);
        block.mlp_w2.matvec_transpose(&self.gz, &mut self.mlp_grad);
        for (g, &raw) in self.mlp_grad.iter_mut().zip(&self.observations.mlp_raw) {
            *g *= silu_derivative(raw);
        }
        block
            .mlp_w1
            .matvec_transpose(&self.mlp_grad, &mut self.gz_raw);
        for (raw, &direct) in self.gz_raw.iter_mut().zip(&self.gz) {
            *raw += direct;
        }
        outer_add(&mut self.parts[5].grad, &self.gz, &block.inf_mlp_act);
        outer_add(&mut self.parts[4].grad, &self.mlp_grad, &block.inf_z_raw);
        block.adapters[0].total_up_matvec_transpose(&self.gz_raw, &mut self.adapter_grad);
        for (g, &slope) in self.adapter_grad.iter_mut().zip(&self.adapter_slope) {
            *g *= slope;
        }
        outer_add(
            &mut self.parts[3].grad,
            &self.gz_raw,
            &block.inf_ad_act[..rank],
        );
        outer_add(
            &mut self.parts[2].grad,
            &self.adapter_grad,
            &block.inf_x_norm,
        );

        for i in 0..d {
            let frequency = (self.steps + self.phase[i]) % d;
            self.delta_dot[i] = crate::linalg::dot_slice(
                &self.cosines[frequency * d..(frequency + 1) * d],
                &block.inf_x_norm,
            );
        }
        for j in 0..s {
            let frequency = (self.steps + self.phase[d + j]) % d;
            self.b_dot[j] = crate::linalg::dot_slice(
                &self.cosines[frequency * d..(frequency + 1) * d],
                &block.inf_x_norm,
            );
        }
        self.c_grad.fill(0.0);
        let ssm_scale = 1.0 / (s as f32).sqrt();
        for i in 0..d {
            let delta = block.inf_delta[i];
            let delta_tangent = self.delta_slope[i] * self.delta_dot[i];
            let gy = self.gz_raw[i] * ssm_scale;
            for j in 0..s {
                let index = i * s + j;
                let a = self.observations.bar_a[index];
                let previous = self.previous_h[index];
                self.ea[index] =
                    a * self.ea[index] + delta * a * previous * self.rate_derivative[index];
                self.ed[index] = a * self.ed[index]
                    + delta_tangent
                        * (block.ssm_rates[index] * a * previous
                            + block.inf_b[j] * block.inf_x_norm[i]);
                self.eb[index] = a * self.eb[index] + delta * block.inf_x_norm[i] * self.b_dot[j];
                let gh = gy * block.inf_c[j];
                self.parts[0].grad[index] += gh * self.ea[index];
                self.delta_grad[i] += gh * self.ed[index];
                self.b_grad[j] += gh * self.eb[index];
                self.c_grad[j] += gy * block.h_persistent[index];
            }
        }
        outer_add(&mut self.parts[1].grad, &self.c_grad, &block.inf_x_norm);
        if let Some(input) = &mut self.input {
            input.observe(
                block,
                embed,
                token,
                &self.previous_h,
                &self.observations,
                &self.gz_raw,
                &self.cosines,
                &self.delta_slope,
                &self.adapter_slope,
            );
        }
    }

    fn norm(&self) -> f64 {
        let local = self
            .parts
            .iter()
            .flat_map(|p| &p.grad)
            .chain(&self.delta_grad)
            .chain(&self.b_grad)
            .map(|&g| (g as f64).powi(2))
            .sum::<f64>();
        (local + self.input.as_ref().map_or(0.0, InputEligibility::norm_sq)).sqrt()
    }

    pub fn validate_update(
        &self,
        model: &mut PSSALayerV2,
        options: InterdiffusionConfig,
    ) -> Result<(), String> {
        let norm = self.norm();
        if !norm.is_finite() {
            return Err("non-finite streaming eligibility gradient".into());
        }
        let clip = (options.max_gradient_norm as f64 / norm.max(f64::MIN_POSITIVE)).min(1.0) as f32;
        let c = model.cfg.clone();
        let bias = biases(&c, self.steps + 1);
        for part in &self.parts {
            let t = tensor(model, part.tensor);
            for (i, &w) in t.data.iter().enumerate() {
                let (w, m, v) = adam_candidate(
                    w,
                    part.grad[i] * clip,
                    part.m[i],
                    part.v[i],
                    &c,
                    options.body_learning_rate,
                    bias,
                );
                if !w.is_finite() || !m.is_finite() || !v.is_finite() {
                    return Err("non-finite eligibility update".into());
                }
            }
        }
        self.validate_wave(
            &model.block.w_delta,
            &self.delta_grad,
            &self.delta_m,
            &self.delta_v,
            0,
            clip,
            &c,
            options,
        )?;
        self.validate_wave(
            &model.block.w_b,
            &self.b_grad,
            &self.b_m,
            &self.b_v,
            c.d_latent,
            clip,
            &c,
            options,
        )?;
        if let Some(input) = &self.input {
            input.validate_update(model, options, clip, &self.cosines)?;
        }
        Ok(())
    }

    fn validate_wave(
        &self,
        matrix: &ParamMatrix,
        grad: &[f32],
        m: &[f32],
        v: &[f32],
        phase_start: usize,
        clip: f32,
        c: &PSSAConfigV2,
        options: InterdiffusionConfig,
    ) -> Result<(), String> {
        let d = matrix.cols;
        let bias = biases(c, self.steps / d + 1);
        for (row, &g) in grad.iter().enumerate() {
            let frequency = (self.steps + self.phase[phase_start + row]) % d;
            let index = row * d + frequency;
            let (delta, m, v) = adam_candidate(
                0.0,
                g * clip,
                m[index],
                v[index],
                c,
                options.body_learning_rate
                    * if options.coordinate_rms_scaling {
                        (d as f32).sqrt()
                    } else {
                        1.0
                    },
                bias,
            );
            if !m.is_finite()
                || !v.is_finite()
                || matrix.data[row * d..(row + 1) * d]
                    .iter()
                    .zip(&self.cosines[frequency * d..(frequency + 1) * d])
                    .any(|(&w, &u)| !(w + delta * u).is_finite())
            {
                return Err("non-finite cosine eligibility update".into());
            }
        }
        Ok(())
    }

    pub fn apply(&mut self, model: &mut PSSALayerV2, options: InterdiffusionConfig) {
        let norm = self.norm();
        let clip = (options.max_gradient_norm as f64 / norm.max(f64::MIN_POSITIVE)).min(1.0) as f32;
        let c = model.cfg.clone();
        let bias = biases(&c, self.steps + 1);
        for part in &mut self.parts {
            let t = tensor(model, part.tensor);
            for (i, w) in t.data.iter_mut().enumerate() {
                let (new_w, m, v) = adam_candidate(
                    *w,
                    part.grad[i] * clip,
                    part.m[i],
                    part.v[i],
                    &c,
                    options.body_learning_rate,
                    bias,
                );
                *w = new_w;
                part.m[i] = m;
                part.v[i] = v;
            }
        }
        let bias = biases(&c, self.steps / c.d_latent + 1);
        apply_wave(
            &mut model.block.w_delta,
            &self.delta_grad,
            &mut self.delta_m,
            &mut self.delta_v,
            &self.phase[..c.d_latent],
            &self.cosines,
            self.steps,
            clip,
            &c,
            options.body_learning_rate
                * if options.coordinate_rms_scaling {
                    (c.d_latent as f32).sqrt()
                } else {
                    1.0
                },
            bias,
        );
        apply_wave(
            &mut model.block.w_b,
            &self.b_grad,
            &mut self.b_m,
            &mut self.b_v,
            &self.phase[c.d_latent..],
            &self.cosines,
            self.steps,
            clip,
            &c,
            options.body_learning_rate
                * if options.coordinate_rms_scaling {
                    (c.d_latent as f32).sqrt()
                } else {
                    1.0
                },
            bias,
        );
        if let Some(input) = &mut self.input {
            input.apply(model, options, clip, &self.cosines);
        }
        self.steps += 1;
    }
}

fn biases(c: &PSSAConfigV2, step: usize) -> (f32, f32) {
    (
        1.0 - c.beta1.powf(step as f32),
        1.0 - c.beta2.powf(step as f32),
    )
}

fn adam_candidate(
    w: f32,
    g: f32,
    old_m: f32,
    old_v: f32,
    c: &PSSAConfigV2,
    lr: f32,
    bias: (f32, f32),
) -> (f32, f32, f32) {
    let m = c.beta1 * old_m + (1.0 - c.beta1) * g;
    let v = c.beta2 * old_v + (1.0 - c.beta2) * g * g;
    (
        w - lr * c.weight_decay * w - lr * (m / bias.0) / ((v / bias.1).sqrt() + c.eps),
        m,
        v,
    )
}

fn apply_wave(
    matrix: &mut ParamMatrix,
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    phase: &[usize],
    cosines: &[f32],
    step: usize,
    clip: f32,
    c: &PSSAConfigV2,
    lr: f32,
    bias: (f32, f32),
) {
    let d = matrix.cols;
    for (row, &g) in grad.iter().enumerate() {
        let frequency = (step + phase[row]) % d;
        let index = row * d + frequency;
        let (delta, new_m, new_v) = adam_candidate(0.0, g * clip, m[index], v[index], c, lr, bias);
        m[index] = new_m;
        v[index] = new_v;
        for (w, &u) in matrix.data[row * d..(row + 1) * d]
            .iter_mut()
            .zip(&cosines[frequency * d..(frequency + 1) * d])
        {
            *w += delta * u;
        }
    }
}

fn silu_derivative(raw: f32) -> f32 {
    let s = sigmoid(raw);
    s * (1.0 + raw * (1.0 - s))
}

fn outer_add(out: &mut [f32], left: &[f32], right: &[f32]) {
    for (row, &g) in out.chunks_exact_mut(right.len()).zip(left) {
        if g == 0.0 {
            continue;
        }
        for (out, &x) in row.iter_mut().zip(right) {
            *out += g * x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_probe_preserves_existing_eligibility_and_head_optimizer_state() {
        let cfg = PSSAConfigV2 {
            d_vocab: 7,
            d_latent: 4,
            d_state: 3,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 4,
            ..Default::default()
        };
        let mut actual = InterdiffusionTrainer::new(
            cfg,
            89,
            InterdiffusionConfig {
                body_every: 1,
                input_eligibility: false,
                ..Default::default()
            },
        )
        .unwrap();
        actual.train_step(&[1, 2], &[2, 3], true, false).unwrap();
        let weights: Vec<_> = (0..tensor_count(&actual.forward.model))
            .map(|id| tensor(&mut actual.forward.model, id).data.to_vec())
            .collect();
        let head = (actual.head_m.clone(), actual.head_v.clone());
        let carry = actual.forward.model.h_persistent.clone();
        let memory = actual.forward.model.memory.clone();
        let optimizer_state = |e: &Eligibility| {
            e.parts
                .iter()
                .flat_map(|p| p.m.iter().chain(&p.v))
                .chain(&e.delta_m)
                .chain(&e.delta_v)
                .chain(&e.b_m)
                .chain(&e.b_v)
                .copied()
                .collect::<Vec<_>>()
        };
        let eligibility = actual.eligibility.as_ref().unwrap();
        let moments = optimizer_state(eligibility);
        assert!(moments.iter().any(|&x| x != 0.0));
        let steps = eligibility.steps;
        let next_tensor = actual.next_tensor;
        actual.config.epsilon = f32::MAX;
        assert!(actual.train_step(&[3, 1], &[1, 2], false, true).is_err());
        for (id, expected) in weights.iter().enumerate() {
            assert_eq!(tensor(&mut actual.forward.model, id).data, expected);
        }
        assert_eq!((actual.head_m.clone(), actual.head_v.clone()), head);
        assert_eq!(actual.forward.model.h_persistent, carry);
        assert_eq!(actual.forward.model.memory, memory);
        assert_eq!(
            optimizer_state(actual.eligibility.as_ref().unwrap()),
            moments
        );
        assert_eq!(actual.eligibility.as_ref().unwrap().steps, steps);
        assert_eq!(actual.forward.model.step_counter, 1);
        assert_eq!(actual.next_tensor, next_tensor);
        actual.config.epsilon = 0.001;
        actual.train_step(&[3, 1], &[1, 2], false, false).unwrap();
        assert_eq!(actual.eligibility.as_ref().unwrap().steps, steps + 1);
    }

    #[test]
    fn streaming_gradients_and_cosine_tangents_match_tbptt_with_nonzero_carry_and_mlp() {
        let cfg = PSSAConfigV2 {
            d_vocab: 7,
            d_latent: 4,
            d_state: 3,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 4,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut actual = InterdiffusionTrainer::new(
            cfg.clone(),
            89,
            InterdiffusionConfig {
                head_learning_rate: 1e-6,
                body_learning_rate: 1e-6,
                max_gradient_norm: 1e6,
                ..Default::default()
            },
        )
        .unwrap();
        let mut reference = PSSALayerV2::new(cfg, 89);
        for model in [&mut actual.forward.model, &mut reference] {
            for (i, w) in model.block.mlp_w2.data.iter_mut().enumerate() {
                *w = (i as f32 % 7.0 - 3.0) * 0.01;
            }
            for (i, w) in model.block.adapters[0].up_proj.data.iter_mut().enumerate() {
                *w = (i as f32 % 5.0 - 2.0) * 0.01;
            }
            model.block.adapters[0].consolidated_up.fill(0.03);
        }
        actual.forward.loss(&[1, 2], &[2, 3], true).unwrap();
        reference.forward_train_chunk(&[1, 2], &[2, 3]);
        let report = actual
            .train_step(&[3, 1, 2, 4], &[1, 2, 4, 3], false, false)
            .unwrap();
        assert!(report.eligibility_updates);
        reference.forward_train_chunk(&[3, 1, 2, 4], &[1, 2, 4, 3]);
        reference.zero_gradients();
        reference.backward_chunk(4, 1.0);
        let eligibility = actual.eligibility.as_ref().unwrap();
        for part in &eligibility.parts {
            let expected = tensor(&mut reference, part.tensor);
            for (&a, &b) in part.grad.iter().zip(expected.grad.iter()) {
                assert!(
                    (a - b).abs() < 2e-5 + b.abs() * 0.01,
                    "tensor {}: forward {a}, reverse {b}",
                    part.tensor
                );
            }
        }
        let d = reference.cfg.d_latent;
        for (id, gradients, phase_start) in
            [(5, &eligibility.delta_grad, 0), (6, &eligibility.b_grad, d)]
        {
            let expected = tensor(&mut reference, id);
            for (row, &actual) in gradients.iter().enumerate() {
                let frequency = (eligibility.steps - 1 + eligibility.phase[phase_start + row]) % d;
                let direction = &eligibility.cosines[frequency * d..(frequency + 1) * d];
                let exact = expected.grad[row * d..(row + 1) * d]
                    .iter()
                    .zip(direction)
                    .map(|(&g, &u)| g as f64 * u as f64)
                    .sum::<f64>();
                assert!(
                    (actual as f64 - exact).abs() < 2e-5 + exact.abs() * 0.01,
                    "tensor {id}/{row}: projected {actual}, reverse projection {exact}"
                );
            }
        }
        let input = eligibility.input.as_ref().unwrap();
        for (index, &id) in input.signals.iter().enumerate() {
            let gradient = if id == reference.cfg.d_vocab {
                &reference.block.norm_gamma.grad[..]
            } else if id == reference.cfg.d_vocab + 1 {
                &reference.block.norm_beta.grad[..]
            } else {
                &reference.embed_w.grad[id * d..(id + 1) * d]
            };
            let frequency = input.frequencies[index];
            let exact = gradient
                .iter()
                .zip(&eligibility.cosines[frequency * d..(frequency + 1) * d])
                .map(|(&g, &u)| g as f64 * u as f64)
                .sum::<f64>();
            assert!(
                (input.grad[index] as f64 - exact).abs() < 2e-5 + exact.abs() * 0.01,
                "input {id}: projected {}, reverse {exact}",
                input.grad[index]
            );
        }
        assert_eq!(report.forward_evaluations, 1);
        assert_eq!(actual.forward.model.h_persistent, reference.h_persistent);
    }
}
