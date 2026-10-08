#[derive(Debug, PartialEq, Clone)]
pub enum UpdateOutcome {
    Defended {
        remaining_conf: f32,
        damage_absorbed: f32,
    },
    Overwritten,
}

pub struct RateLimiterGate;

impl RateLimiterGate {
    #[inline(always)]
    pub fn compute_refractory_gain(delta_tokens: usize, tau_refractory: f32) -> f32 {
        let dt = delta_tokens as f32;
        let linear_factor = 1.0 - (-dt / tau_refractory).exp();
        (linear_factor * linear_factor).max(0.005)
    }

    #[inline(always)]
    pub fn apply_refractory_overwrite(
        conf: &mut f32,
        last_seen_step: &mut usize,
        current_step: usize,
        surprise: f32,
    ) -> UpdateOutcome {
        let delta_t = current_step.saturating_sub(*last_seen_step);
        let gain_mult = Self::compute_refractory_gain(delta_t, 60.0);
        let damage = 0.08 * surprise * gain_mult;

        if *conf > damage {
            *conf -= damage;
            *last_seen_step = current_step;
            UpdateOutcome::Defended {
                remaining_conf: *conf,
                damage_absorbed: damage,
            }
        } else {
            *conf = 1.0;
            *last_seen_step = current_step;
            UpdateOutcome::Overwritten
        }
    }
}
