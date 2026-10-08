//! Exact, certificate-driven sparse inference helpers.
//!
//! Memory keys are grouped geometrically, with interval distance bounds and a
//! bounded sparse-work scheduler. Vocabulary clusters store a centroid and the
//! maximum row deviation. Both readers fall back to exhaustive evaluation
//! whenever a requested certificate cannot be proved.

use crate::linalg::dot_slice;
use crate::memory::HyperbolicEpisodicBankV2;
use crate::pssa::ParamMatrix;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseReadStats {
    pub scanned_slots: usize,
    pub populated_slots: usize,
    pub omitted_mass_bound: f32,
    pub certified: bool,
    pub exact_fallback: bool,
    pub center_distance_evaluations: usize,
    pub value_rows_mixed: usize,
    pub cluster_bound_evaluations: usize,
    pub budget_fallback: bool,
    pub routing_bypassed: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct MemoryCluster {
    lower: Vec<f32>,
    upper: Vec<f32>,
    min_norm_sq: f32,
    start: usize,
    end: usize,
}

/// Geometric confidence-scheduled retrieval (GCSR) over occupied memory slots.
/// The index is runtime-only and must be rebuilt after a memory write.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedMemoryIndex {
    dim_key: usize,
    count: usize,
    clusters: Vec<MemoryCluster>,
    slots: Vec<usize>,
    order: Vec<usize>,
    lower_bounds: Vec<f64>,
    tail_mass: Vec<f64>,
    unprofitable_reads: usize,
    dense_reads_remaining: usize,
}

impl CertifiedMemoryIndex {
    pub fn build(bank: &HyperbolicEpisodicBankV2, cluster_size: usize) -> Result<Self, String> {
        if cluster_size == 0 {
            return Err("certified memory cluster size must be positive".into());
        }
        let key_len = bank
            .count
            .checked_mul(bank.dim_key)
            .ok_or("certified memory dimensions overflow")?;
        if bank.dim_key == 0
            || bank.count > bank.capacity
            || bank.keys.len() < key_len
            || bank.norm_sq.len() < bank.count
            || bank.norm_sq[..bank.count]
                .iter()
                .any(|&s| !s.is_finite() || !(0.0..1.0).contains(&s))
        {
            return Err("certified memory bank has invalid dimensions or key norms".into());
        }
        // Farthest-first centres use the monotone argument of asinh, avoiding
        // transcendental work during assignment. Identical keys stay together.
        let mut nearest = vec![f64::INFINITY; bank.count];
        let mut assignments = vec![0; bank.count];
        let mut centers = 0;
        for group in 0..bank.count.div_ceil(cluster_size) {
            let center = (0..bank.count)
                .max_by(|&a, &b| nearest[a].total_cmp(&nearest[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            if nearest[center] == 0.0 {
                break;
            }
            let key = &bank.keys[center * bank.dim_key..(center + 1) * bank.dim_key];
            for idx in 0..bank.count {
                let other = &bank.keys[idx * bank.dim_key..(idx + 1) * bank.dim_key];
                let squared: f64 = key
                    .iter()
                    .zip(other)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum();
                let denom = (1.0 - bank.norm_sq[center] as f64) * (1.0 - bank.norm_sq[idx] as f64);
                if !(squared.is_finite() && denom.is_finite() && denom > 0.0) {
                    return Err(
                        "certified memory keys must be finite and inside the open ball".into(),
                    );
                }
                let distance = squared / denom;
                if distance < nearest[idx] {
                    nearest[idx] = distance;
                    assignments[idx] = group;
                }
            }
            centers += 1;
        }
        let mut clusters = Vec::new();
        let mut slots = Vec::with_capacity(bank.count);
        for group in 0..centers {
            let start = slots.len();
            slots.extend((0..bank.count).filter(|&idx| assignments[idx] == group));
            let end = slots.len();
            for offset in (start..end).step_by(cluster_size) {
                let stop = (offset + cluster_size).min(end);
                let mut lower = vec![f32::INFINITY; bank.dim_key];
                let mut upper = vec![f32::NEG_INFINITY; bank.dim_key];
                let mut min_norm_sq = f32::INFINITY;
                for &idx in &slots[offset..stop] {
                    min_norm_sq = min_norm_sq.min(bank.norm_sq[idx]);
                    for j in 0..bank.dim_key {
                        let key = bank.keys[idx * bank.dim_key + j];
                        lower[j] = lower[j].min(key);
                        upper[j] = upper[j].max(key);
                    }
                }
                clusters.push(MemoryCluster {
                    lower,
                    upper,
                    min_norm_sq,
                    start: offset,
                    end: stop,
                });
            }
        }
        let order = (0..clusters.len()).collect();
        let lower_bounds = vec![0.0; clusters.len()];
        Ok(Self {
            dim_key: bank.dim_key,
            count: bank.count,
            tail_mass: vec![0.0; clusters.len() + 1],
            clusters,
            slots,
            order,
            lower_bounds,
            unprofitable_reads: 0,
            dense_reads_remaining: 0,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    pub fn numeric_storage_bytes(&self) -> usize {
        self.clusters
            .iter()
            .map(|c| (c.lower.capacity() + c.upper.capacity()) * 4)
            .sum::<usize>()
            + self.order.capacity() * std::mem::size_of::<usize>()
            + self.lower_bounds.capacity() * 8
            + self.slots.capacity() * std::mem::size_of::<usize>()
            + self.tail_mass.capacity() * 8
    }

    /// Read with a certified upper bound on omitted softmax mass.
    ///
    /// `out_weights` is zero for omitted slots and is normalized over the
    /// returned support.  The exact path is selected automatically if the
    /// bound cannot reach `epsilon`.
    pub fn retrieve_soft_into(
        &mut self,
        bank: &HyperbolicEpisodicBankV2,
        q_pnc: &[f32],
        tau: f32,
        epsilon: f32,
        out_val: &mut [f32],
        out_weights: &mut [f32],
    ) -> Result<SparseReadStats, String> {
        if !(epsilon.is_finite() && (0.0..1.0).contains(&epsilon)) {
            return Err("certified memory epsilon must be finite and in [0, 1)".into());
        }
        if self.dim_key != bank.dim_key || self.count != bank.count {
            return Err("certified memory index is stale; rebuild it after memory writes".into());
        }
        if q_pnc.len() != bank.dim_key || out_val.len() != bank.dim_val {
            return Err("certified memory buffer dimensions do not match the bank".into());
        }
        if out_weights.len() < bank.count {
            return Err("certified memory weight buffer is too small".into());
        }
        if !(tau.is_finite() && tau > 0.0) {
            return Err("memory temperature must be positive and finite".into());
        }
        if bank.count == 0 {
            out_val.fill(0.0);
            out_weights.fill(0.0);
            return Ok(SparseReadStats::default());
        }

        let q_sq = HyperbolicEpisodicBankV2::squared_norm(q_pnc);
        if !(q_sq.is_finite() && q_sq < 1.0) {
            return Err("certified memory query must be in the open Poincare ball".into());
        }
        if epsilon == 0.0 || self.clusters.len() < 2 || self.dense_reads_remaining > 0 {
            self.dense_reads_remaining = self.dense_reads_remaining.saturating_sub(1);
            bank.retrieve_soft_into(q_pnc, tau, out_val, out_weights);
            out_weights[bank.count..].fill(0.0);
            return Ok(SparseReadStats {
                scanned_slots: bank.count,
                populated_slots: bank.count,
                value_rows_mixed: bank.count,
                exact_fallback: true,
                routing_bypassed: true,
                ..Default::default()
            });
        }
        out_weights[..bank.count].fill(f32::NAN);
        for (bound, cluster) in self.lower_bounds.iter_mut().zip(&self.clusters) {
            // Direct interval bounds use the dense reader's cached FP32 norms;
            // no triangle inequality for a rounded pseudo-metric is assumed.
            let squared: f64 = q_pnc
                .iter()
                .zip(&cluster.lower)
                .zip(&cluster.upper)
                .map(|((&q, &lo), &hi)| {
                    let delta = (lo as f64 - q as f64).max(q as f64 - hi as f64).max(0.0);
                    delta * delta
                })
                .sum();
            let denom = (1.0 - q_sq as f64) * (1.0 - cluster.min_norm_sq as f64);
            let distance = 2.0 * (squared / denom).sqrt().asinh();
            let guard = 8.0 * f32::EPSILON as f64 * (distance + 1.0);
            *bound = (distance - guard).max(0.0);
        }
        self.order.sort_unstable_by(|&a, &b| {
            self.lower_bounds[a]
                .total_cmp(&self.lower_bounds[b])
                .then_with(|| a.cmp(&b))
        });

        // One common shift and a suffix sum replace quadratic tail rescans and
        // per-slot rescaling. Underflow cannot accept an empty observed mass.
        let reference = self.lower_bounds[self.order[0]];
        self.tail_mass[self.order.len()] = 0.0;
        for position in (0..self.order.len()).rev() {
            let id = self.order[position];
            let cluster = &self.clusters[id];
            self.tail_mass[position] = self.tail_mass[position + 1]
                + (cluster.end - cluster.start) as f64
                    * ((reference - self.lower_bounds[id]) / tau as f64).exp();
        }
        let mut scanned_slots = 0usize;
        let mut min_scanned = f32::INFINITY;
        let mut omitted_bound = 1.0;
        let mut scanned_mass = 0.0f64;
        let mut certified = false;
        let mut budget_fallback = false;
        for (position, &cluster_id) in self.order.iter().enumerate() {
            let cluster = &self.clusters[cluster_id];
            for &idx in &self.slots[cluster.start..cluster.end] {
                let off = idx * bank.dim_key;
                let dist = HyperbolicEpisodicBankV2::poincare_distance(
                    q_pnc,
                    q_sq,
                    &bank.keys[off..off + bank.dim_key],
                    bank.norm_sq[idx],
                );
                out_weights[idx] = dist;
                min_scanned = min_scanned.min(dist);
                scanned_mass += ((reference - dist as f64) / tau as f64).exp();
                scanned_slots += 1;
            }
            let unscanned_mass = self.tail_mass[position + 1];
            if scanned_mass > 0.0 && position + 1 < self.order.len() {
                omitted_bound = (unscanned_mass / (scanned_mass + unscanned_mass)
                    + 16.0 * f32::EPSILON as f64)
                    .min(1.0) as f32;
                if omitted_bound <= epsilon {
                    certified = true;
                    break;
                }
            }
            if scanned_slots >= bank.count / 2 {
                budget_fallback = position + 1 < self.order.len();
                break;
            }
        }

        if !certified {
            // Preserve the production reader's accumulation order and bits on
            // fallback.  This is also the safety valve for diffuse banks.
            // Reuse verified slots and finish only the missing distances.
            for (idx, distance) in out_weights[..bank.count].iter_mut().enumerate() {
                if distance.is_nan() {
                    *distance = HyperbolicEpisodicBankV2::poincare_distance(
                        q_pnc,
                        q_sq,
                        &bank.keys[idx * bank.dim_key..(idx + 1) * bank.dim_key],
                        bank.norm_sq[idx],
                    );
                    min_scanned = min_scanned.min(*distance);
                }
            }
            self.unprofitable_reads += 1;
            if self.unprofitable_reads >= 3 {
                // ponytail: fixed bounded backoff; profile-based scheduling if
                // future deployments need hardware/load-specific calibration.
                self.dense_reads_remaining = 31;
                self.unprofitable_reads = 0;
            }
            let mut sum = 0.0f32;
            for w in &mut out_weights[..bank.count] {
                *w = ((min_scanned - *w) / tau).exp();
                sum += *w;
            }
            out_weights[bank.count..].fill(0.0);
            out_val.fill(0.0);
            for idx in 0..bank.count {
                let weight = out_weights[idx] / sum;
                out_weights[idx] = weight;
                for j in 0..bank.dim_val {
                    out_val[j] += weight * bank.values[idx * bank.dim_val + j];
                }
            }
            return Ok(SparseReadStats {
                scanned_slots: bank.count,
                populated_slots: bank.count,
                omitted_mass_bound: 0.0,
                certified: false,
                exact_fallback: true,
                value_rows_mixed: bank.count,
                cluster_bound_evaluations: self.clusters.len(),
                budget_fallback,
                ..Default::default()
            });
        }

        self.unprofitable_reads = 0;

        let mut sum = 0.0f64;
        for weight in &mut out_weights[..bank.count] {
            if weight.is_finite() {
                *weight = ((*weight - min_scanned) / -tau).exp();
                sum += *weight as f64;
            } else {
                *weight = 0.0;
            }
        }
        out_val.fill(0.0);
        let mut value_rows_mixed = 0;
        for idx in 0..bank.count {
            let weight = (out_weights[idx] as f64 / sum) as f32;
            out_weights[idx] = weight;
            if weight == 0.0 {
                continue;
            }
            value_rows_mixed += 1;
            let off = idx * bank.dim_val;
            for j in 0..bank.dim_val {
                out_val[j] += weight * bank.values[off + j];
            }
        }
        out_weights[bank.count..].fill(0.0);
        Ok(SparseReadStats {
            scanned_slots,
            populated_slots: bank.count,
            omitted_mass_bound: omitted_bound,
            certified,
            exact_fallback: false,
            value_rows_mixed,
            cluster_bound_evaluations: self.clusters.len(),
            ..Default::default()
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct VocabularyCluster {
    centroid: Vec<f32>,
    radius: f64,
    max_row_norm: f64,
    centroid_norm: f64,
    rows: Vec<usize>,
}

/// Branch-and-bound index for exact greedy output projection.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedVocabularyIndex {
    dim: usize,
    vocab: usize,
    clusters: Vec<VocabularyCluster>,
    order: Vec<usize>,
    upper_bounds: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VocabularySearchStats {
    pub visited_clusters: usize,
    pub exact_rows: usize,
    pub total_rows: usize,
    pub centroid_rows: usize,
}

impl VocabularySearchStats {
    pub fn certified(&self) -> bool {
        self.exact_rows < self.total_rows
    }
    pub fn exact_fallback(&self) -> bool {
        self.total_rows > 0 && !self.certified()
    }
}

impl CertifiedVocabularyIndex {
    pub fn build(weights: &ParamMatrix, cluster_count: usize) -> Result<Self, String> {
        if weights.rows == 0 || weights.cols == 0 || cluster_count == 0 {
            return Err("vocabulary index dimensions and cluster count must be positive".into());
        }
        if weights.data.iter().any(|x| !x.is_finite()) {
            return Err("vocabulary weights must be finite".into());
        }
        let clusters_n = cluster_count.min(weights.rows);
        let rows_per_cluster = weights.rows.div_ceil(clusters_n);
        let mut clusters = Vec::new();
        for start in (0..weights.rows).step_by(rows_per_cluster) {
            let end = (start + rows_per_cluster).min(weights.rows);
            let mut centroid = vec![0.0f32; weights.cols];
            for (col, dst) in centroid.iter_mut().enumerate() {
                *dst = ((start..end)
                    .map(|row| weights.data[row * weights.cols + col] as f64)
                    .sum::<f64>()
                    / (end - start) as f64) as f32;
            }
            let mut radius = 0.0f64;
            let mut max_row_norm = 0.0f64;
            for row in start..end {
                let row_data = &weights.data[row * weights.cols..(row + 1) * weights.cols];
                let distance = row_data
                    .iter()
                    .zip(&centroid)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                radius = radius.max(distance * (1.0 + 8.0 * f64::EPSILON));
                max_row_norm = max_row_norm.max(
                    row_data
                        .iter()
                        .map(|&w| (w as f64).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                );
            }
            let centroid_norm = centroid
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            clusters.push(VocabularyCluster {
                centroid,
                radius,
                max_row_norm,
                centroid_norm,
                rows: (start..end).collect(),
            });
        }
        Ok(Self {
            dim: weights.cols,
            vocab: weights.rows,
            order: (0..clusters.len()).collect(),
            upper_bounds: vec![0.0; clusters.len()],
            clusters,
        })
    }

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    pub fn numeric_storage_bytes(&self) -> usize {
        self.clusters
            .iter()
            .map(|c| c.centroid.capacity() * 4 + c.rows.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
            + self.order.capacity() * std::mem::size_of::<usize>()
            + self.upper_bounds.capacity() * 8
    }

    /// Return the exact greedy token and the amount of row work performed.
    /// Equality is deliberately not pruned, preserving lowest-ID tie rules.
    pub fn exact_greedy(
        &mut self,
        weights: &ParamMatrix,
        z: &[f32],
    ) -> Result<(usize, VocabularySearchStats), String> {
        self.exact_greedy_from(weights, z, 0)
    }

    pub fn exact_greedy_from(
        &mut self,
        weights: &ParamMatrix,
        z: &[f32],
        first_row: usize,
    ) -> Result<(usize, VocabularySearchStats), String> {
        if weights.rows != self.vocab || weights.cols != self.dim || z.len() != self.dim {
            return Err("vocabulary index is stale or has incompatible dimensions".into());
        }
        if first_row >= weights.rows {
            return Err("vocabulary search has no valid rows".into());
        }
        if z.iter().any(|x| !x.is_finite()) {
            return Err("vocabulary query must be finite".into());
        }
        let norm = z.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let scale = 1.0 / (self.dim as f32).sqrt();
        let u = (self.dim + 8) as f64 * f32::EPSILON as f64;
        let gamma = u / (1.0 - u).max(f64::MIN_POSITIVE);
        for (upper, cluster) in self.upper_bounds.iter_mut().zip(&self.clusters) {
            // Cover FP32 dot/FMA and final scale rounding as well as real-valued
            // Cauchy-Schwarz. Exact rows use the dense head's SIMD dot kernel.
            let bound = (dot_slice(&cluster.centroid, z) as f64
                + (cluster.radius + gamma * (cluster.max_row_norm + cluster.centroid_norm)) * norm)
                * scale as f64;
            *upper = bound + 2.0 * f32::EPSILON as f64 * bound.abs();
        }
        self.order.sort_unstable_by(|&a, &b| {
            self.upper_bounds[b]
                .total_cmp(&self.upper_bounds[a])
                .then_with(|| a.cmp(&b))
        });

        let mut best_id = 0usize;
        let mut best = f32::NEG_INFINITY;
        let mut visited_clusters = 0;
        let mut exact_rows = 0;
        for (position, &cluster_id) in self.order.iter().enumerate() {
            let cluster = &self.clusters[cluster_id];
            for &row in &cluster.rows {
                if row < first_row {
                    continue;
                }
                let score =
                    dot_slice(&weights.data[row * self.dim..(row + 1) * self.dim], z) * scale;
                exact_rows += 1;
                if score > best || (score == best && row < best_id) {
                    best = score;
                    best_id = row;
                }
            }
            visited_clusters += 1;
            if position + 1 < self.order.len() {
                let upper = self.upper_bounds[self.order[position + 1]];
                if upper < best as f64 {
                    break;
                }
            }
        }
        Ok((
            best_id,
            VocabularySearchStats {
                visited_clusters,
                exact_rows,
                total_rows: self.vocab.saturating_sub(first_row),
                centroid_rows: self.clusters.len(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::SimpleRng;

    #[test]
    fn retrieval_counts_follow_scan_order_and_zero_tolerance_is_bit_exact() {
        let mut bank = HyperbolicEpisodicBankV2::new(10, 2, 2);
        for _ in 0..8 {
            bank.insert(&[0.85, 0.0], &[1.0, -1.0]);
        }
        bank.insert(&[0.0, 0.0], &[0.2, 0.3]);
        let mut index = CertifiedMemoryIndex::build(&bank, 4).unwrap();
        let mut out = [0.0; 2];
        let mut weights = [0.0; 10];
        let stats = index
            .retrieve_soft_into(&bank, &[0.0, 0.0], 0.05, 0.01, &mut out, &mut weights)
            .unwrap();
        assert!(stats.certified);
        assert_eq!(stats.scanned_slots, 1); // final short cluster visited first
        assert_eq!(stats.value_rows_mixed, 1);
        assert_eq!(stats.center_distance_evaluations, 0);
        assert_eq!(stats.cluster_bound_evaluations, 3);
        let exact = index
            .retrieve_soft_into(&bank, &[0.0, 0.0], 0.05, 0.0, &mut out, &mut weights)
            .unwrap();
        let mut reference = [0.0; 2];
        let mut reference_weights = [0.0; 10];
        bank.retrieve_soft_into(&[0.0, 0.0], 0.05, &mut reference, &mut reference_weights);
        assert!(exact.exact_fallback);
        assert_eq!(out, reference);
        assert_eq!(weights, reference_weights);
    }

    #[test]
    fn vocabulary_matches_simd_dense_head_across_paired_random_queries_and_ties() {
        for seed in 1..=5 {
            let mut rng = SimpleRng::new(seed);
            let mut weights = ParamMatrix::random_xavier(128, 64, &mut rng);
            weights.data[64..128].copy_from_slice(&vec![0.25; 64]);
            weights.data[128..192].copy_from_slice(&vec![0.25; 64]);
            let mut index = CertifiedVocabularyIndex::build(&weights, 16).unwrap();
            let mut logits = vec![0.0; 128];
            for _ in 0..32 {
                let z: Vec<_> = (0..64).map(|_| rng.gen_range_f32(-1.0, 1.0)).collect();
                weights.matvec(&z, &mut logits);
                for logit in &mut logits {
                    *logit *= 1.0 / 8.0;
                }
                let expected = (1..128)
                    .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
                    .unwrap();
                let (actual, stats) = index.exact_greedy_from(&weights, &z, 1).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(stats.centroid_rows, 16);
                assert_eq!(stats.certified(), !stats.exact_fallback());
            }
        }
    }

    #[test]
    fn csr_matches_full_read_and_certifies_separated_clusters() {
        let mut bank = HyperbolicEpisodicBankV2::new(8, 2, 2);
        for (key, value) in [
            ([0.05, 0.0], [1.0, 0.0]),
            ([0.06, 0.0], [1.0, 0.0]),
            ([0.70, 0.0], [0.0, 1.0]),
            ([0.71, 0.0], [0.0, 1.0]),
        ] {
            bank.insert(
                &HyperbolicEpisodicBankV2::diffeomorphic_projected(&key),
                &value,
            );
        }
        let query = HyperbolicEpisodicBankV2::diffeomorphic_projected(&[0.05, 0.0]);
        let mut index = CertifiedMemoryIndex::build(&bank, 2).unwrap();
        let mut sparse = [0.0; 2];
        let mut sparse_weights = [0.0; 8];
        let stats = index
            .retrieve_soft_into(&bank, &query, 0.05, 0.01, &mut sparse, &mut sparse_weights)
            .unwrap();
        let mut full = [0.0; 2];
        let mut full_weights = [0.0; 8];
        bank.retrieve_soft_into(&query, 0.05, &mut full, &mut full_weights);
        assert!(stats.certified);
        assert!(stats.scanned_slots < bank.count);
        assert!(sparse.iter().zip(full).all(|(a, b)| (a - b).abs() < 0.02));
    }

    #[test]
    fn csr_falls_back_exactly_for_zero_tolerance() {
        let mut rng = SimpleRng::new(4);
        let mut bank = HyperbolicEpisodicBankV2::new(12, 3, 2);
        for _ in 0..12 {
            let raw = [
                rng.gen_range_f32(-1.0, 1.0),
                rng.gen_range_f32(-1.0, 1.0),
                rng.gen_range_f32(-1.0, 1.0),
            ];
            let mut key = [0.0; 3];
            HyperbolicEpisodicBankV2::diffeomorphic_project(&raw, &mut key);
            bank.insert(&key, &[raw[0], raw[1]]);
        }
        let mut query = [0.0; 3];
        HyperbolicEpisodicBankV2::diffeomorphic_project(&[0.2, -0.1, 0.3], &mut query);
        let mut index = CertifiedMemoryIndex::build(&bank, 3).unwrap();
        let mut sparse = [0.0; 2];
        let mut sparse_weights = [0.0; 12];
        let stats = index
            .retrieve_soft_into(&bank, &query, 0.2, 0.0, &mut sparse, &mut sparse_weights)
            .unwrap();
        let mut full = [0.0; 2];
        let mut full_weights = [0.0; 12];
        bank.retrieve_soft_into(&query, 0.2, &mut full, &mut full_weights);
        assert!(stats.exact_fallback);
        assert_eq!(sparse, full);
        assert_eq!(sparse_weights, full_weights);
    }

    #[test]
    fn geometric_certificates_bound_actual_omitted_mass_for_interleaved_keys() {
        let mut bank = HyperbolicEpisodicBankV2::new(64, 2, 3);
        let centers = [[0.0, 0.0], [0.7, 0.0], [-0.4, 0.6], [0.99999, 0.0]];
        for i in 0..64 {
            let mut key = centers[i % centers.len()];
            key[1] += (i / 4) as f32 * 1e-6;
            bank.insert(&key, &[i as f32 / 64.0, -0.5, (i % 7) as f32]);
        }
        let mut certificates = 0;
        for query in centers {
            for tau in [0.04, 0.3, 10.0, f32::MIN_POSITIVE] {
                let mut index = CertifiedMemoryIndex::build(&bank, 8).unwrap();
                let (mut sparse, mut full) = ([0.0; 3], [0.0; 3]);
                let (mut weights, mut reference) = ([0.0; 64], [0.0; 64]);
                let stats = index
                    .retrieve_soft_into(&bank, &query, tau, 0.01, &mut sparse, &mut weights)
                    .unwrap();
                bank.retrieve_soft_into(&query, tau, &mut full, &mut reference);
                if stats.certified {
                    certificates += 1;
                    let omitted: f64 = reference
                        .iter()
                        .zip(weights)
                        .filter(|(_, sparse)| *sparse == 0.0)
                        .map(|(&full, _)| full as f64)
                        .sum();
                    assert!(
                        omitted <= stats.omitted_mass_bound as f64 + 1e-7,
                        "omitted {omitted} exceeds {:?}",
                        stats
                    );
                    for (&a, b) in sparse.iter().zip(full) {
                        assert!((a - b).abs() <= 12.0 * stats.omitted_mass_bound + 2e-5);
                    }
                } else {
                    assert_eq!(sparse, full);
                    assert_eq!(weights, reference);
                }
            }
        }
        assert!(certificates >= 4);
    }

    #[test]
    fn costly_routing_backs_off_exactly_then_retries_a_changed_query() {
        let mut bank = HyperbolicEpisodicBankV2::new(64, 2, 2);
        for i in 0..64 {
            bank.insert(
                &[if i % 2 == 0 { 0.0 } else { 0.8 }, 0.0],
                &[i as f32, -1.0],
            );
        }
        let mut index = CertifiedMemoryIndex::build(&bank, 8).unwrap();
        let (mut out, mut weights) = ([0.0; 2], [0.0; 64]);
        let (mut reference, mut full_weights) = ([0.0; 2], [0.0; 64]);
        bank.retrieve_soft_into(&[0.5, 0.0], 0.05, &mut reference, &mut full_weights);
        for _ in 0..3 {
            let stats = index
                .retrieve_soft_into(&bank, &[0.5, 0.0], 0.05, 0.01, &mut out, &mut weights)
                .unwrap();
            assert!(stats.exact_fallback && stats.budget_fallback);
            assert_eq!(out, reference);
            assert_eq!(weights, full_weights);
        }
        for _ in 0..31 {
            let stats = index
                .retrieve_soft_into(&bank, &[0.5, 0.0], 0.05, 0.01, &mut out, &mut weights)
                .unwrap();
            assert!(stats.exact_fallback && stats.routing_bypassed);
            assert_eq!(stats.cluster_bound_evaluations, 0);
            assert_eq!(out, reference);
        }
        let stats = index
            .retrieve_soft_into(&bank, &[0.0, 0.0], 0.05, 0.01, &mut out, &mut weights)
            .unwrap();
        assert!(stats.certified && !stats.routing_bypassed);
        assert_eq!(stats.scanned_slots, 32);
    }

    #[test]
    fn cvp_greedy_is_exact_and_can_prune_rows() {
        let mut weights = ParamMatrix::zeros(8, 2);
        for row in 0..8 {
            weights.data[row * 2] = if row == 3 { 4.0 } else { -1.0 };
            weights.data[row * 2 + 1] = row as f32 * 0.001;
        }
        let mut index = CertifiedVocabularyIndex::build(&weights, 4).unwrap();
        let (id, stats) = index.exact_greedy(&weights, &[1.0, 0.0]).unwrap();
        assert_eq!(id, 3);
        assert!(stats.exact_rows < stats.total_rows);
    }
}

// Kept private to the tests above so production callers use the public
// projection API without introducing another allocation helper.
#[cfg(test)]
trait ProjectedKey {
    fn diffeomorphic_projected(raw: &[f32]) -> Vec<f32>;
}
#[cfg(test)]
impl ProjectedKey for HyperbolicEpisodicBankV2 {
    fn diffeomorphic_projected(raw: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; raw.len()];
        Self::diffeomorphic_project(raw, &mut out);
        out
    }
}
