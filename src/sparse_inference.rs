//! Exact, certificate-driven sparse inference helpers.
//!
//! The index is deliberately small and conservative.  Memory clusters use an
//! existing key as their centre, so their radius is measured directly and can
//! never be optimistic.  Vocabulary clusters store a centroid and the maximum
//! row deviation.  Both readers fall back to exhaustive evaluation whenever a
//! requested certificate cannot be proved.

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
}

#[derive(Clone, Debug, PartialEq)]
struct MemoryCluster {
    center: Vec<f32>,
    radius: f32,
    center_sq: f32,
    start: usize,
    end: usize,
}

/// A conservative metric-ball index over the currently occupied memory slots.
/// The index is runtime-only and must be rebuilt after a memory write.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedMemoryIndex {
    dim_key: usize,
    count: usize,
    clusters: Vec<MemoryCluster>,
    order: Vec<usize>,
    lower_bounds: Vec<f64>,
}

impl CertifiedMemoryIndex {
    pub fn build(bank: &HyperbolicEpisodicBankV2, cluster_size: usize) -> Result<Self, String> {
        if cluster_size == 0 {
            return Err("certified memory cluster size must be positive".into());
        }
        let mut clusters = Vec::new();
        for start in (0..bank.count).step_by(cluster_size) {
            let end = (start + cluster_size).min(bank.count);
            // A stored key is a valid centre.  This costs no geometric mean
            // calculation and makes the radius certificate straightforward.
            let center = bank.keys[start * bank.dim_key..(start + 1) * bank.dim_key].to_vec();
            let center_sq = bank.norm_sq[start];
            let mut radius = 0.0f32;
            for idx in start..end {
                let off = idx * bank.dim_key;
                radius = radius.max(HyperbolicEpisodicBankV2::poincare_distance(
                    &center,
                    center_sq,
                    &bank.keys[off..off + bank.dim_key],
                    bank.norm_sq[idx],
                ));
            }
            clusters.push(MemoryCluster {
                center,
                radius,
                center_sq,
                start,
                end,
            });
        }
        let order = (0..clusters.len()).collect();
        let lower_bounds = vec![0.0; clusters.len()];
        Ok(Self {
            dim_key: bank.dim_key,
            count: bank.count,
            clusters,
            order,
            lower_bounds,
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
            .map(|c| c.center.capacity() * 4)
            .sum::<usize>()
            + self.order.capacity() * std::mem::size_of::<usize>()
            + self.lower_bounds.capacity() * 8
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
        out_weights[..bank.count].fill(f32::NAN);
        for (bound, cluster) in self.lower_bounds.iter_mut().zip(&self.clusters) {
            let distance = HyperbolicEpisodicBankV2::poincare_distance(
                q_pnc,
                q_sq,
                &cluster.center,
                cluster.center_sq,
            ) as f64;
            let guard = 8.0 * f32::EPSILON as f64 * (distance.abs() + cluster.radius as f64 + 1.0);
            *bound = (distance - cluster.radius as f64 - guard).max(0.0);
        }
        self.order.sort_unstable_by(|&a, &b| {
            self.lower_bounds[a]
                .total_cmp(&self.lower_bounds[b])
                .then_with(|| a.cmp(&b))
        });

        let mut scanned_clusters = 0usize;
        let mut scanned_slots = 0usize;
        let mut min_scanned = f32::INFINITY;
        let mut omitted_bound = 1.0f32;
        let mut scanned_mass = 0.0f64;
        for &cluster_id in &self.order {
            let cluster = &self.clusters[cluster_id];
            for idx in cluster.start..cluster.end {
                let off = idx * bank.dim_key;
                let dist = HyperbolicEpisodicBankV2::poincare_distance(
                    q_pnc,
                    q_sq,
                    &bank.keys[off..off + bank.dim_key],
                    bank.norm_sq[idx],
                );
                out_weights[idx] = dist;
                if dist < min_scanned {
                    scanned_mass *= ((dist as f64 - min_scanned as f64) / tau as f64).exp();
                    min_scanned = dist;
                }
                scanned_mass += ((min_scanned as f64 - dist as f64) / tau as f64).exp();
                scanned_slots += 1;
            }
            scanned_clusters += 1;

            // Shift both scanned and unscanned terms by the nearest scanned
            // distance.  This is the same stable softmax as the full reader,
            // with a conservative upper bound for every unscanned cluster.
            let mut unscanned_mass = 0.0f64;
            for &other_id in &self.order[scanned_clusters..] {
                let other = &self.clusters[other_id];
                unscanned_mass += (other.end - other.start) as f64
                    * ((min_scanned as f64 - self.lower_bounds[other_id]) / tau as f64).exp();
            }
            omitted_bound = if unscanned_mass == 0.0 {
                0.0
            } else if !unscanned_mass.is_finite() {
                1.0
            } else {
                (unscanned_mass / (scanned_mass + unscanned_mass) + 8.0 * f32::EPSILON as f64)
                    .min(1.0) as f32
            };
            if epsilon > 0.0 && omitted_bound <= epsilon {
                break;
            }
        }

        let certified =
            epsilon > 0.0 && omitted_bound <= epsilon && scanned_clusters < self.clusters.len();
        let exact_fallback = !certified;
        if exact_fallback {
            // Preserve the production reader's accumulation order and bits on
            // fallback.  This is also the safety valve for diffuse banks.
            // Distances are already exact. Reuse them with the production
            // reader's FP32 accumulation order instead of scanning twice.
            let mut sum = 0.0f32;
            for w in &mut out_weights[..bank.count] {
                *w = ((min_scanned - *w) / tau).exp();
                sum += *w;
            }
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
                center_distance_evaluations: self.clusters.len(),
                value_rows_mixed: bank.count,
            });
        }

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
            exact_fallback,
            center_distance_evaluations: self.clusters.len(),
            value_rows_mixed,
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
        assert_eq!(stats.center_distance_evaluations, 3);
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
