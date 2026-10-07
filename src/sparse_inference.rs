//! Exact, certificate-driven sparse inference helpers.
//!
//! The index is deliberately small and conservative.  Memory clusters use an
//! existing key as their centre, so their radius is measured directly and can
//! never be optimistic.  Vocabulary clusters store a centroid and the maximum
//! row deviation.  Both readers fall back to exhaustive evaluation whenever a
//! requested certificate cannot be proved.

use crate::memory::HyperbolicEpisodicBankV2;
use crate::pssa::ParamMatrix;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseReadStats {
    pub scanned_slots: usize,
    pub populated_slots: usize,
    pub omitted_mass_bound: f32,
    pub certified: bool,
    pub exact_fallback: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct MemoryCluster {
    center: Vec<f32>,
    radius: f32,
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
                start,
                end,
            });
        }
        let order = (0..clusters.len()).collect();
        Ok(Self {
            dim_key: bank.dim_key,
            count: bank.count,
            clusters,
            order,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
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
        self.order.sort_unstable_by(|&a, &b| {
            let lower = |cluster: &MemoryCluster| {
                HyperbolicEpisodicBankV2::poincare_distance(
                    q_pnc,
                    q_sq,
                    &cluster.center,
                    HyperbolicEpisodicBankV2::squared_norm(&cluster.center),
                )
                .max(0.0)
                    - cluster.radius
            };
            lower(&self.clusters[a])
                .max(0.0)
                .total_cmp(&lower(&self.clusters[b]).max(0.0))
                .then_with(|| a.cmp(&b))
        });

        let mut scanned_clusters = 0usize;
        let mut min_scanned = f32::INFINITY;
        let mut omitted_bound = 1.0f32;
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
                min_scanned = min_scanned.min(dist);
            }
            scanned_clusters += 1;

            // Shift both scanned and unscanned terms by the nearest scanned
            // distance.  This is the same stable softmax as the full reader,
            // with a conservative upper bound for every unscanned cluster.
            let mut scanned_mass = 0.0f64;
            for &dist in &out_weights[..bank.count] {
                if dist.is_finite() {
                    scanned_mass += ((min_scanned - dist) / tau).exp() as f64;
                }
            }
            let mut unscanned_mass = 0.0f64;
            for &other_id in &self.order[scanned_clusters..] {
                let other = &self.clusters[other_id];
                let center_dist = HyperbolicEpisodicBankV2::poincare_distance(
                    q_pnc,
                    q_sq,
                    &other.center,
                    HyperbolicEpisodicBankV2::squared_norm(&other.center),
                );
                let lower = (center_dist - other.radius).max(0.0);
                unscanned_mass +=
                    (other.end - other.start) as f64 * ((min_scanned - lower) / tau).exp() as f64;
            }
            omitted_bound = if unscanned_mass == 0.0 {
                0.0
            } else {
                (unscanned_mass / (scanned_mass + unscanned_mass)) as f32
            };
            if omitted_bound <= epsilon {
                break;
            }
        }

        let certified = omitted_bound <= epsilon && scanned_clusters < self.clusters.len();
        let exact_fallback = !certified;
        if exact_fallback {
            // Preserve the production reader's accumulation order and bits on
            // fallback.  This is also the safety valve for diffuse banks.
            bank.retrieve_soft_into(q_pnc, tau, out_val, out_weights);
            return Ok(SparseReadStats {
                scanned_slots: bank.count,
                populated_slots: bank.count,
                omitted_mass_bound: 0.0,
                certified: false,
                exact_fallback: true,
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
        for idx in 0..bank.count {
            let weight = (out_weights[idx] as f64 / sum) as f32;
            out_weights[idx] = weight;
            let off = idx * bank.dim_val;
            for j in 0..bank.dim_val {
                out_val[j] += weight * bank.values[off + j];
            }
        }
        out_weights[bank.count..].fill(0.0);
        Ok(SparseReadStats {
            scanned_slots: if exact_fallback {
                bank.count
            } else {
                self.clusters[..scanned_clusters]
                    .iter()
                    .map(|c| c.end - c.start)
                    .sum()
            },
            populated_slots: bank.count,
            omitted_mass_bound: omitted_bound,
            certified,
            exact_fallback,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct VocabularyCluster {
    centroid: Vec<f32>,
    radius: f32,
    rows: Vec<usize>,
}

/// Branch-and-bound index for exact greedy output projection.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedVocabularyIndex {
    dim: usize,
    vocab: usize,
    clusters: Vec<VocabularyCluster>,
    order: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VocabularySearchStats {
    pub visited_clusters: usize,
    pub exact_rows: usize,
    pub total_rows: usize,
}

impl CertifiedVocabularyIndex {
    pub fn build(weights: &ParamMatrix, cluster_count: usize) -> Result<Self, String> {
        if weights.rows == 0 || weights.cols == 0 || cluster_count == 0 {
            return Err("vocabulary index dimensions and cluster count must be positive".into());
        }
        let clusters_n = cluster_count.min(weights.rows);
        let rows_per_cluster = weights.rows.div_ceil(clusters_n);
        let mut clusters = Vec::new();
        for start in (0..weights.rows).step_by(rows_per_cluster) {
            let end = (start + rows_per_cluster).min(weights.rows);
            let mut centroid = vec![0.0f32; weights.cols];
            for row in start..end {
                for (dst, &x) in centroid
                    .iter_mut()
                    .zip(&weights.data[row * weights.cols..(row + 1) * weights.cols])
                {
                    *dst += x;
                }
            }
            let inv = 1.0 / (end - start) as f32;
            for x in &mut centroid {
                *x *= inv;
            }
            let mut radius = 0.0f32;
            for row in start..end {
                let row_data = &weights.data[row * weights.cols..(row + 1) * weights.cols];
                let distance = row_data
                    .iter()
                    .zip(&centroid)
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                radius = radius.max(distance);
            }
            clusters.push(VocabularyCluster {
                centroid,
                radius,
                rows: (start..end).collect(),
            });
        }
        Ok(Self {
            dim: weights.cols,
            vocab: weights.rows,
            order: (0..clusters.len()).collect(),
            clusters,
        })
    }

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
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
        let norm = z.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = 1.0 / (self.dim as f32).sqrt();
        self.order.sort_unstable_by(|&a, &b| {
            let upper = |cluster: &VocabularyCluster| {
                (cluster
                    .centroid
                    .iter()
                    .zip(z)
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    + cluster.radius * norm)
                    * scale
            };
            upper(&self.clusters[b])
                .total_cmp(&upper(&self.clusters[a]))
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
                let score = weights.data[row * self.dim..(row + 1) * self.dim]
                    .iter()
                    .zip(z)
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    * scale;
                exact_rows += 1;
                if score > best || (score == best && row < best_id) {
                    best = score;
                    best_id = row;
                }
            }
            visited_clusters += 1;
            if position + 1 < self.order.len() {
                let next = &self.clusters[self.order[position + 1]];
                let upper = (next
                    .centroid
                    .iter()
                    .zip(z)
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    + next.radius * norm)
                    * scale;
                if upper < best {
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
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::SimpleRng;

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
