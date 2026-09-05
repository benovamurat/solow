//! Affinity Propagation (Frey-Dueck 2007).
//!
//! Iteratively passes "responsibility" and "availability" messages over
//! a similarity graph until a stable set of exemplars emerges.

use ndarray::ArrayView2;
use solow_core::{Error, Result};

/// Fitted AffinityPropagation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct AffinityPropagation {
    /// Cluster labels (one per input row).
    pub labels: Vec<i64>,
    /// Exemplar indices for each cluster.
    pub cluster_centers_indices: Vec<usize>,
    /// Iterations run.
    pub n_iter: usize,
    /// Whether the fixed point was reached before `max_iter`.
    pub converged: bool,
    /// Damping factor used.
    pub damping: f64,
}

impl AffinityPropagation {
    /// Fit with the reference defaults `damping = 0.5`, `max_iter = 200`,
    /// `convergence_iter = 15`, and preference = median similarity.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 0.5, 200, 15, None)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        damping: f64,
        max_iter: usize,
        convergence_iter: usize,
        preference: Option<f64>,
    ) -> Result<Self> {
        let n = x.nrows();
        if n == 0 {
            return Err(Error::Value("AffinityPropagation: empty input".into()));
        }
        if !(0.5..1.0).contains(&damping) {
            return Err(Error::Value(format!(
                "AffinityPropagation: damping must be in [0.5, 1) (got {damping})"
            )));
        }
        let d = x.ncols();
        // Similarity = -‖xᵢ − xⱼ‖² (negative squared Euclidean, the reference default).
        let mut s = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let mut acc = 0.0_f64;
                for k in 0..d {
                    let e = x[[i, k]] - x[[j, k]];
                    acc += e * e;
                }
                s[i][j] = -acc;
                s[j][i] = -acc;
            }
        }
        let pref = preference.unwrap_or_else(|| {
            let mut off: Vec<f64> = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    off.push(s[i][j]);
                }
            }
            off.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if off.is_empty() { 0.0 } else { off[off.len() / 2] }
        });
        for i in 0..n {
            s[i][i] = pref;
        }
        let mut r = vec![vec![0.0_f64; n]; n];
        let mut a = vec![vec![0.0_f64; n]; n];
        let mut ex_history: Vec<Vec<bool>> = Vec::new();
        let mut used = 0_usize;
        let mut converged = false;
        for it in 0..max_iter {
            used = it + 1;
            // Responsibility update: r(i, k) ← s(i, k) − max_{k' ≠ k} (a(i, k') + s(i, k'))
            let mut r_new = vec![vec![0.0_f64; n]; n];
            for i in 0..n {
                let mut m1 = f64::NEG_INFINITY;
                let mut m2 = f64::NEG_INFINITY;
                let mut m1_idx = 0_usize;
                for k in 0..n {
                    let val = a[i][k] + s[i][k];
                    if val > m1 {
                        m2 = m1;
                        m1 = val;
                        m1_idx = k;
                    } else if val > m2 {
                        m2 = val;
                    }
                }
                for k in 0..n {
                    let max_other = if k == m1_idx { m2 } else { m1 };
                    r_new[i][k] = s[i][k] - max_other;
                }
            }
            for i in 0..n {
                for k in 0..n {
                    r[i][k] = damping * r[i][k] + (1.0 - damping) * r_new[i][k];
                }
            }
            // Availability update
            let mut a_new = vec![vec![0.0_f64; n]; n];
            for k in 0..n {
                let mut col_sum_pos = 0.0_f64;
                for i in 0..n {
                    if i != k {
                        col_sum_pos += r[i][k].max(0.0);
                    }
                }
                a_new[k][k] = col_sum_pos;
                for i in 0..n {
                    if i == k {
                        continue;
                    }
                    let mut inner = r[k][k] + col_sum_pos - r[i][k].max(0.0);
                    inner = inner.min(0.0);
                    a_new[i][k] = inner;
                }
            }
            for i in 0..n {
                for k in 0..n {
                    a[i][k] = damping * a[i][k] + (1.0 - damping) * a_new[i][k];
                }
            }
            // Extract exemplars.
            let ex: Vec<bool> = (0..n).map(|i| a[i][i] + r[i][i] > 0.0).collect();
            ex_history.push(ex.clone());
            if ex_history.len() > convergence_iter {
                ex_history.remove(0);
            }
            if ex_history.len() == convergence_iter {
                let stable = ex_history
                    .iter()
                    .all(|snap| snap == ex_history.last().unwrap());
                if stable {
                    converged = true;
                    break;
                }
            }
        }
        let exemplars: Vec<usize> = (0..n).filter(|&i| a[i][i] + r[i][i] > 0.0).collect();
        let labels: Vec<i64> = (0..n)
            .map(|i| {
                if exemplars.is_empty() {
                    -1
                } else {
                    let mut best = 0;
                    let mut best_s = f64::NEG_INFINITY;
                    for (k, &e) in exemplars.iter().enumerate() {
                        if s[i][e] > best_s {
                            best_s = s[i][e];
                            best = k;
                        }
                    }
                    best as i64
                }
            })
            .collect();
        Ok(Self {
            labels,
            cluster_centers_indices: exemplars,
            n_iter: used,
            converged,
            damping,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn affinity_propagation_finds_a_positive_number_of_exemplars() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.05, 0.15],
            [5.0, 5.0], [5.1, 5.1], [5.05, 5.15]
        ];
        let m = AffinityPropagation::fit(x.view()).unwrap();
        assert!(!m.cluster_centers_indices.is_empty());
    }
}
