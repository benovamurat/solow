//! t-Distributed Stochastic Neighbour Embedding (van der Maaten-Hinton 2008).
//!
//! Full-matrix O(n²) implementation with the classical
//! perplexity-based Gaussian kernel in the high-dimensional space
//! and the Student-t kernel in the low-dimensional embedding.
//!
//! Gradient descent with momentum and the standard early-exaggeration
//! phase (first 250 iterations multiply high-dim probabilities by 12).
//! Deterministic under a caller-supplied seed.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Fitted t-SNE embedding.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Tsne {
    /// The embedding, `n × n_components`.
    pub embedding: Array2<f64>,
    /// Perplexity target used.
    pub perplexity: f64,
    /// Number of gradient-descent iterations run.
    pub n_iter: usize,
}

impl Tsne {
    /// Fit t-SNE with the classical defaults (`perplexity = 30`,
    /// `learning_rate = 200`, `n_iter = 1000`).
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize, seed: u64) -> Result<Self> {
        Self::fit_with(x, n_components, 30.0, 200.0, 1000, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() < 2 || x.ncols() == 0 {
            return Err(Error::Value("Tsne::fit_with: need ≥ 2 samples".into()));
        }
        if !(perplexity > 1.0 && perplexity.is_finite()) {
            return Err(Error::Value(format!(
                "Tsne::fit_with: perplexity must be > 1 (got {perplexity})"
            )));
        }
        if n_components < 1 {
            return Err(Error::Value(
                "Tsne::fit_with: n_components must be ≥ 1".into(),
            ));
        }
        let n = x.nrows();
        // High-dimensional pairwise squared distances.
        let mut sqd = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let mut s = 0.0_f64;
                for k in 0..x.ncols() {
                    let dd = x[[i, k]] - x[[j, k]];
                    s += dd * dd;
                }
                sqd[[i, j]] = s;
                sqd[[j, i]] = s;
            }
        }
        // Per-row Gaussian bandwidth by binary search on perplexity.
        let mut p = Array2::<f64>::zeros((n, n));
        let target = (perplexity).ln();
        for i in 0..n {
            let (mut beta_lo, mut beta_hi) = (1e-20_f64, f64::INFINITY);
            let mut beta = 1.0_f64;
            for _ in 0..50 {
                let (h, row_p) = row_gaussian(&sqd, i, beta);
                let h_diff = h - target;
                if h_diff.abs() < 1e-5 {
                    for j in 0..n {
                        p[[i, j]] = row_p[j];
                    }
                    break;
                }
                if h_diff > 0.0 {
                    beta_lo = beta;
                    beta = if beta_hi.is_finite() {
                        0.5 * (beta_lo + beta_hi)
                    } else {
                        beta * 2.0
                    };
                } else {
                    beta_hi = beta;
                    beta = 0.5 * (beta_lo + beta_hi);
                }
                for j in 0..n {
                    p[[i, j]] = row_p[j];
                }
            }
        }
        // Symmetrise and normalise.
        let mut p_sym = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                p_sym[[i, j]] = 0.5 * (p[[i, j]] + p[[j, i]]);
            }
        }
        let sum: f64 = p_sym.iter().sum();
        if sum > 0.0 {
            for v in p_sym.iter_mut() {
                *v /= sum;
                *v = v.max(1e-12);
            }
        }
        // Initialise Y ~ small Gaussian.
        let mut state = seed.wrapping_add(0xF00D_BABE_FACE);
        let mut y = Array2::<f64>::zeros((n, n_components));
        for i in 0..n {
            for c in 0..n_components {
                y[[i, c]] = 1e-4 * (uniform_f64(&mut state) - 0.5);
            }
        }
        let mut y_prev = y.clone();
        // Gradient descent.
        for iter in 0..n_iter {
            let ex = if iter < 250 { 12.0 } else { 1.0 };
            let mom = if iter < 250 { 0.5 } else { 0.8 };
            // Low-dim Q.
            let mut q_sqd = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut s = 0.0_f64;
                    for c in 0..n_components {
                        let dd = y[[i, c]] - y[[j, c]];
                        s += dd * dd;
                    }
                    let val = 1.0 / (1.0 + s);
                    q_sqd[[i, j]] = val;
                    q_sqd[[j, i]] = val;
                }
            }
            let q_sum: f64 = q_sqd.iter().sum();
            let z = q_sum.max(1e-12);
            // Gradient.
            let mut grad = Array2::<f64>::zeros(y.dim());
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let qij = q_sqd[[i, j]] / z;
                    let pij = ex * p_sym[[i, j]];
                    let mult = 4.0 * (pij - qij) * q_sqd[[i, j]];
                    for c in 0..n_components {
                        grad[[i, c]] += mult * (y[[i, c]] - y[[j, c]]);
                    }
                }
            }
            // Update with momentum.
            let mut y_new = Array2::<f64>::zeros(y.dim());
            for i in 0..n {
                for c in 0..n_components {
                    y_new[[i, c]] = y[[i, c]] - learning_rate * grad[[i, c]]
                        + mom * (y[[i, c]] - y_prev[[i, c]]);
                }
            }
            y_prev = y;
            y = y_new;
        }
        Ok(Self {
            embedding: y,
            perplexity,
            n_iter,
        })
    }
}

fn row_gaussian(sqd: &Array2<f64>, i: usize, beta: f64) -> (f64, Vec<f64>) {
    let n = sqd.nrows();
    let mut row = vec![0.0_f64; n];
    let mut s = 0.0_f64;
    for j in 0..n {
        if j == i {
            continue;
        }
        row[j] = (-beta * sqd[[i, j]]).exp();
        s += row[j];
    }
    if s > 0.0 {
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    // Shannon entropy of the row.
    let mut h = 0.0_f64;
    for &v in &row {
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    (h, row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsne_produces_finite_2d_embedding() {
        let n = 25usize;
        let mut rows: Vec<[f64; 3]> = Vec::new();
        for i in 0..n {
            let t = i as f64 * 0.3;
            rows.push([t.cos(), t.sin(), 0.05 * t]);
        }
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = Array2::from_shape_vec((n, 3), flat).unwrap();
        let out = Tsne::fit_with(x.view(), 2, 5.0, 100.0, 300, 42).unwrap();
        assert_eq!(out.embedding.dim(), (n, 2));
        for v in out.embedding.iter() {
            assert!(v.is_finite(), "non-finite entry {v}");
        }
    }
}
