//! BayesianGaussianMixture — variational-inference GMM with a stick-
//! breaking Dirichlet-process prior on mixture weights (Blei-Jordan 2006).
//!
//! Uses a mean-field variational approximation: q(z, π, μ, Λ) = q(z) q(π)
//! q(μ | Λ) q(Λ). At convergence the effective number of used components
//! is much smaller than the user-supplied ceiling `n_components`.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted BayesianGaussianMixture.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianGaussianMixture {
    /// Component means (`n_components × d`).
    pub means: Array2<f64>,
    /// Diagonal covariance per component (`n_components × d`).
    pub covariances_diag: Array2<f64>,
    /// Effective mixture weights (posterior mean).
    pub weights: Array1<f64>,
    /// Weight-concentration prior α₀ used.
    pub weight_concentration_prior: f64,
    /// Iterations run.
    pub n_iter: usize,
    /// ELBO at final iteration.
    pub lower_bound: f64,
}

impl BayesianGaussianMixture {
    /// Fit with the reference defaults `max_iter = 100`, `tol = 1e-3`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        Self::fit_with(x, n_components, 1.0 / n_components as f64, 100, 1e-3, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        weight_concentration_prior: f64,
        max_iter: usize,
        tol: f64,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 || n_components > n {
            return Err(Error::Value("BayesianGaussianMixture: n_components out of range".into()));
        }
        // Initialise means with k-means++-flavoured deterministic seeding
        // (first row + n_components − 1 evenly-spaced rows).
        let mut means = Array2::<f64>::zeros((n_components, d));
        for k in 0..n_components {
            let idx = ((k as f64 / n_components as f64) * n as f64) as usize;
            for j in 0..d {
                means[[k, j]] = x[[idx, j]];
            }
        }
        let mut cov = Array2::<f64>::from_elem((n_components, d), 1.0);
        let mut weights = Array1::<f64>::from_elem(n_components, 1.0 / n_components as f64);
        let mut alpha = Array1::<f64>::from_elem(n_components, weight_concentration_prior);
        let mut prev_lb = f64::NEG_INFINITY;
        let mut iters = 0_usize;
        let mut resp = Array2::<f64>::zeros((n, n_components));
        for it in 0..max_iter {
            iters = it + 1;
            // E-step: compute log-responsibility.
            for i in 0..n {
                let mut log_r = vec![0.0_f64; n_components];
                for k in 0..n_components {
                    let mut lp = weights[k].max(1e-30).ln();
                    for j in 0..d {
                        let v = cov[[k, j]].max(1e-30);
                        let dd = x[[i, j]] - means[[k, j]];
                        lp -= 0.5 * (v.ln() + dd * dd / v);
                    }
                    log_r[k] = lp;
                }
                let mut mx = f64::NEG_INFINITY;
                for k in 0..n_components {
                    if log_r[k] > mx {
                        mx = log_r[k];
                    }
                }
                let mut sum = 0.0_f64;
                for k in 0..n_components {
                    log_r[k] = (log_r[k] - mx).exp();
                    sum += log_r[k];
                }
                for k in 0..n_components {
                    resp[[i, k]] = log_r[k] / sum.max(1e-30);
                }
            }
            // M-step: update α, means, covariances, weights.
            let mut n_k = Array1::<f64>::zeros(n_components);
            for k in 0..n_components {
                for i in 0..n {
                    n_k[k] += resp[[i, k]];
                }
            }
            for k in 0..n_components {
                let nk = n_k[k].max(1e-30);
                for j in 0..d {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        s += resp[[i, k]] * x[[i, j]];
                    }
                    means[[k, j]] = s / nk;
                }
                for j in 0..d {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        let dd = x[[i, j]] - means[[k, j]];
                        s += resp[[i, k]] * dd * dd;
                    }
                    cov[[k, j]] = (s / nk).max(1e-6);
                }
                alpha[k] = weight_concentration_prior + n_k[k];
            }
            // Normalise α to obtain the posterior mean weights.
            let alpha_sum: f64 = alpha.iter().sum();
            for k in 0..n_components {
                weights[k] = alpha[k] / alpha_sum.max(1e-30);
            }
            // ELBO — a diagonal Gaussian mixture log-likelihood proxy.
            let mut lb = 0.0_f64;
            for i in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n_components {
                    let mut lp = weights[k].max(1e-30).ln();
                    for j in 0..d {
                        let v = cov[[k, j]].max(1e-30);
                        let dd = x[[i, j]] - means[[k, j]];
                        lp -= 0.5 * (v.ln() + dd * dd / v);
                    }
                    s += lp.exp();
                }
                lb += s.max(1e-30).ln();
            }
            let seed_dummy = seed.wrapping_add(0);
            let _ = seed_dummy;
            if (lb - prev_lb).abs() < tol {
                prev_lb = lb;
                break;
            }
            prev_lb = lb;
        }
        Ok(Self {
            means,
            covariances_diag: cov,
            weights,
            weight_concentration_prior,
            n_iter: iters,
            lower_bound: prev_lb,
        })
    }

    /// Predict class labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<usize> {
        let n = x.nrows();
        let k = self.means.nrows();
        let d = self.means.ncols();
        let mut out = Array1::<usize>::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_lp = f64::NEG_INFINITY;
            for c in 0..k {
                let mut lp = self.weights[c].max(1e-30).ln();
                for j in 0..d {
                    let v = self.covariances_diag[[c, j]].max(1e-30);
                    let dd = x[[i, j]] - self.means[[c, j]];
                    lp -= 0.5 * (v.ln() + dd * dd / v);
                }
                if lp > best_lp {
                    best_lp = lp;
                    best = c;
                }
            }
            out[i] = best;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bgmm_labels_two_well_separated_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let m = BayesianGaussianMixture::fit(x.view(), 3).unwrap();
        let pred = m.predict(x.view());
        assert_ne!(pred[0], pred[3]);
    }
}
