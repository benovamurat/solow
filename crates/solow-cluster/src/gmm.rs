//! Gaussian mixture models fit by Expectation-Maximisation
//! (Dempster-Laird-Rubin 1977; Bishop 2006, §9.2).
//!
//! Full-covariance and diagonal-covariance parameterisations.
//! Uses `k-means++` initialisation on the means (bit-identical to
//! [`crate::kmeans::KMeansInit::KMeansPlusPlus`]).
//!
//! The log-likelihood is monitored per EM iteration and the fit
//! terminates when the improvement falls below `tol` or after
//! `max_iter` sweeps.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Covariance parameterisation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CovType {
    /// One full covariance matrix per component.
    Full,
    /// Diagonal covariance per component.
    Diag,
}

/// Fitted Gaussian mixture.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianMixture {
    /// Component weights (sum to 1).
    pub weights: Array1<f64>,
    /// Component means, `(k, d)`.
    pub means: Array2<f64>,
    /// Per-component covariance matrices — for `CovType::Diag`, only
    /// the diagonal entries are populated; the off-diagonal are 0.
    pub covariances: Vec<Array2<f64>>,
    /// Covariance parameterisation.
    pub covariance_type: CovType,
    /// Number of components.
    pub k: usize,
    /// Number of EM iterations run.
    pub n_iter: usize,
    /// Final log-likelihood.
    pub log_likelihood: f64,
    /// Whether EM converged inside `max_iter`.
    pub converged: bool,
}

impl GaussianMixture {
    /// Fit with defaults (`CovType::Full`, `max_iter = 100`, `tol = 1e-3`,
    /// `reg_covar = 1e-6`).
    pub fn fit(x: ArrayView2<'_, f64>, k: usize, seed: u64) -> Result<Self> {
        Self::fit_with(x, k, CovType::Full, 100, 1e-3, 1e-6, seed)
    }

    /// Full-configuration fit.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        k: usize,
        covariance_type: CovType,
        max_iter: usize,
        tol: f64,
        reg_covar: f64,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "GaussianMixture::fit_with: x must be non-empty".into(),
            ));
        }
        if k < 1 || k > x.nrows() {
            return Err(Error::Value(format!(
                "GaussianMixture::fit_with: k must be in [1, n] (got {k})"
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        // k-means++ init on the means.
        let mut means = kpp_init(x, k, seed)?;
        // Uniform weights.
        let mut weights = Array1::<f64>::from_elem(k, 1.0 / k as f64);
        // Global sample covariance as the initial per-component covariance.
        let sample_cov = sample_covariance(x, reg_covar);
        let mut covariances: Vec<Array2<f64>> = (0..k).map(|_| sample_cov.clone()).collect();

        let mut log_lik = f64::NEG_INFINITY;
        let mut converged = false;
        let mut iter = 0usize;
        for it in 0..max_iter {
            iter = it + 1;
            // E-step — responsibilities.
            let (resp, ll) = e_step(x, &weights, &means, &covariances, covariance_type)?;
            // M-step.
            let (new_weights, new_means, new_cov) =
                m_step(x, &resp, k, d, covariance_type, reg_covar);
            weights = new_weights;
            means = new_means;
            covariances = new_cov;
            if (ll - log_lik).abs() < tol {
                converged = true;
                log_lik = ll;
                break;
            }
            log_lik = ll;
        }
        // Suppress an unused-variable warning if the E-step is never
        // reached (impossible above but keeps the compiler happy on refactors).
        let _ = n;
        Ok(Self {
            weights,
            means,
            covariances,
            covariance_type,
            k,
            n_iter: iter,
            log_likelihood: log_lik,
            converged,
        })
    }

    /// Assign each row of `x` to its highest-responsibility component.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        let (resp, _) = e_step(
            x,
            &self.weights,
            &self.means,
            &self.covariances,
            self.covariance_type,
        )?;
        let mut out = Array1::<usize>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let (mut best_c, mut best_p) = (0usize, f64::NEG_INFINITY);
            for c in 0..self.k {
                if resp[[i, c]] > best_p {
                    best_p = resp[[i, c]];
                    best_c = c;
                }
            }
            out[i] = best_c;
        }
        Ok(out)
    }

    /// Return the row-normalised posterior responsibilities.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Ok(e_step(
            x,
            &self.weights,
            &self.means,
            &self.covariances,
            self.covariance_type,
        )?
        .0)
    }
}

// ---------------------------------------------------------------------------
// EM primitives
// ---------------------------------------------------------------------------

fn e_step(
    x: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    means: &Array2<f64>,
    cov: &[Array2<f64>],
    ctype: CovType,
) -> Result<(Array2<f64>, f64)> {
    let n = x.nrows();
    let k = weights.len();
    let d = x.ncols();
    // Per-component precomputed Cholesky + log-determinant.
    let mut chol_cache = Vec::with_capacity(k);
    let mut logdet_cache = vec![0.0_f64; k];
    for c in 0..k {
        let (l, log_det) = cholesky_and_log_det(&cov[c])?;
        chol_cache.push(l);
        logdet_cache[c] = log_det;
    }
    let mut log_prob = Array2::<f64>::zeros((n, k));
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    for i in 0..n {
        for c in 0..k {
            let mut diff = vec![0.0_f64; d];
            for j in 0..d {
                diff[j] = x[[i, j]] - means[[c, j]];
            }
            let mah = mahalanobis_via_chol(&chol_cache[c], &diff);
            log_prob[[i, c]] =
                -0.5 * (d as f64 * ln_2pi + logdet_cache[c] + mah) + weights[c].max(1e-300).ln();
        }
    }
    // Row log-sum-exp.
    let mut log_lik = 0.0_f64;
    let mut resp = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let m = log_prob
            .row(i)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0.0_f64;
        for c in 0..k {
            let e = (log_prob[[i, c]] - m).exp();
            resp[[i, c]] = e;
            s += e;
        }
        let log_z = m + s.ln();
        for c in 0..k {
            resp[[i, c]] /= s;
        }
        log_lik += log_z;
    }
    let _ = ctype;
    Ok((resp, log_lik))
}

fn m_step(
    x: ArrayView2<'_, f64>,
    resp: &Array2<f64>,
    k: usize,
    d: usize,
    ctype: CovType,
    reg_covar: f64,
) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
    let n = x.nrows();
    // Nk = Σᵢ resp[i, c].
    let mut nk = Array1::<f64>::zeros(k);
    for c in 0..k {
        for i in 0..n {
            nk[c] += resp[[i, c]];
        }
        if nk[c] < 1e-12 {
            nk[c] = 1e-12;
        }
    }
    let weights = nk.mapv(|v| v / n as f64);
    // Means.
    let mut means = Array2::<f64>::zeros((k, d));
    for c in 0..k {
        for i in 0..n {
            for j in 0..d {
                means[[c, j]] += resp[[i, c]] * x[[i, j]];
            }
        }
        for j in 0..d {
            means[[c, j]] /= nk[c];
        }
    }
    // Covariances.
    let mut covs: Vec<Array2<f64>> = Vec::with_capacity(k);
    for c in 0..k {
        let mut cov = Array2::<f64>::zeros((d, d));
        for i in 0..n {
            let mut diff = vec![0.0_f64; d];
            for j in 0..d {
                diff[j] = x[[i, j]] - means[[c, j]];
            }
            match ctype {
                CovType::Full => {
                    for j in 0..d {
                        for k2 in 0..d {
                            cov[[j, k2]] += resp[[i, c]] * diff[j] * diff[k2];
                        }
                    }
                }
                CovType::Diag => {
                    for j in 0..d {
                        cov[[j, j]] += resp[[i, c]] * diff[j] * diff[j];
                    }
                }
            }
        }
        for j in 0..d {
            for k2 in 0..d {
                cov[[j, k2]] /= nk[c];
            }
            cov[[j, j]] += reg_covar;
        }
        covs.push(cov);
    }
    (weights, means, covs)
}

// ---------------------------------------------------------------------------
// Numeric helpers (kept private — no dep on solow-linalg)
// ---------------------------------------------------------------------------

fn sample_covariance(x: ArrayView2<'_, f64>, reg_covar: f64) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();
    let mut mean = vec![0.0_f64; d];
    for j in 0..d {
        for i in 0..n {
            mean[j] += x[[i, j]];
        }
        mean[j] /= n as f64;
    }
    let mut cov = Array2::<f64>::zeros((d, d));
    for i in 0..n {
        for j in 0..d {
            for k in 0..d {
                cov[[j, k]] += (x[[i, j]] - mean[j]) * (x[[i, k]] - mean[k]);
            }
        }
    }
    for j in 0..d {
        for k in 0..d {
            cov[[j, k]] /= (n as f64 - 1.0).max(1.0);
        }
        cov[[j, j]] += reg_covar;
    }
    cov
}

fn cholesky_and_log_det(m: &Array2<f64>) -> Result<(Array2<f64>, f64)> {
    let n = m.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    let mut log_det = 0.0_f64;
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(Error::Value(
                        "GaussianMixture: Cholesky failed on a component covariance".into(),
                    ));
                }
                l[[i, j]] = s.sqrt();
                log_det += 2.0 * l[[i, j]].ln();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Ok((l, log_det))
}

fn mahalanobis_via_chol(l: &Array2<f64>, diff: &[f64]) -> f64 {
    let n = diff.len();
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = diff[i];
        for k in 0..i {
            s -= l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }
    z.iter().map(|v| v * v).sum()
}

// k-means++ initialisation — reuses the same deterministic MMIX-LCG as
// solow-cluster::kmeans, so seeds transfer cleanly.
fn kpp_init(x: ArrayView2<'_, f64>, k: usize, seed: u64) -> Result<Array2<f64>> {
    let n = x.nrows();
    let d = x.ncols();
    let mut state = seed.wrapping_add(0xF00D_BABE_1234_5678);
    let mut means = Array2::<f64>::zeros((k, d));
    // First mean — uniform random row.
    let first = uniform_index(&mut state, n as u64);
    for j in 0..d {
        means[[0, j]] = x[[first, j]];
    }
    let mut d2 = vec![f64::INFINITY; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..d {
            let diff = x[[i, j]] - means[[0, j]];
            s += diff * diff;
        }
        d2[i] = s;
    }
    for c in 1..k {
        let total: f64 = d2.iter().sum();
        let pick = if total > 0.0 {
            let target = uniform_f64(&mut state) * total;
            let mut acc = 0.0_f64;
            let mut idx = 0usize;
            for (i, &v) in d2.iter().enumerate() {
                acc += v;
                if acc >= target {
                    idx = i;
                    break;
                }
            }
            idx
        } else {
            uniform_index(&mut state, n as u64)
        };
        for j in 0..d {
            means[[c, j]] = x[[pick, j]];
        }
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..d {
                let diff = x[[i, j]] - means[[c, j]];
                s += diff * diff;
            }
            if s < d2[i] {
                d2[i] = s;
            }
        }
    }
    Ok(means)
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

fn uniform_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gmm_recovers_two_well_separated_gaussians() {
        // Two blobs at (0, 0) and (10, 10) — a K=2 mixture reaches perfect
        // hard clustering.
        let mut rows: Vec<[f64; 2]> = (0..20)
            .map(|i| {
                [
                    ((i as f64) * 0.05).sin() * 0.1,
                    ((i as f64) * 0.07).cos() * 0.1,
                ]
            })
            .collect();
        rows.extend((0..20).map(|i| {
            [
                10.0 + ((i as f64) * 0.05).sin() * 0.1,
                10.0 + ((i as f64) * 0.07).cos() * 0.1,
            ]
        }));
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = Array2::from_shape_vec((40, 2), flat).unwrap();
        let gmm = GaussianMixture::fit(x.view(), 2, 42).unwrap();
        let pred = gmm.predict(x.view()).unwrap();
        let (a, b) = (pred[0], pred[20]);
        assert_ne!(a, b);
        for i in 0..20 {
            assert_eq!(pred[i], a);
        }
        for i in 20..40 {
            assert_eq!(pred[i], b);
        }
    }

    #[test]
    fn gmm_predict_proba_rows_sum_to_one() {
        let x = array![[0.0], [1.0], [10.0], [11.0]];
        let gmm = GaussianMixture::fit(x.view(), 2, 7).unwrap();
        let p = gmm.predict_proba(x.view()).unwrap();
        for i in 0..p.nrows() {
            let s: f64 = p.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn gmm_rejects_bad_k() {
        let x = array![[1.0]];
        assert!(GaussianMixture::fit(x.view(), 5, 0).is_err());
    }
}
