//! `BayesianRidge` and `ARDRegression` — Bayesian linear regression with
//! evidence-maximised hyperparameters (MacKay 1992, Tipping 2001).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Fitted BayesianRidge.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianRidge {
    /// Posterior mean of the coefficient vector `(d + 1)` (last = intercept).
    pub coef: Array1<f64>,
    /// Posterior precision of the noise `β`.
    pub alpha_noise: f64,
    /// Posterior precision of the coefficients `α`.
    pub alpha_coef: f64,
    /// Number of EM iterations run.
    pub n_iter: usize,
    /// Whether the EM converged to `tol`.
    pub converged: bool,
    /// Fit intercept flag.
    pub fit_intercept: bool,
}

impl BayesianRidge {
    /// Fit with the reference defaults `max_iter = 300`, `tol = 1e-3`,
    /// `alpha_1 = alpha_2 = lambda_1 = lambda_2 = 1e-6`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        Self::fit_with(x, y, true, 300, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        alpha_1: f64,
        alpha_2: f64,
        lambda_1: f64,
        lambda_2: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("BayesianRidge: y/x row mismatch".into()));
        }
        if n == 0 || d == 0 {
            return Err(Error::Value("BayesianRidge: empty input".into()));
        }
        let mut xd = if fit_intercept {
            let mut xa = Array2::<f64>::zeros((n, d + 1));
            for i in 0..n {
                for j in 0..d {
                    xa[[i, j]] = x[[i, j]];
                }
                xa[[i, d]] = 1.0;
            }
            xa
        } else {
            x.to_owned()
        };
        let p = xd.ncols();
        // Precompute XᵀX and Xᵀy once.
        let mut xtx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0_f64;
                for r in 0..n {
                    s += xd[[r, i]] * xd[[r, j]];
                }
                xtx[[i, j]] = s;
            }
        }
        let mut xty = Array1::<f64>::zeros(p);
        for i in 0..p {
            let mut s = 0.0_f64;
            for r in 0..n {
                s += xd[[r, i]] * y[r];
            }
            xty[i] = s;
        }
        let y_var = {
            let mut s = 0.0_f64;
            let mean = y.iter().sum::<f64>() / n as f64;
            for &v in y.iter() {
                s += (v - mean).powi(2);
            }
            (s / n as f64).max(1e-6)
        };
        let mut alpha = 1.0 / y_var;
        let mut lambda = 1.0_f64;
        let mut prev_coef = Array1::<f64>::from_elem(p, f64::INFINITY);
        let mut coef = Array1::<f64>::zeros(p);
        let mut sigma_diag = Array1::<f64>::zeros(p);
        let mut iters = 0_usize;
        let mut converged = false;
        for it in 0..max_iter {
            iters = it + 1;
            // Σ⁻¹ = α · XᵀX + λ · I  →  invert; μ = α · Σ · Xᵀy.
            let mut sinv = xtx.clone();
            for i in 0..p {
                for j in 0..p {
                    sinv[[i, j]] *= alpha;
                }
                sinv[[i, i]] += lambda;
            }
            let sigma = invert(&sinv)?;
            coef = matvec(&sigma, &xty);
            for c in coef.iter_mut() {
                *c *= alpha;
            }
            for i in 0..p {
                sigma_diag[i] = sigma[[i, i]];
            }
            // γ = Σᵢ (1 − λ · Σᵢᵢ) — effective number of well-determined parameters.
            let gamma: f64 = (0..p).map(|i| 1.0 - lambda * sigma_diag[i]).sum();
            // λ_new = (γ + 2·lambda_1) / (‖μ‖² + 2·lambda_2)
            let mu2: f64 = coef.iter().map(|c| c * c).sum();
            lambda = (gamma + 2.0 * lambda_1) / (mu2 + 2.0 * lambda_2).max(1e-30);
            // α_new = (n − γ + 2·alpha_1) / (‖y − Xμ‖² + 2·alpha_2)
            let mut resid_sq = 0.0_f64;
            for r in 0..n {
                let mut yhat = 0.0_f64;
                for j in 0..p {
                    yhat += xd[[r, j]] * coef[j];
                }
                resid_sq += (y[r] - yhat).powi(2);
            }
            alpha = (n as f64 - gamma + 2.0 * alpha_1)
                / (resid_sq + 2.0 * alpha_2).max(1e-30);
            let mut delta = 0.0_f64;
            for i in 0..p {
                delta += (coef[i] - prev_coef[i]).abs();
            }
            prev_coef = coef.clone();
            if delta < tol {
                converged = true;
                break;
            }
        }
        // Suppress unused mut warning on xd.
        let _ = &mut xd;
        Ok(Self {
            coef,
            alpha_noise: alpha,
            alpha_coef: lambda,
            n_iter: iters,
            converged,
            fit_intercept,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let p = self.coef.len();
        if self.fit_intercept && p != d + 1 {
            return Err(Error::Shape("BayesianRidge::predict: shape mismatch".into()));
        }
        if !self.fit_intercept && p != d {
            return Err(Error::Shape("BayesianRidge::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = if self.fit_intercept { self.coef[d] } else { 0.0 };
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

/// Fitted ARDRegression — Automatic Relevance Determination.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct ARDRegression {
    /// Posterior mean of the coefficient vector `(d + 1)` (last = intercept).
    pub coef: Array1<f64>,
    /// Posterior precision of the noise `β`.
    pub alpha_noise: f64,
    /// Per-feature ARD precision `λⱼ`.
    pub lambda: Array1<f64>,
    /// EM iterations run.
    pub n_iter: usize,
    /// Whether the EM converged.
    pub converged: bool,
    /// Fit intercept flag.
    pub fit_intercept: bool,
}

impl ARDRegression {
    /// Fit with the reference defaults `max_iter = 300`, `tol = 1e-3`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        Self::fit_with(x, y, true, 300, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        alpha_1: f64,
        alpha_2: f64,
        lambda_1: f64,
        lambda_2: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("ARDRegression: y/x row mismatch".into()));
        }
        let mut xd = if fit_intercept {
            let mut xa = Array2::<f64>::zeros((n, d + 1));
            for i in 0..n {
                for j in 0..d {
                    xa[[i, j]] = x[[i, j]];
                }
                xa[[i, d]] = 1.0;
            }
            xa
        } else {
            x.to_owned()
        };
        let p = xd.ncols();
        let mut xtx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0_f64;
                for r in 0..n {
                    s += xd[[r, i]] * xd[[r, j]];
                }
                xtx[[i, j]] = s;
            }
        }
        let mut xty = Array1::<f64>::zeros(p);
        for i in 0..p {
            let mut s = 0.0_f64;
            for r in 0..n {
                s += xd[[r, i]] * y[r];
            }
            xty[i] = s;
        }
        let mut alpha = 1.0 / (y.iter().map(|v| v * v).sum::<f64>() / n as f64 + 1e-6);
        let mut lambda = Array1::<f64>::from_elem(p, 1.0);
        let mut prev_coef = Array1::<f64>::from_elem(p, f64::INFINITY);
        let mut coef = Array1::<f64>::zeros(p);
        let mut iters = 0_usize;
        let mut converged = false;
        for it in 0..max_iter {
            iters = it + 1;
            let mut sinv = xtx.clone();
            for i in 0..p {
                for j in 0..p {
                    sinv[[i, j]] *= alpha;
                }
                sinv[[i, i]] += lambda[i];
            }
            let sigma = invert(&sinv)?;
            coef = matvec(&sigma, &xty);
            for c in coef.iter_mut() {
                *c *= alpha;
            }
            for i in 0..p {
                let gamma_i = 1.0 - lambda[i] * sigma[[i, i]];
                lambda[i] = (gamma_i + 2.0 * lambda_1) / (coef[i] * coef[i] + 2.0 * lambda_2).max(1e-30);
            }
            let gamma_total: f64 = (0..p).map(|i| 1.0 - lambda[i] * sigma[[i, i]]).sum();
            let mut resid_sq = 0.0_f64;
            for r in 0..n {
                let mut yhat = 0.0_f64;
                for j in 0..p {
                    yhat += xd[[r, j]] * coef[j];
                }
                resid_sq += (y[r] - yhat).powi(2);
            }
            alpha = (n as f64 - gamma_total + 2.0 * alpha_1)
                / (resid_sq + 2.0 * alpha_2).max(1e-30);
            let mut delta = 0.0_f64;
            for i in 0..p {
                delta += (coef[i] - prev_coef[i]).abs();
            }
            prev_coef = coef.clone();
            if delta < tol {
                converged = true;
                break;
            }
        }
        let _ = &mut xd;
        Ok(Self {
            coef,
            alpha_noise: alpha,
            lambda,
            n_iter: iters,
            converged,
            fit_intercept,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let p = self.coef.len();
        if self.fit_intercept && p != d + 1 {
            return Err(Error::Shape("ARDRegression::predict: shape mismatch".into()));
        }
        if !self.fit_intercept && p != d {
            return Err(Error::Shape("ARDRegression::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = if self.fit_intercept { self.coef[d] } else { 0.0 };
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

fn matvec(m: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let n = m.nrows();
    let p = m.ncols();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..p {
            s += m[[i, j]] * v[j];
        }
        out[i] = s;
    }
    out
}

fn invert(m: &Array2<f64>) -> Result<Array2<f64>> {
    let n = m.nrows();
    let mut a = vec![vec![0.0_f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[[i, j]];
        }
        a[i][n + i] = 1.0;
    }
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if best < 1e-30 {
            return Err(Error::Value("bayesian::invert: singular matrix".into()));
        }
        if piv != i {
            a.swap(i, piv);
        }
        let d = a[i][i];
        for c in 0..(2 * n) {
            a[i][c] /= d;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for c in 0..(2 * n) {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = a[i][n + j];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bayesian_ridge_recovers_a_simple_linear_signal() {
        // y = 2·x + 1
        let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![3.0_f64, 5.0, 7.0, 9.0, 11.0, 13.0];
        let m = BayesianRidge::fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..6 {
            assert!((p[i] - y[i]).abs() < 0.1);
        }
    }

    #[test]
    fn ard_regression_shrinks_irrelevant_features() {
        // y = 2·x₀ + noise-only column x₁.
        let x = array![
            [1.0_f64, 3.7], [2.0, 0.1], [3.0, -2.5], [4.0, 1.9],
            [5.0, -0.3], [6.0, 4.2], [7.0, 3.1], [8.0, -1.4]
        ];
        let y = array![2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let m = ARDRegression::fit(x.view(), y.view()).unwrap();
        // λ for the irrelevant column should be much larger than λ for x0.
        assert!(m.lambda[1] > m.lambda[0], "lambda: {:?}", m.lambda);
    }
}
