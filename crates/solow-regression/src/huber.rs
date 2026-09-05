//! Huber robust linear regression (Huber 1964).
//!
//! Minimises the composite objective
//!
//! ```text
//! Σᵢ ρ_ε((yᵢ − xᵢᵀβ) / σ) · σ  + α · ‖β‖²
//! ```
//!
//! where `ρ_ε(z) = ½ z²` for `|z| ≤ ε` and `ε · (|z| − ε/2)` otherwise.
//! Fits jointly over `(β, σ)` by iteratively reweighted least squares
//! (IRLS): given a scale estimate `σ`, compute per-sample weights
//! `wᵢ = min(1, ε · σ / |rᵢ|)`, solve the weighted ridge problem for
//! `β`, and update `σ = median(|r|) / 0.6745` (the MAD-based scale
//! estimator that matches the reference `HuberRegressor`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Fitted Huber regressor.
#[derive(Clone, Debug)]
pub struct HuberRegressor {
    /// Fitted coefficients.
    pub coef: Array1<f64>,
    /// Fitted intercept.
    pub intercept: f64,
    /// Fitted scale `σ` (MAD-based).
    pub scale: f64,
    /// Huber ε (threshold in scaled-residual space).
    pub epsilon: f64,
    /// L² penalty strength on `β`.
    pub alpha: f64,
    /// Number of IRLS iterations run.
    pub n_iter: usize,
    /// Whether the fit converged within `max_iter`.
    pub converged: bool,
}

impl HuberRegressor {
    /// Fit with defaults (`epsilon = 1.35`, `alpha = 1e-4`,
    /// `max_iter = 100`, `tol = 1e-5`).
    ///
    /// `epsilon = 1.35` is the classical Huber-recommended trade-off
    /// between efficiency at the Gaussian (~95%) and robustness to
    /// 5-10% outlier contamination.
    pub fn fit(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        fit_intercept: bool,
    ) -> Result<Self> {
        Self::fit_with(y, x, 1.35, 1e-4, 100, 1e-5, fit_intercept)
    }

    /// Full-configuration fit.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        epsilon: f64,
        alpha: f64,
        max_iter: usize,
        tol: f64,
        fit_intercept: bool,
    ) -> Result<Self> {
        if y.len() != x.nrows() || x.ncols() == 0 {
            return Err(Error::Shape(format!(
                "HuberRegressor::fit_with: shape mismatch (y: {}, x: {}×{})",
                y.len(),
                x.nrows(),
                x.ncols()
            )));
        }
        if !(epsilon > 1.0 && epsilon.is_finite()) {
            return Err(Error::Value(format!(
                "HuberRegressor::fit_with: epsilon must be > 1 (got {epsilon})"
            )));
        }
        if !(alpha >= 0.0 && alpha.is_finite()) {
            return Err(Error::Value(format!(
                "HuberRegressor::fit_with: alpha must be finite and ≥ 0 (got {alpha})"
            )));
        }
        let n = x.nrows();
        let p = x.ncols();
        // Build the effective design matrix (with intercept column if requested).
        let d = if fit_intercept { p + 1 } else { p };
        let mut design = vec![vec![0.0_f64; d]; n];
        for i in 0..n {
            if fit_intercept {
                design[i][0] = 1.0;
                for j in 0..p {
                    design[i][j + 1] = x[[i, j]];
                }
            } else {
                for j in 0..p {
                    design[i][j] = x[[i, j]];
                }
            }
        }
        // Initialise with an OLS-Ridge fit.
        let mut beta = solve_ridge(&design, y.as_slice().unwrap(), None, alpha)?;
        let mut sigma = mad_scale(y, &beta, &design).max(1e-6);
        let mut converged = false;
        let mut iter_used = 0usize;
        for it in 0..max_iter {
            iter_used = it + 1;
            // Compute per-sample IRLS weights.
            let mut w = vec![0.0_f64; n];
            for i in 0..n {
                let mut pred = 0.0_f64;
                for j in 0..d {
                    pred += design[i][j] * beta[j];
                }
                let r = (y[i] - pred).abs() / sigma;
                w[i] = if r <= epsilon { 1.0 } else { epsilon / r };
            }
            // Re-solve weighted ridge.
            let new_beta = solve_ridge(&design, y.as_slice().unwrap(), Some(&w), alpha)?;
            // Update scale.
            let new_sigma = mad_scale(y, &new_beta, &design).max(1e-6);
            // Convergence: max coefficient change and scale change small.
            let mut d_beta = 0.0_f64;
            for j in 0..d {
                d_beta = d_beta.max((new_beta[j] - beta[j]).abs());
            }
            let d_sigma = (new_sigma - sigma).abs();
            beta = new_beta;
            sigma = new_sigma;
            if d_beta < tol && d_sigma < tol {
                converged = true;
                break;
            }
        }
        let (intercept, coef) = if fit_intercept {
            let mut c = Array1::<f64>::zeros(p);
            for j in 0..p {
                c[j] = beta[j + 1];
            }
            (beta[0], c)
        } else {
            (0.0, Array1::from(beta))
        };
        Ok(Self {
            coef,
            intercept,
            scale: sigma,
            epsilon,
            alpha,
            n_iter: iter_used,
            converged,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(Error::Shape(format!(
                "HuberRegressor::predict: expected {} columns, got {}",
                self.coef.len(),
                x.ncols()
            )));
        }
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut s = self.intercept;
            for j in 0..x.ncols() {
                s += self.coef[j] * x[[i, j]];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// KernelRidge — same shape as kernel_ridge.KernelRidge
// ---------------------------------------------------------------------------

/// Kernel choice for [`KernelRidge`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RidgeKernel {
    /// Linear kernel `⟨x, y⟩`.
    Linear,
    /// RBF (Gaussian) kernel `exp(−γ · ‖x − y‖²)`.
    Rbf {
        /// Bandwidth γ.
        gamma: f64,
    },
    /// Polynomial kernel `(γ · ⟨x, y⟩ + coef0)^degree`.
    Polynomial {
        /// Polynomial degree.
        degree: u32,
        /// Scale on the dot product.
        gamma: f64,
        /// Additive offset.
        coef0: f64,
    },
}

impl RidgeKernel {
    fn apply(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            RidgeKernel::Linear => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            RidgeKernel::Rbf { gamma } => {
                let mut s = 0.0_f64;
                for k in 0..a.len() {
                    let d = a[k] - b[k];
                    s += d * d;
                }
                (-gamma * s).exp()
            }
            RidgeKernel::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let base: f64 =
                    gamma * a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>() + coef0;
                let mut acc = 1.0_f64;
                for _ in 0..*degree {
                    acc *= base;
                }
                acc
            }
        }
    }
}

/// Kernel ridge regression — the closed-form dual solution
/// `α = (K + λI)⁻¹ y`, predicting via `f(x) = Σᵢ αᵢ · k(xᵢ, x)`.
#[derive(Clone, Debug)]
pub struct KernelRidge {
    /// Training-set support vectors (needed at predict time).
    pub x_train: Array2<f64>,
    /// Dual coefficients `α` (length `n_train`).
    pub dual_coef: Array1<f64>,
    /// Kernel used at fit time.
    pub kernel: RidgeKernel,
    /// Regularisation strength.
    pub alpha: f64,
}

impl KernelRidge {
    /// Fit `α = (K + α·I)⁻¹ y` with the given kernel.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        kernel: RidgeKernel,
        alpha: f64,
    ) -> Result<Self> {
        if x.nrows() != y.len() || x.ncols() == 0 {
            return Err(Error::Shape(format!(
                "KernelRidge::fit: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if !(alpha >= 0.0 && alpha.is_finite()) {
            return Err(Error::Value(format!(
                "KernelRidge::fit: alpha must be finite and ≥ 0 (got {alpha})"
            )));
        }
        let n = x.nrows();
        // Build kernel matrix K (n × n), symmetric.
        let mut k = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).to_vec();
            for j in 0..=i {
                let xj: Vec<f64> = x.row(j).to_vec();
                let v = kernel.apply(&xi, &xj);
                k[i][j] = v;
                k[j][i] = v;
            }
        }
        // (K + αI) α = y  — Cholesky solve.
        for i in 0..n {
            k[i][i] += alpha;
        }
        let dual = cholesky_solve(&k, y.as_slice().unwrap())?;
        Ok(Self {
            x_train: x.to_owned(),
            dual_coef: Array1::from(dual),
            kernel,
            alpha,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.x_train.ncols() {
            return Err(Error::Shape(format!(
                "KernelRidge::predict: expected {} columns, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        let n = x.nrows();
        let n_train = self.x_train.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).to_vec();
            for j in 0..n_train {
                let xj: Vec<f64> = self.x_train.row(j).to_vec();
                let k = self.kernel.apply(&xi, &xj);
                out[i] += self.dual_coef[j] * k;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Shared numeric helpers
// ---------------------------------------------------------------------------

fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(Error::Value(
                        "Cholesky failed — matrix is not positive definite".into(),
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Forward L z = b.
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * z[k];
        }
        z[i] = s / l[i][i];
    }
    // Backward Lᵀ x = z.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Ok(x)
}

fn solve_ridge(
    design: &[Vec<f64>],
    y: &[f64],
    weights: Option<&[f64]>,
    alpha: f64,
) -> Result<Vec<f64>> {
    let n = design.len();
    let d = design[0].len();
    // A = XᵀWX + α·I (skip α on the intercept — following the reference convention).
    let mut a = vec![vec![0.0_f64; d]; d];
    let mut b = vec![0.0_f64; d];
    for i in 0..n {
        let w = weights.map(|w| w[i]).unwrap_or(1.0);
        for j in 0..d {
            let xj = design[i][j];
            b[j] += w * xj * y[i];
            for k in j..d {
                a[j][k] += w * xj * design[i][k];
            }
        }
    }
    for j in 0..d {
        for k in 0..j {
            a[j][k] = a[k][j];
        }
    }
    // Add α to every diagonal (matches solow's Ridge; the reference Huber does
    // the same after intercept-centering).
    for j in 0..d {
        a[j][j] += alpha;
    }
    cholesky_solve(&a, &b)
}

fn mad_scale(y: ArrayView1<'_, f64>, beta: &[f64], design: &[Vec<f64>]) -> f64 {
    let n = design.len();
    let d = beta.len();
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        let mut pred = 0.0_f64;
        for j in 0..d {
            pred += design[i][j] * beta[j];
        }
        r.push((y[i] - pred).abs());
    }
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = r.len();
    let med = if m % 2 == 0 {
        0.5 * (r[m / 2 - 1] + r[m / 2])
    } else {
        r[m / 2]
    };
    // Normal-consistent MAD scale.
    med / 0.6745
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn huber_ignores_lone_outlier_better_than_ols() {
        // y = 2x + 1 exactly, plus one 100σ outlier at row 5.
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let mut y = x.column(0).mapv(|v| 2.0 * v + 1.0);
        y[5] += 500.0; // huge outlier
        let huber = HuberRegressor::fit(y.view(), x.view(), true).unwrap();
        // Slope should still be close to 2 despite the outlier.
        assert!(
            (huber.coef[0] - 2.0).abs() < 0.5,
            "coef = {}, expected ~2",
            huber.coef[0]
        );
    }

    #[test]
    fn kernel_ridge_linear_reduces_to_dual_ridge() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let kr = KernelRidge::fit(x.view(), y.view(), RidgeKernel::Linear, 0.001).unwrap();
        let pred = kr.predict(x.view()).unwrap();
        for (a, b) in pred.iter().zip(y.iter()) {
            assert!((a - b).abs() < 0.05, "|{a} − {b}| large");
        }
    }

    #[test]
    fn kernel_ridge_rbf_fits_nonlinear_target() {
        // sin(x) on 20 samples.
        let n = 20usize;
        let x = ndarray::Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64 * 0.3).collect())
            .unwrap();
        let y = x.column(0).mapv(f64::sin);
        let kr =
            KernelRidge::fit(x.view(), y.view(), RidgeKernel::Rbf { gamma: 0.5 }, 1e-3).unwrap();
        let pred = kr.predict(x.view()).unwrap();
        let mse: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mse < 0.01, "kernel-ridge MSE = {mse}");
    }
}
