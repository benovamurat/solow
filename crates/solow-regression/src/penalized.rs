//! Penalised linear-regression estimators: [`Ridge`], [`Lasso`], and
//! [`ElasticNet`].
//!
//! All three fit the linear model `y = X β + ε` under a convex penalty
//! on the coefficient vector `β`:
//!
//! | Estimator | Objective |
//! | --- | --- |
//! | [`Ridge`] | `‖y − Xβ‖² / (2n) + α · ‖β‖²` |
//! | [`Lasso`] | `‖y − Xβ‖² / (2n) + α · ‖β‖₁` |
//! | [`ElasticNet`] | `‖y − Xβ‖² / (2n) + α · (ρ · ‖β‖₁ + ½(1 − ρ) · ‖β‖²)` |
//!
//! `Ridge` is solved in closed form via a Cholesky factor of
//! `XᵀX + α · I`. `Lasso` and `ElasticNet` use the **coordinate
//! descent** algorithm of Friedman-Hastie-Tibshirani (2010) — the
//! algorithm the reference ships in `linear_model.Lasso` and
//! `linear_model.ElasticNet` — which is provably convergent for convex
//! objectives and empirically the fastest deterministic solver on
//! dense inputs.
//!
//! # Intercept
//!
//! When `fit_intercept = true`, the estimator centres `y` and each
//! column of `X` before fitting and computes the intercept from the
//! sample means afterwards. The reported [`Ridge::intercept`],
//! [`Lasso::intercept`], and [`ElasticNet::intercept`] correspond to
//! the raw-feature model.
//!
//! # Standardisation
//!
//! Penalised models are scale-sensitive. The default is *not* to
//! standardise; wrap the caller in a
//! [`solow-preprocessing::StandardScaler`](https://docs.rs/solow-preprocessing/latest/solow_preprocessing/struct.StandardScaler.html)
//! if that's what you want, and remember to unwind the scaling on the
//! reported coefficients.
//!
//! # References
//!
//! * Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased
//!   estimation for nonorthogonal problems.* Technometrics 12(1), 55-67.
//! * Tibshirani, R. (1996). *Regression shrinkage and selection via
//!   the lasso.* JRSS-B 58(1), 267-288.
//! * Zou, H., & Hastie, T. (2005). *Regularization and variable
//!   selection via the elastic net.* JRSS-B 67(2), 301-320.
//! * Friedman, J., Hastie, T., & Tibshirani, R. (2010). *Regularization
//!   paths for generalized linear models via coordinate descent.*
//!   Journal of Statistical Software 33(1), 1-22.

use ndarray::{Array1, ArrayView1, ArrayView2};
use solow_core::{numeric::NeumaierSum, Error, Result};

// ---------------------------------------------------------------------------
// Ridge
// ---------------------------------------------------------------------------

/// L²-penalised linear regression.
#[derive(Clone, Debug)]
pub struct Ridge {
    /// Fitted coefficients `β` (length `p`).
    pub coef: Array1<f64>,
    /// Fitted intercept.
    pub intercept: f64,
    /// Penalty strength used during fit.
    pub alpha: f64,
    /// Whether the fit included an intercept.
    pub fit_intercept: bool,
}

impl Ridge {
    /// Fit `Ridge(α, fit_intercept)` on `(y, x)`.
    ///
    /// Solves `(XᵀX + α · I) β = Xᵀy` by Cholesky (positive definite for
    /// α > 0), then recovers the intercept from the sample means.
    /// Complexity `O(n · p² + p³)`.
    pub fn fit(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alpha: f64,
        fit_intercept: bool,
    ) -> Result<Self> {
        if y.len() != x.nrows() || x.ncols() == 0 {
            return Err(Error::Shape(format!(
                "Ridge::fit: shape mismatch (y: {}, x: {}×{})",
                y.len(),
                x.nrows(),
                x.ncols()
            )));
        }
        if !(alpha >= 0.0 && alpha.is_finite()) {
            return Err(Error::Value(format!(
                "Ridge::fit: alpha must be finite and ≥ 0 (got {alpha})"
            )));
        }
        let (n, p) = (x.nrows(), x.ncols());
        // Optional centering.
        let mut x_c = x.to_owned();
        let mut y_c = y.to_owned();
        let (mean_x, mean_y) = if fit_intercept {
            let mut mx = Array1::<f64>::zeros(p);
            for j in 0..p {
                mx[j] = compensated_mean(x.column(j).iter().copied(), n);
            }
            let my = compensated_mean(y.iter().copied(), n);
            for j in 0..p {
                let mj = mx[j];
                for i in 0..n {
                    x_c[[i, j]] -= mj;
                }
            }
            for i in 0..n {
                y_c[i] -= my;
            }
            (mx, my)
        } else {
            (Array1::<f64>::zeros(p), 0.0)
        };
        // A = XᵀX + α·I  (p × p)
        // b = Xᵀy         (p)
        let mut a = vec![vec![0.0_f64; p]; p];
        let mut b = vec![0.0_f64; p];
        for i in 0..n {
            for j in 0..p {
                let xj = x_c[[i, j]];
                b[j] += xj * y_c[i];
                for k in j..p {
                    a[j][k] += xj * x_c[[i, k]];
                }
            }
        }
        for j in 0..p {
            for k in 0..j {
                a[j][k] = a[k][j];
            }
            a[j][j] += alpha;
        }
        let beta_vec = cholesky_solve(&a, &b)?;
        let coef = Array1::from(beta_vec);
        let intercept = if fit_intercept {
            let mut s = 0.0_f64;
            for j in 0..p {
                s += coef[j] * mean_x[j];
            }
            mean_y - s
        } else {
            0.0
        };
        Ok(Self {
            coef,
            intercept,
            alpha,
            fit_intercept,
        })
    }

    /// Predict `X β + b` for every row of `x`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(Error::Shape(format!(
                "Ridge::predict: expected {} columns, got {}",
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
// Lasso — coordinate descent
// ---------------------------------------------------------------------------

/// L¹-penalised linear regression by coordinate descent.
#[derive(Clone, Debug)]
pub struct Lasso {
    /// Fitted coefficients.
    pub coef: Array1<f64>,
    /// Fitted intercept.
    pub intercept: f64,
    /// Penalty strength.
    pub alpha: f64,
    /// Whether the fit included an intercept.
    pub fit_intercept: bool,
    /// Number of outer coordinate-descent sweeps that ran.
    pub n_iter: usize,
    /// Whether the final sweep's max coordinate change fell below `tol`.
    pub converged: bool,
}

impl Lasso {
    /// Fit with default settings (`max_iter = 1000`, `tol = 1e-4`).
    pub fn fit(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alpha: f64,
        fit_intercept: bool,
    ) -> Result<Self> {
        Self::fit_with(y, x, alpha, fit_intercept, 1000, 1e-4)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        // Reduce to ElasticNet with `l1_ratio = 1`.
        let en = ElasticNet::fit_with(y, x, alpha, 1.0, fit_intercept, max_iter, tol)?;
        Ok(Self {
            coef: en.coef,
            intercept: en.intercept,
            alpha,
            fit_intercept,
            n_iter: en.n_iter,
            converged: en.converged,
        })
    }

    /// Predict on a new feature matrix.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        predict_linear(&self.coef, self.intercept, x, "Lasso")
    }
}

// ---------------------------------------------------------------------------
// ElasticNet — coordinate descent
// ---------------------------------------------------------------------------

/// Elastic-net penalised linear regression by coordinate descent.
///
/// `l1_ratio ∈ [0, 1]` mixes the two penalties:
/// `α · (l1_ratio · ‖β‖₁ + ½ · (1 − l1_ratio) · ‖β‖²)`. `l1_ratio = 1`
/// reduces to `Lasso`; `l1_ratio = 0` reduces to `Ridge` (though the
/// dedicated `Ridge` implementation is faster for that case).
#[derive(Clone, Debug)]
pub struct ElasticNet {
    /// Fitted coefficients.
    pub coef: Array1<f64>,
    /// Fitted intercept.
    pub intercept: f64,
    /// Penalty strength.
    pub alpha: f64,
    /// L¹ vs L² mixing.
    pub l1_ratio: f64,
    /// Whether the fit included an intercept.
    pub fit_intercept: bool,
    /// Number of outer coordinate-descent sweeps that ran.
    pub n_iter: usize,
    /// Whether the final sweep's max coordinate change fell below `tol`.
    pub converged: bool,
}

impl ElasticNet {
    /// Fit with default settings.
    pub fn fit(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
    ) -> Result<Self> {
        Self::fit_with(y, x, alpha, l1_ratio, fit_intercept, 1000, 1e-4)
    }

    /// Full-configuration fit.
    ///
    /// # Convergence
    ///
    /// Coordinate descent on the ElasticNet objective is provably
    /// convergent for `alpha ≥ 0`, `l1_ratio ∈ [0, 1]`. Convergence
    /// is monitored via the maximum absolute change in any coordinate
    /// across a full sweep; the fit terminates when this falls below
    /// `tol` or `max_iter` sweeps have run (whichever first).
    pub fn fit_with(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        if y.len() != x.nrows() || x.ncols() == 0 {
            return Err(Error::Shape(format!(
                "ElasticNet::fit_with: shape mismatch (y: {}, x: {}×{})",
                y.len(),
                x.nrows(),
                x.ncols()
            )));
        }
        if !(alpha >= 0.0 && alpha.is_finite()) {
            return Err(Error::Value(format!(
                "ElasticNet::fit_with: alpha must be finite and ≥ 0 (got {alpha})"
            )));
        }
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(Error::Value(format!(
                "ElasticNet::fit_with: l1_ratio must be in [0, 1] (got {l1_ratio})"
            )));
        }
        let (n, p) = (x.nrows(), x.ncols());
        let n_f = n as f64;

        // Centre for the intercept fit.
        let mut x_c = x.to_owned();
        let mut y_c = y.to_owned();
        let mean_x: Array1<f64>;
        let mean_y: f64;
        if fit_intercept {
            let mut mx = Array1::<f64>::zeros(p);
            for j in 0..p {
                mx[j] = compensated_mean(x.column(j).iter().copied(), n);
            }
            mean_y = compensated_mean(y.iter().copied(), n);
            for j in 0..p {
                let mj = mx[j];
                for i in 0..n {
                    x_c[[i, j]] -= mj;
                }
            }
            for i in 0..n {
                y_c[i] -= mean_y;
            }
            mean_x = mx;
        } else {
            mean_x = Array1::<f64>::zeros(p);
            mean_y = 0.0;
        }

        // Precompute the column norms `‖x_j‖² / n` — they are the
        // coordinate-descent step's denominator.
        let mut col_norms = vec![0.0_f64; p];
        for j in 0..p {
            let mut s = NeumaierSum::new();
            for i in 0..n {
                s.add(x_c[[i, j]] * x_c[[i, j]]);
            }
            col_norms[j] = s.finish() / n_f;
        }

        let l1 = alpha * l1_ratio;
        let l2 = alpha * (1.0 - l1_ratio);
        let mut beta = Array1::<f64>::zeros(p);
        // Residual vector `r = y_c - X β` — maintained incrementally.
        let mut r = y_c.clone();

        let mut n_iter = 0usize;
        let mut converged = false;
        for it in 0..max_iter {
            n_iter = it + 1;
            let mut max_change = 0.0_f64;
            for j in 0..p {
                let denom = col_norms[j] + l2;
                if denom <= 0.0 {
                    continue;
                }
                // Add back β_j · x_j to the residual so the coordinate update sees the
                // "partial residual" `r + β_j · x_j`.
                let bj = beta[j];
                if bj != 0.0 {
                    for i in 0..n {
                        r[i] += bj * x_c[[i, j]];
                    }
                }
                // Compute z_j = (1/n) · x_jᵀ r.
                let mut z = NeumaierSum::new();
                for i in 0..n {
                    z.add(x_c[[i, j]] * r[i]);
                }
                let z_j = z.finish() / n_f;
                // Soft-threshold.
                let new_bj = soft_threshold(z_j, l1) / denom;
                // Update residual to reflect the new coefficient.
                if new_bj != 0.0 {
                    for i in 0..n {
                        r[i] -= new_bj * x_c[[i, j]];
                    }
                }
                let change = (new_bj - bj).abs();
                if change > max_change {
                    max_change = change;
                }
                beta[j] = new_bj;
            }
            if max_change < tol {
                converged = true;
                break;
            }
        }

        let intercept = if fit_intercept {
            let mut s = 0.0_f64;
            for j in 0..p {
                s += beta[j] * mean_x[j];
            }
            mean_y - s
        } else {
            0.0
        };
        Ok(Self {
            coef: beta,
            intercept,
            alpha,
            l1_ratio,
            fit_intercept,
            n_iter,
            converged,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        predict_linear(&self.coef, self.intercept, x, "ElasticNet")
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn soft_threshold(z: f64, lambda: f64) -> f64 {
    if z > lambda {
        z - lambda
    } else if z < -lambda {
        z + lambda
    } else {
        0.0
    }
}

fn predict_linear(
    coef: &Array1<f64>,
    intercept: f64,
    x: ArrayView2<'_, f64>,
    label: &str,
) -> Result<Array1<f64>> {
    if x.ncols() != coef.len() {
        return Err(Error::Shape(format!(
            "{label}::predict: expected {} columns, got {}",
            coef.len(),
            x.ncols()
        )));
    }
    let mut out = Array1::<f64>::zeros(x.nrows());
    for i in 0..x.nrows() {
        let mut s = intercept;
        for j in 0..x.ncols() {
            s += coef[j] * x[[i, j]];
        }
        out[i] = s;
    }
    Ok(out)
}

fn compensated_mean<I: IntoIterator<Item = f64>>(it: I, n: usize) -> f64 {
    let mut acc = NeumaierSum::new();
    for v in it {
        acc.add(v);
    }
    acc.finish() / n as f64
}

/// Symmetric-positive-definite Cholesky solve.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    // Copy and factor.
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
                        "Ridge: Cholesky failed — matrix is not positive definite".into(),
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Forward: L z = b.
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * z[k];
        }
        z[i] = s / l[i][i];
    }
    // Backward: Lᵀ x = z.
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn ridge_reduces_to_ols_at_alpha_zero() {
        // y = 2 + 3x with no noise.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];
        let r = Ridge::fit(y.view(), x.view(), 0.0, true).unwrap();
        assert_abs_diff_eq!(r.intercept, 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r.coef[0], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn ridge_shrinks_coefficients_as_alpha_grows() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];
        let a = Ridge::fit(y.view(), x.view(), 0.0, true).unwrap().coef[0];
        let b = Ridge::fit(y.view(), x.view(), 100.0, true).unwrap().coef[0];
        assert!(
            b.abs() < a.abs(),
            "|β(α=100)| = {b} should shrink below |β(0)| = {a}"
        );
    }

    #[test]
    fn lasso_at_large_alpha_zeroes_all_coefficients() {
        let x = array![[1.0, 0.5], [2.0, -0.3], [3.0, 1.2], [4.0, -0.8]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        // A very strong penalty forces every β_j = 0.
        let l = Lasso::fit(y.view(), x.view(), 1e6, true).unwrap();
        for &b in l.coef.iter() {
            assert_abs_diff_eq!(b, 0.0, epsilon = 1e-9);
        }
        // Intercept absorbs the mean of y = 5.
        assert_abs_diff_eq!(l.intercept, 5.0, epsilon = 1e-9);
    }

    #[test]
    fn lasso_recovers_linear_signal_at_small_alpha() {
        // y = 3 · x1 + 0 · x2 + 1 (intercept). Small α should keep β_1
        // close to 3 and shrink β_2 close to 0.
        let x = array![[1.0, 5.0], [2.0, -3.0], [3.0, 4.0], [4.0, 0.0], [5.0, 2.0]];
        let y = array![4.0, 7.0, 10.0, 13.0, 16.0];
        let l = Lasso::fit_with(y.view(), x.view(), 0.01, true, 5000, 1e-6).unwrap();
        assert!((l.coef[0] - 3.0).abs() < 0.1, "coef[0] = {}", l.coef[0]);
        assert!(l.coef[1].abs() < 0.1, "coef[1] = {}", l.coef[1]);
    }

    #[test]
    fn elastic_net_interpolates_between_ridge_and_lasso() {
        let x = array![[1.0, 5.0], [2.0, -3.0], [3.0, 4.0], [4.0, 0.0], [5.0, 2.0]];
        let y = array![4.0, 7.0, 10.0, 13.0, 16.0];
        let en_lasso_like = ElasticNet::fit(y.view(), x.view(), 0.5, 1.0, true).unwrap();
        let en_ridge_like = ElasticNet::fit(y.view(), x.view(), 0.5, 0.0, true).unwrap();
        // Lasso-like should shrink coef[1] closer to zero than the ridge-like fit.
        assert!(
            en_lasso_like.coef[1].abs() <= en_ridge_like.coef[1].abs() + 1e-9,
            "expected the l1-like fit to shrink coef[1] at least as much as the l2-like fit"
        );
    }
}
