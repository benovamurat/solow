//! Multivariate analysis of variance (MANOVA).
//!
//! Mirrors the reference's `multivariate.manova.MANOVA`. For the multivariate
//! linear model `Y = X · B` (with `Y` the `nobs × k_endog` dependent variables
//! and `X` the `nobs × k_exog` design matrix), [`Manova::mv_test`] tests, for
//! each exogenous column `i`, the hypothesis `L · B = 0` where `L` selects that
//! column. Each test reports the four classical statistics — Wilks' lambda,
//! Pillai's trace, the Hotelling-Lawley trace and Roy's greatest root — with
//! their F-approximations and p-values.
//!
//! The estimation reproduces the reference's singular-value-decomposition fit:
//! given `X = U diag(s) Vᵀ`, the coefficient matrix is
//! `B = V diag(1/s) Uᵀ Y`, the (scaled) inverse cross-product is
//! `inv_cov = V diag(1/s²) Vᵀ`, and the residual sums of squares and
//! cross-products are `sscpr = YᵀY - (diag(s) V B)ᵀ (diag(s) V B)`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::{cholesky, eigh, inv, matrix_rank, solve_matrix, svd};

use crate::mvstats::{multivariate_stats, MultivariateStats};

/// A fitted multivariate OLS, holding the quantities needed for hypothesis
/// testing.
#[derive(Clone, Debug)]
pub struct Manova {
    params: Array2<f64>,
    df_resid: f64,
    inv_cov: Array2<f64>,
    sscpr: Array2<f64>,
    k_exog: usize,
}

impl Manova {
    /// Fit the multivariate linear model `endog = exog · B` by SVD.
    ///
    /// `endog` is `nobs × k_endog` (at least two dependent variables) and
    /// `exog` is `nobs × k_exog` (typically including an intercept column).
    pub fn new(endog: Array2<f64>, exog: Array2<f64>) -> Result<Self> {
        let (nobs, k_endog) = endog.dim();
        let (nobs1, k_exog) = exog.dim();
        if k_endog < 2 {
            return Err(Error::Shape(
                "there must be more than one dependent variable to fit MANOVA".into(),
            ));
        }
        if nobs != nobs1 {
            return Err(Error::Shape(
                "endog and exog must have the same number of rows".into(),
            ));
        }
        let df_resid = (nobs - k_exog) as f64;

        // SVD fit (method='svd'): X = U diag(s) Vt.
        let (u, s, vt) = svd(&exog)?;
        let tolerance = 1e-8;
        if s.iter().filter(|&&x| x > tolerance).count() < s.len() {
            return Err(Error::Singular("covariance of exog singular".into()));
        }
        let k = s.len();
        let invs = s.mapv(|x| 1.0 / x);

        // v = vt.T (k_exog × k). params = v · diag(invs) · uᵀ · y.
        // Compute uᵀ y (k × k_endog).
        let uty = u.t().dot(&endog);
        // diag(invs) · uᵀy.
        let mut dinv_uty = uty.clone();
        for i in 0..k {
            for j in 0..k_endog {
                dinv_uty[[i, j]] *= invs[i];
            }
        }
        // params = vt.T · (diag(invs) uᵀy)  -> (k_exog × k_endog).
        let params = vt.t().dot(&dinv_uty);

        // inv_cov = vt.T · diag(invs²) · vt   (k_exog × k_exog).
        let mut vt_scaled = vt.clone();
        for i in 0..k {
            let w = invs[i] * invs[i];
            for j in 0..vt.ncols() {
                vt_scaled[[i, j]] *= w;
            }
        }
        let inv_cov = vt.t().dot(&vt_scaled);

        // t = diag(s) · vt · params  (k × k_endog); sscpr = YᵀY - tᵀt.
        let vt_params = vt.dot(&params);
        let mut t = vt_params.clone();
        for i in 0..k {
            for j in 0..k_endog {
                t[[i, j]] *= s[i];
            }
        }
        let yty = endog.t().dot(&endog);
        let sscpr = &yty - &t.t().dot(&t);

        Ok(Manova {
            params,
            df_resid,
            inv_cov,
            sscpr,
            k_exog,
        })
    }

    /// Number of exogenous variables.
    pub fn k_exog(&self) -> usize {
        self.k_exog
    }

    /// Run the default multivariate tests: one hypothesis per exogenous column.
    ///
    /// Returns a vector with one [`ManovaTest`] per column of `exog`, in order
    /// (named `x0`, `x1`, …, matching the reference).
    pub fn mv_test(&self) -> Result<Vec<ManovaTest>> {
        let mut out = Vec::with_capacity(self.k_exog);
        for i in 0..self.k_exog {
            // L is a 1 × k_exog row selecting column i; M = I (k_endog); C = 0.
            let mut l = Array2::<f64>::zeros((1, self.k_exog));
            l[[0, i]] = 1.0;
            out.push(self.test_contrast(&l, format!("x{i}"))?);
        }
        Ok(out)
    }

    /// Test the contrast `L · B · M = 0` with `M = I` (identity transform on the
    /// dependent variables) and `C = 0`.
    pub fn test_contrast(&self, l: &Array2<f64>, name: String) -> Result<ManovaTest> {
        // t1 = L · params   (rows_L × k_endog).
        let t1 = l.dot(&self.params);
        // t2 = L · inv_cov · Lᵀ ; q = rank(t2).
        let t2 = l.dot(&self.inv_cov).dot(&l.t());
        let q = matrix_rank(&t2)?;
        // H = t1ᵀ · inv(t2) · t1   (k_endog × k_endog).
        let inv_t2 = inv(&t2)?;
        let h = t1.t().dot(&inv_t2).dot(&t1);
        // E = sscpr (since M = I).
        let e = self.sscpr.clone();

        let eh = &e + &h;
        let p = matrix_rank(&eh)?;

        // Eigenvalues of inv(E + H) · H. Both symmetric; E+H is SPD, so use the
        // Cholesky reduction to a symmetric standard eigenproblem.
        let eigvals = gen_sym_eigvals(&h, &eh)?;
        let eigvals_s = eigvals
            .as_slice()
            .ok_or_else(|| Error::Value("eigvals must be contiguous".into()))?;
        let stat = multivariate_stats(eigvals_s, p, q, self.df_resid);

        Ok(ManovaTest {
            name,
            stats: stat,
            e,
            h,
        })
    }
}

/// Eigenvalues of `inv(B) · A` where `A` (= H) is symmetric and `B` (= E + H) is
/// symmetric positive definite. Reduces to the symmetric standard eigenproblem
/// `C = L⁻¹ A L⁻ᵀ` with `B = L Lᵀ`, whose eigenvalues are real and equal those
/// of `inv(B) · A`. Returned in ascending order (matching the reference's
/// `np.sort(eigvals(...))`).
fn gen_sym_eigvals(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array1<f64>> {
    let l = cholesky(b)?; // B = L Lᵀ, L lower-triangular.
                          // Solve L · Y = A  ->  Y = L⁻¹ A.
    let y = solve_matrix(&l, a)?;
    // Solve L · Z = Yᵀ -> Z = L⁻¹ Yᵀ = L⁻¹ A L⁻ᵀ (since (L⁻¹A)ᵀ = Aᵀ L⁻ᵀ = A L⁻ᵀ).
    let yt = y.t().to_owned();
    let c = solve_matrix(&l, &yt)?;
    // Symmetrise to kill rounding asymmetry, then eigendecompose.
    let mut csym = c.clone();
    let n = c.nrows();
    for i in 0..n {
        for j in 0..n {
            csym[[i, j]] = 0.5 * (c[[i, j]] + c[[j, i]]);
        }
    }
    let (w, _v) = eigh(&csym)?;
    Ok(w) // ascending
}

/// A single MANOVA hypothesis result.
#[derive(Clone, Debug)]
pub struct ManovaTest {
    /// Name of the hypothesis (e.g. `x0`).
    pub name: String,
    /// The four multivariate statistics with their F-approximations.
    pub stats: MultivariateStats,
    /// Error sums-of-squares-and-cross-products matrix `E`.
    pub e: Array2<f64>,
    /// Hypothesis sums-of-squares-and-cross-products matrix `H`.
    pub h: Array2<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_group_two_dv_runs() {
        // Two groups, intercept + dummy, two dependent variables.
        let exog = array![
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];
        let endog = array![
            [1.0, 2.0],
            [1.2, 1.9],
            [0.9, 2.1],
            [3.0, 0.5],
            [3.1, 0.6],
            [2.9, 0.4],
        ];
        let m = Manova::new(endog, exog).unwrap();
        let tests = m.mv_test().unwrap();
        assert_eq!(tests.len(), 2);
        // Wilks lambda is in (0, 1].
        for t in &tests {
            assert!(t.stats.wilks_lambda.value > 0.0 && t.stats.wilks_lambda.value <= 1.0 + 1e-9);
            assert!(t.stats.pillai_trace.value >= -1e-12);
        }
    }

    #[test]
    fn rejects_single_dependent_variable() {
        let exog = array![[1.0], [1.0], [1.0]];
        let endog = array![[1.0], [2.0], [3.0]];
        assert!(Manova::new(endog, exog).is_err());
    }
}
