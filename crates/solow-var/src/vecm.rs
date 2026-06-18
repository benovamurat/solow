//! Vector error-correction model (VECM) and the Johansen cointegration test.
//!
//! This module implements Johansen's reduced-rank maximum-likelihood estimator
//! of a VECM at a fixed cointegration rank together with the
//! [`coint_johansen`] trace and maximum-eigenvalue test, mirroring the reference
//! `tsa.vector_ar.vecm` module (Lütkepohl, *New Introduction to Multiple Time
//! Series Analysis*, pp. 286-299).
//!
//! The VEC representation of a `K`-dimensional series is
//!
//! ```text
//! Δy_t = Π y_{t-1} + Γ_1 Δy_{t-1} + ... + Γ_{p-1} Δy_{t-p+1} + u_t
//! ```
//!
//! with `Π = α βᵀ` of reduced rank `r`. Estimation proceeds by concentrating
//! out the short-run dynamics `Γ` and solving a generalized symmetric
//! eigenproblem for the cointegrating space.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::{cholesky, eigh, inv, pinv};

use crate::coint_tables::{c_sja, c_sjt};

/// Deterministic-term specification for a [`Vecm`] model.
///
/// Mirrors the reference `deterministic` keyword. Only the terms relevant to
/// the constant are exposed (which is the common practical case).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deterministic {
    /// No deterministic terms (`"n"`).
    None,
    /// Constant *outside* the cointegration relation (`"co"`): an intercept in
    /// the short-run (`Γ`) equation.
    ConstantOutside,
    /// Constant *inside* the cointegration relation (`"ci"`): an intercept that
    /// enters through the cointegrating vector.
    ConstantInside,
}

impl Deterministic {
    /// Number of deterministic columns appended to the short-run regressor
    /// block `Δx` (the `Γ`/`det_coef` side).
    fn n_outside(self) -> usize {
        match self {
            Deterministic::ConstantOutside => 1,
            _ => 0,
        }
    }

    /// Number of deterministic rows appended to the lagged level `y_{t-1}`
    /// (the `β`/`det_coef_coint` side).
    fn n_inside(self) -> usize {
        match self {
            Deterministic::ConstantInside => 1,
            _ => 0,
        }
    }
}

/// Result of the Johansen cointegration test (see [`coint_johansen`]).
#[derive(Debug, Clone)]
pub struct JohansenResult {
    /// Ordered (descending) eigenvalues `λ_1 ≥ ... ≥ λ_K`.
    pub eig: Array1<f64>,
    /// Trace statistics `lr1[i] = -T Σ_{j≥i} log(1 - λ_j)`.
    pub lr1: Array1<f64>,
    /// Maximum-eigenvalue statistics `lr2[i] = -T log(1 - λ_i)`.
    pub lr2: Array1<f64>,
    /// Trace-statistic critical values, shape `(K, 3)` for the 90/95/99%
    /// quantiles (`NaN` where unavailable).
    pub cvt: Array2<f64>,
    /// Maximum-eigenvalue critical values, shape `(K, 3)`.
    pub cvm: Array2<f64>,
}

/// Johansen cointegration test of the cointegration rank of a VECM.
///
/// `det_order` selects the deterministic trend assumption used both for
/// detrending and for the tabulated asymptotic critical values:
///
/// * `-1` - no deterministic terms,
/// * `0` - constant term,
/// * `1` - linear trend.
///
/// `k_ar_diff` is the number of lagged differences in the model. The returned
/// eigenvalues solve the generalized symmetric eigenproblem
/// `S_{k0} S_{00}^{-1} S_{0k} v = λ S_{kk} v` on the residuals of `Δy_t` and
/// `y_{t-1}` after partialling out the lagged differences.
pub fn coint_johansen(
    endog: &Array2<f64>,
    det_order: i32,
    k_ar_diff: usize,
) -> Result<JohansenResult> {
    let (nobs, neqs) = endog.dim();
    if nobs < 2 {
        return Err(Error::Value("endog must have at least two rows".into()));
    }

    // Detrend the levels by `det_order` (regression on a Vandermonde basis).
    let endog_dt = detrend(endog, det_order)?;

    // First differences of the detrended levels.
    let dx_full = diff_rows(&endog_dt); // (nobs-1, neqs)

    // Stacked lagged differences z = lagmat(dx, k_ar_diff)[k_ar_diff:].
    let z = lagmat(&dx_full, k_ar_diff);
    let z = z.slice(s![k_ar_diff.., ..]).to_owned();
    // For det_order == -1, detrend z by f = -1 (constant); else f = 0 (none).
    let f = if det_order > -1 { 0 } else { det_order };
    let z = detrend(&z, f)?;

    // Δy on the same window.
    let dx = dx_full.slice(s![k_ar_diff.., ..]).to_owned();
    let dx = detrend(&dx, f)?;
    let r0t = resid_on(&dx, &z)?; // (t, neqs)

    // Lagged levels: endog[:nobs-k_ar_diff][1:].
    let upper = endog_dt.shape()[0] - k_ar_diff;
    let lx = endog_dt.slice(s![..upper, ..]).to_owned();
    let lx = lx.slice(s![1.., ..]).to_owned();
    let lx = detrend(&lx, f)?;
    let rkt = resid_on(&lx, &z)?; // (t, neqs)

    let t = rkt.shape()[0] as f64;
    let skk = rkt.t().dot(&rkt).mapv(|v| v / t);
    let sk0 = rkt.t().dot(&r0t).mapv(|v| v / t);
    let s00 = r0t.t().dot(&r0t).mapv(|v| v / t);

    // sig = sk0 inv(s00) sk0ᵀ ; eigenvalues of inv(skk) sig.
    let s00_inv = inv(&s00)?;
    let sig = sk0.dot(&s00_inv).dot(&sk0.t());
    let eig = gen_sym_eig_desc(&sig, &skk)?;

    // Trace and max-eigenvalue statistics with critical values.
    let mut lr1 = Array1::<f64>::zeros(neqs);
    let mut lr2 = Array1::<f64>::zeros(neqs);
    let mut cvm = Array2::<f64>::zeros((neqs, 3));
    let mut cvt = Array2::<f64>::zeros((neqs, 3));
    for i in 0..neqs {
        let mut trace = 0.0;
        for j in i..neqs {
            trace += (1.0 - eig[j]).ln();
        }
        lr1[i] = -t * trace;
        lr2[i] = -t * (1.0 - eig[i]).ln();
        let cm = c_sja(neqs - i, det_order);
        let ct = c_sjt(neqs - i, det_order);
        for k in 0..3 {
            cvm[[i, k]] = cm[k];
            cvt[[i, k]] = ct[k];
        }
    }

    Ok(JohansenResult {
        eig,
        lr1,
        lr2,
        cvt,
        cvm,
    })
}

/// A vector error-correction model estimated by Johansen's reduced-rank ML.
///
/// Construct with [`Vecm::new`] (no deterministic term) or
/// [`Vecm::with_deterministic`], then call [`Vecm::fit`].
#[derive(Debug, Clone)]
pub struct Vecm {
    endog: Array2<f64>,
    k_ar_diff: usize,
    coint_rank: usize,
    deterministic: Deterministic,
}

/// Fitted results of a [`Vecm`] model.
#[derive(Debug, Clone)]
pub struct VecmResults {
    /// Number of equations / series `K`.
    pub neqs: usize,
    /// Number of lagged differences `p - 1`.
    pub k_ar_diff: usize,
    /// Cointegration rank `r`.
    pub coint_rank: usize,
    /// Number of estimation observations `T`.
    pub nobs: usize,
    /// Loading matrix `α`, shape `(K, r)`.
    pub alpha: Array2<f64>,
    /// Cointegration matrix `β`, shape `(K, r)`, normalized so the first `r`
    /// rows form the identity.
    pub beta: Array2<f64>,
    /// Short-run coefficient matrices stacked as `Γ = [Γ_1, ..., Γ_{p-1}]`,
    /// shape `(K, K * k_ar_diff)`.
    pub gamma: Array2<f64>,
    /// Deterministic coefficients on the `Γ` side (`"co"`), shape
    /// `(K, n_outside)` (empty if none).
    pub det_coef: Array2<f64>,
    /// Deterministic coefficients on the `β` side (`"ci"`), shape
    /// `(n_inside, r)` (empty if none).
    pub det_coef_coint: Array2<f64>,
    /// ML residual covariance `Σ_u = U Uᵀ / T`, shape `(K, K)`.
    pub sigma_u: Array2<f64>,
    /// Gaussian log-likelihood at the estimates.
    pub llf: f64,
}

impl Vecm {
    /// Create a VECM with no deterministic terms.
    pub fn new(endog: Array2<f64>, k_ar_diff: usize, coint_rank: usize) -> Result<Self> {
        Self::with_deterministic(endog, k_ar_diff, coint_rank, Deterministic::None)
    }

    /// Create a VECM with an explicit [`Deterministic`] specification.
    pub fn with_deterministic(
        endog: Array2<f64>,
        k_ar_diff: usize,
        coint_rank: usize,
        deterministic: Deterministic,
    ) -> Result<Self> {
        let (n, k) = endog.dim();
        if k == 0 || n == 0 {
            return Err(Error::Value("endog must be non-empty".into()));
        }
        if coint_rank == 0 || coint_rank > k {
            return Err(Error::Value("coint_rank must be in 1..=neqs".into()));
        }
        if n <= k_ar_diff + 2 {
            return Err(Error::Value("not enough observations for k_ar_diff".into()));
        }
        Ok(Self {
            endog,
            k_ar_diff,
            coint_rank,
            deterministic,
        })
    }

    /// Estimate the VECM by maximum likelihood (Johansen reduced rank).
    pub fn fit(&self) -> Result<VecmResults> {
        let k = self.endog.dim().1;
        let r = self.coint_rank;
        let (delta_y, y_lag1, delta_x) = self.endog_matrices();
        let t = delta_y.dim().1 as f64;

        // S-matrices and the cointegration eigenvectors (p. 294-295).
        let sij = compute_sij(&delta_x, &delta_y, &y_lag1)?;

        // beta_tilde = (v[:, :r]ᵀ s11_)ᵀ then normalized so the first r rows
        // form the identity: β ← β inv(β[:r]).
        let v_r = sij.v.slice(s![.., ..r]); // (m, r) where m = rows of y_lag1
        let beta_unnorm = sij.s11_.t().dot(&v_r); // (m, r)
        let top = beta_unnorm.slice(s![..r, ..]).to_owned();
        let top_inv = inv(&top)?;
        let beta_full = beta_unnorm.dot(&top_inv); // (m, r)

        // alpha = s01 β inv(βᵀ s11 β).
        let bt_s11_b = beta_full.t().dot(&sij.s11).dot(&beta_full);
        let alpha = sij.s01.dot(&beta_full).dot(&inv(&bt_s11_b)?); // (K, r)

        // gamma = (Δy - α βᵀ y_lag1) Δxᵀ inv(Δx Δxᵀ).
        let ab = alpha.dot(&beta_full.t()); // (K, m)
        let resid_long = &delta_y - &ab.dot(&y_lag1); // (K, T)
        let dxt = delta_x.t();
        let gamma_full = resid_long.dot(&dxt).dot(&inv(&delta_x.dot(&dxt))?); // (K, n_x)

        // Residual covariance.
        let temp = &resid_long - &gamma_full.dot(&delta_x);
        let sigma_u = temp.dot(&temp.t()).mapv(|v| v / t);

        // Split deterministic pieces off beta_full / gamma_full.
        let beta = beta_full.slice(s![..k, ..]).to_owned();
        let n_inside = self.deterministic.n_inside();
        let det_coef_coint = if n_inside > 0 {
            beta_full.slice(s![k.., ..]).to_owned()
        } else {
            Array2::<f64>::zeros((0, r))
        };
        let n_gamma = k * self.k_ar_diff;
        let gamma = gamma_full.slice(s![.., ..n_gamma]).to_owned();
        let det_coef = if self.deterministic.n_outside() > 0 {
            gamma_full.slice(s![.., n_gamma..]).to_owned()
        } else {
            Array2::<f64>::zeros((k, 0))
        };

        // Log-likelihood (Lütkepohl p. 295, eq. 7.2.20).
        let s00 = &sij.s00;
        let logdet_s00 = det_logdet(s00)?;
        let mut sum_log = 0.0;
        for i in 0..r {
            sum_log += (1.0 - sij.lambd[i]).ln();
        }
        let kf = k as f64;
        let llf = -kf * t * (2.0 * std::f64::consts::PI).ln() / 2.0
            - t * (logdet_s00 + sum_log) / 2.0
            - kf * t / 2.0;

        Ok(VecmResults {
            neqs: k,
            k_ar_diff: self.k_ar_diff,
            coint_rank: r,
            nobs: t as usize,
            alpha,
            beta,
            gamma,
            det_coef,
            det_coef_coint,
            sigma_u,
            llf,
        })
    }

    /// Build the estimation matrices following the reference `_endog_matrices`.
    ///
    /// Returns `(delta_y_1_T, y_lag1, delta_x)` with the reference's
    /// `(rows x nobs)` (transposed) orientation:
    /// * `delta_y_1_T`: `(K, T)`,
    /// * `y_lag1`: `(K + n_inside, T)`,
    /// * `delta_x`: `(K*k_ar_diff + n_outside, T)`.
    fn endog_matrices(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let (nobs, k) = self.endog.dim();
        let diff_lags = self.k_ar_diff;
        let p = diff_lags + 1;
        let tt = nobs - p; // T

        // delta_y[:, i] = y[:, i+1] - y[:, i] over the full sample; we work
        // with the natural (row=time) layout and index columns of the
        // transposed result.
        // delta_y full has nobs-1 columns; delta_y_1_T = delta_y[:, p-1:].
        let mut delta_y = Array2::<f64>::zeros((k, tt));
        for j in 0..tt {
            let src = (p - 1) + j; // column index into the full diff
            for c in 0..k {
                delta_y[[c, j]] = self.endog[[src + 1, c]] - self.endog[[src, c]];
            }
        }

        // y_lag1 = y[:, p-1:-1] -> times p-1 .. nobs-2 inclusive (T columns).
        let n_inside = self.deterministic.n_inside();
        let mut y_lag1 = Array2::<f64>::zeros((k + n_inside, tt));
        for j in 0..tt {
            let row = (p - 1) + j;
            for c in 0..k {
                y_lag1[[c, j]] = self.endog[[row, c]];
            }
            if n_inside > 0 {
                y_lag1[[k, j]] = 1.0; // constant inside cointegration
            }
        }

        // delta_x rows: for lag block, delta_x[:, j] stacks
        // delta_y[:, j+p-2 : j-1 : -1] reshaped (K*(p-1)). I.e. the most recent
        // lagged difference first.
        let n_outside = self.deterministic.n_outside();
        let mut delta_x = Array2::<f64>::zeros((k * diff_lags + n_outside, tt));
        for j in 0..tt {
            // The l-th lag block (l = 0..diff_lags) holds delta_y at time
            // (p-2+j) - l, i.e. Δy_{t-1-l}.
            for l in 0..diff_lags {
                let src = (p - 2 + j) - l; // column index into full diff
                for c in 0..k {
                    delta_x[[l * k + c, j]] = self.endog[[src + 1, c]] - self.endog[[src, c]];
                }
            }
            if n_outside > 0 {
                delta_x[[k * diff_lags, j]] = 1.0; // constant outside
            }
        }

        (delta_y, y_lag1, delta_x)
    }
}

/// Bundle of S-matrices and eigen-decomposition from the reference `_sij`.
struct Sij {
    s00: Array2<f64>,
    s01: Array2<f64>,
    s11: Array2<f64>,
    s11_: Array2<f64>,
    lambd: Array1<f64>,
    v: Array2<f64>,
}

/// Compute the S-matrices and the descending-ordered eigenpairs (reference
/// `_sij`).
fn compute_sij(delta_x: &Array2<f64>, delta_y: &Array2<f64>, y_lag1: &Array2<f64>) -> Result<Sij> {
    let nobs = y_lag1.dim().1 as f64;
    // M = I - Δxᵀ inv(Δx Δxᵀ) Δx ; r0 = Δy M, r1 = y_lag1 M.
    let dxt = delta_x.t();
    let proj = dxt.dot(&inv(&delta_x.dot(&dxt))?).dot(delta_x); // (T, T)
    let id = Array2::<f64>::eye(proj.dim().0);
    let m = &id - &proj;
    let r0 = delta_y.dot(&m); // (K, T)
    let r1 = y_lag1.dot(&m); // (m, T)

    let s00 = r0.dot(&r0.t()).mapv(|v| v / nobs);
    let s01 = r0.dot(&r1.t()).mapv(|v| v / nobs);
    let s11 = r1.dot(&r1.t()).mapv(|v| v / nobs);
    let s11_ = inv(&mat_sqrt(&s11)?)?; // inverse symmetric square root

    // eig(s01_s11_ᵀ inv(s00) s01_s11_), symmetric PSD.
    let s01_s11_ = s01.dot(&s11_); // (K, m)
    let mid = s01_s11_.t().dot(&inv(&s00)?).dot(&s01_s11_); // (m, m), symmetric
    let mid = symmetrize(&mid);
    let (w_asc, v_asc) = eigh(&mid)?;
    // Reorder to descending eigenvalues.
    let n = w_asc.len();
    let mut lambd = Array1::<f64>::zeros(n);
    let mut v = Array2::<f64>::zeros(v_asc.dim());
    for i in 0..n {
        let src = n - 1 - i;
        lambd[i] = w_asc[src];
        for row in 0..v_asc.dim().0 {
            v[[row, i]] = v_asc[[row, src]];
        }
    }
    Ok(Sij {
        s00,
        s01,
        s11,
        s11_,
        lambd,
        v,
    })
}

/// Symmetric matrix square root via SVD: `A^{1/2} = U diag(sqrt(s)) Vᵀ`
/// (reference `_mat_sqrt`). For a symmetric PD matrix this is the principal
/// square root.
fn mat_sqrt(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (u, s, vt) = solow_linalg::svd(a)?;
    let mut scaled = u.clone();
    let n = s.len();
    for j in 0..n {
        let sq = s[j].sqrt();
        for i in 0..scaled.dim().0 {
            scaled[[i, j]] *= sq;
        }
    }
    Ok(scaled.dot(&vt))
}

/// Force exact symmetry on a numerically-symmetric matrix.
fn symmetrize(a: &Array2<f64>) -> Array2<f64> {
    let at = a.t();
    (a + &at).mapv(|v| v / 2.0)
}

/// Solve the generalized symmetric eigenproblem `A v = λ B v` (with `A`
/// symmetric, `B` symmetric positive definite) returning eigenvalues sorted
/// descending. Uses the Cholesky factor of `B` to reduce to a standard
/// symmetric eigenproblem.
fn gen_sym_eig_desc(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array1<f64>> {
    // B = L Lᵀ ; C = L^{-1} A L^{-ᵀ}, eig(C) = eig(B^{-1} A).
    let l = cholesky(b)?;
    let l_inv = inv(&l)?;
    let c = l_inv.dot(a).dot(&l_inv.t());
    let c = symmetrize(&c);
    let (w_asc, _) = eigh(&c)?;
    let n = w_asc.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        out[i] = w_asc[n - 1 - i];
    }
    Ok(out)
}

/// log|det(A)| for a symmetric positive-definite matrix via Cholesky.
fn det_logdet(a: &Array2<f64>) -> Result<f64> {
    let l = cholesky(a)?;
    let n = a.dim().0;
    let mut s = 0.0;
    for i in 0..n {
        s += l[[i, i]].ln();
    }
    Ok(2.0 * s)
}

/// First differences along rows: `out[i] = x[i+1] - x[i]`.
fn diff_rows(x: &Array2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut out = Array2::<f64>::zeros((n - 1, k));
    for i in 0..n - 1 {
        for c in 0..k {
            out[[i, c]] = x[[i + 1, c]] - x[[i, c]];
        }
    }
    out
}

/// Detrend `y` (rows = observations) by regressing on a Vandermonde basis of a
/// linear grid from `-1` to `1`, returning the residuals. Matches the
/// reference `detrend`:
/// * `order == -1` -> identity (no detrending),
/// * `order >= 0`  -> regress on `vander(linspace(-1,1,n), order+1)`.
fn detrend(y: &Array2<f64>, order: i32) -> Result<Array2<f64>> {
    if order == -1 {
        return Ok(y.clone());
    }
    let (n, _) = y.dim();
    let deg = (order + 1) as usize; // number of columns in the Vandermonde
                                    // np.vander(x, N): columns are x^{N-1}, ..., x^1, x^0.
    let mut x = Array2::<f64>::zeros((n, deg));
    for i in 0..n {
        let t = if n == 1 {
            -1.0
        } else {
            -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)
        };
        for j in 0..deg {
            let power = (deg - 1 - j) as i32;
            x[[i, j]] = t.powi(power);
        }
    }
    resid_on(y, &x)
}

/// Residuals of `y` regressed on `x` via the pseudo-inverse:
/// `r = y - x pinv(x) y`. If `x` has no columns, returns `y` unchanged
/// (matching the reference `resid`).
fn resid_on(y: &Array2<f64>, x: &Array2<f64>) -> Result<Array2<f64>> {
    if x.dim().1 == 0 {
        return Ok(y.clone());
    }
    let (xpinv, _) = pinv(x)?; // (cols, rows)
    let fitted = x.dot(&xpinv.dot(y));
    Ok(y - &fitted)
}

/// Stack `lags` columns of lagged copies of `x` (reference `lagmat` with
/// `trim="forward"`, the default): `out[i] = [x[i-1], x[i-2], ..., x[i-lags]]`
/// with zeros where the index is out of range; row count equals `x`'s row
/// count.
fn lagmat(x: &Array2<f64>, lags: usize) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut out = Array2::<f64>::zeros((n, k * lags));
    for i in 0..n {
        for l in 1..=lags {
            if i >= l {
                for c in 0..k {
                    out[[i, (l - 1) * k + c]] = x[[i - l, c]];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A short cointegrated bivariate series: two random walks sharing a common
    /// stochastic trend so that `y1 - y2` is stationary.
    fn coint_series() -> Array2<f64> {
        array![
            [0.0, 0.1],
            [0.5, 0.4],
            [0.3, 0.5],
            [0.9, 0.7],
            [1.2, 1.3],
            [1.0, 1.1],
            [1.6, 1.5],
            [1.9, 2.0],
            [2.1, 1.9],
            [2.5, 2.6],
            [2.3, 2.4],
            [2.9, 3.0],
            [3.2, 3.1],
            [3.0, 3.2],
            [3.6, 3.5],
            [3.9, 4.0],
        ]
    }

    #[test]
    fn beta_is_identity_normalized() {
        let res = Vecm::new(coint_series(), 1, 1).unwrap().fit().unwrap();
        // First r rows of beta form the identity (here a single 1.0).
        assert!((res.beta[[0, 0]] - 1.0).abs() < 1e-12);
        assert_eq!(res.beta.dim(), (2, 1));
        assert_eq!(res.alpha.dim(), (2, 1));
        assert_eq!(res.gamma.dim(), (2, 2)); // K * k_ar_diff = 2 * 1
        assert_eq!(res.det_coef.dim().1, 0);
        assert_eq!(res.det_coef_coint.dim().0, 0);
    }

    #[test]
    fn pi_has_reduced_rank() {
        let res = Vecm::new(coint_series(), 1, 1).unwrap().fit().unwrap();
        // Pi = alpha beta^T should be 2x2 with (numerically) rank 1, so its
        // determinant is ~0.
        let pi = res.alpha.dot(&res.beta.t());
        let det = pi[[0, 0]] * pi[[1, 1]] - pi[[0, 1]] * pi[[1, 0]];
        assert!(det.abs() < 1e-9, "Pi determinant not ~0: {det}");
    }

    #[test]
    fn full_rank_recovers_sigma() {
        // At full rank the residual covariance is positive definite.
        let res = Vecm::new(coint_series(), 1, 2).unwrap().fit().unwrap();
        assert_eq!(res.coint_rank, 2);
        let l = cholesky(&res.sigma_u);
        assert!(l.is_ok(), "sigma_u not positive definite at full rank");
    }

    #[test]
    fn johansen_eigenvalues_in_unit_interval_and_descending() {
        let jr = coint_johansen(&coint_series(), 0, 1).unwrap();
        assert_eq!(jr.eig.len(), 2);
        for &e in jr.eig.iter() {
            assert!((0.0..=1.0).contains(&e), "eigenvalue out of [0,1]: {e}");
        }
        assert!(jr.eig[0] >= jr.eig[1] - 1e-15, "eigenvalues not descending");
        // Trace statistic at rank 0 dominates the max-eigenvalue statistic.
        assert!(jr.lr1[0] >= jr.lr2[0] - 1e-9);
    }

    #[test]
    fn rejects_invalid_rank() {
        assert!(Vecm::new(coint_series(), 1, 0).is_err());
        assert!(Vecm::new(coint_series(), 1, 3).is_err());
    }

    #[test]
    fn detrend_order_minus_one_is_identity() {
        let y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let d = detrend(&y, -1).unwrap();
        assert_eq!(d, y);
    }

    #[test]
    fn detrend_constant_removes_mean() {
        let y = array![[1.0], [2.0], [3.0], [4.0]];
        // Regressing on a constant removes the mean.
        let d = detrend(&y, 0).unwrap();
        let s: f64 = d.iter().sum();
        assert!(
            s.abs() < 1e-12,
            "residuals after constant detrend not zero-mean"
        );
    }
}
