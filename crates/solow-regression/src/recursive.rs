//! Recursive least squares (RLS).
//!
//! The regression coefficients are updated observation-by-observation as data
//! arrives. Equivalently, this is the exact-diffuse Kalman filter applied to the
//! state-space form of the linear model (state = the coefficient vector, a random
//! walk with zero innovation variance; observation = `yₜ = xₜᵀ β + εₜ`). That
//! formulation reproduces the canonical reference's quantities exactly:
//!
//! * the *filtered recursive coefficients* (one vector per observation),
//! * the concentrated Gaussian *log-likelihood* (`llf`, `llf_obs`) — the first
//!   `k` (= number of regressors) observations form the diffuse phase and use the
//!   diffuse forecast-variance term `−½(ln 2π + ln F∞)`,
//! * the standardized *recursive residuals*, and the
//! * *CUSUM* and *CUSUM-of-squares* structural-stability statistics.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use std::f64::consts::PI;

/// A recursive-least-squares model awaiting `fit`.
#[derive(Clone, Debug)]
pub struct RecursiveLS {
    endog: Array1<f64>,
    exog: Array2<f64>,
}

impl RecursiveLS {
    /// Build a recursive-least-squares model from response and design.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if exog.nrows() == 0 {
            return Err(Error::Value("empty design matrix".into()));
        }
        Ok(RecursiveLS { endog, exog })
    }

    /// Run the recursion and assemble all results.
    pub fn fit(&self) -> Result<RecursiveLSResults> {
        let (n, k) = self.exog.dim();
        if n < k {
            return Err(Error::Value(
                "need at least as many observations as regressors".into(),
            ));
        }

        // -- Pass 1: exact-diffuse Kalman filter --------------------------------
        // State a (k), diffuse covariance P∞ = I, stationary covariance P* = 0,
        // observation variance H = 1 (the scale is concentrated out afterwards).
        // During the diffuse phase F∞ = xₜᵀ P∞ xₜ > 0; once P∞ collapses to 0 the
        // filter switches to the ordinary recursion.
        let mut a = Array1::<f64>::zeros(k);
        let mut p_inf = Array2::<f64>::eye(k);
        let mut p_star = Array2::<f64>::zeros((k, k));
        let h = 1.0;

        // Standardized recursive residuals vₜ / √Fₜ (only defined post-diffuse;
        // zero during the diffuse phase, matching the reference).
        let mut resid_recursive = Array1::<f64>::zeros(n);
        // Per-observation forecast variance (for the concentrated llf below).
        let mut diffuse = vec![false; n];
        let mut f_used = Array1::<f64>::zeros(n); // F∞ in diffuse phase, F* after
        let mut v_used = Array1::<f64>::zeros(n); // forecast error vₜ
                                                  // Recursive coefficient path: filtered state after each update.
        let mut coefs = Array2::<f64>::zeros((n, k));
        let mut nobs_diffuse = 0usize;

        for t in 0..n {
            let xt = self.exog.row(t).to_owned();
            let v = self.endog[t] - xt.dot(&a);
            let m_inf = p_inf.dot(&xt);
            let m_star = p_star.dot(&xt);
            let f_inf = xt.dot(&m_inf);
            let f_star = xt.dot(&m_star) + h;
            v_used[t] = v;

            if f_inf > 1e-12 {
                // Diffuse update (Durbin–Koopman exact-diffuse recursion).
                diffuse[t] = true;
                nobs_diffuse += 1;
                f_used[t] = f_inf;
                let k0 = &m_inf / f_inf;
                a = &a + &(&k0 * v);
                // P* = P* + M∞ M∞ᵀ F*/F∞² − (M* M∞ᵀ + M∞ M*ᵀ)/F∞
                let outer_inf = outer(&m_inf, &m_inf);
                let cross = &outer(&m_star, &m_inf) + &outer(&m_inf, &m_star);
                p_star = &p_star + &(&outer_inf * (f_star / (f_inf * f_inf))) - &(&cross / f_inf);
                // P∞ = P∞ − M∞ M∞ᵀ / F∞
                p_inf = &p_inf - &(&outer_inf / f_inf);
                // recursive residual undefined during diffuse phase
                resid_recursive[t] = 0.0;
            } else {
                // Ordinary Kalman/RLS update.
                f_used[t] = f_star;
                let kk = &m_star / f_star;
                a = &a + &(&kk * v);
                p_star = &p_star - &(&outer(&m_star, &m_star) / f_star);
                resid_recursive[t] = v / f_star.sqrt();
            }
            for j in 0..k {
                coefs[[t, j]] = a[j];
            }
        }

        let params = a.clone();

        // -- Concentrated scale from post-diffuse standardized residuals --------
        let d = nobs_diffuse;
        let n_eff = (n - d) as f64;
        let mut ssr = 0.0;
        for t in d..n {
            ssr += resid_recursive[t] * resid_recursive[t];
        }
        let scale = ssr / n_eff;

        // -- Per-observation concentrated log-likelihood ------------------------
        // Diffuse obs: −½(ln 2π + ln F∞).
        // Post-diffuse obs: −½(ln 2π + ln(scale·F*) + vₜ²/(scale·F*)).
        let mut llf_obs = Array1::<f64>::zeros(n);
        for t in 0..n {
            if diffuse[t] {
                llf_obs[t] = -0.5 * ((2.0 * PI).ln() + f_used[t].ln());
            } else {
                let sf = scale * f_used[t];
                llf_obs[t] = -0.5 * ((2.0 * PI).ln() + sf.ln() + v_used[t] * v_used[t] / sf);
            }
        }
        let llf = llf_obs.sum();

        // -- CUSUM / CUSUM-of-squares over post-diffuse recursive residuals -----
        let post: Vec<f64> = (d..n).map(|t| resid_recursive[t]).collect();
        let m = post.len();
        // Sample std with ddof = 1 (matching np.std(..., ddof=1)).
        let mean: f64 = post.iter().sum::<f64>() / m as f64;
        let var: f64 = post.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
        let sd = var.sqrt();
        let mut cusum = Array1::<f64>::zeros(m);
        let mut acc = 0.0;
        for (i, &w) in post.iter().enumerate() {
            acc += w;
            cusum[i] = acc / sd;
        }
        let mut cusum_squares = Array1::<f64>::zeros(m);
        let mut acc2 = 0.0;
        let denom: f64 = post.iter().map(|w| w * w).sum();
        for (i, &w) in post.iter().enumerate() {
            acc2 += w * w;
            cusum_squares[i] = acc2 / denom;
        }

        Ok(RecursiveLSResults {
            params,
            recursive_coefficients: coefs,
            resid_recursive,
            llf,
            llf_obs,
            scale,
            nobs_diffuse: d,
            nobs: n as f64,
            k_exog: k,
            cusum,
            cusum_squares,
        })
    }
}

/// The fitted result of a [`RecursiveLS`].
#[derive(Clone, Debug)]
pub struct RecursiveLSResults {
    /// Final (full-sample) coefficient estimates.
    pub params: Array1<f64>,
    /// Filtered recursive coefficients — one row per observation.
    pub recursive_coefficients: Array2<f64>,
    /// Standardized recursive residuals (zero during the diffuse phase).
    pub resid_recursive: Array1<f64>,
    /// Concentrated Gaussian log-likelihood.
    pub llf: f64,
    /// Per-observation log-likelihood contributions (sums to `llf`).
    pub llf_obs: Array1<f64>,
    /// Concentrated scale (residual variance) estimate.
    pub scale: f64,
    /// Number of diffuse-phase observations (equals the number of regressors).
    pub nobs_diffuse: usize,
    /// Number of observations.
    pub nobs: f64,
    /// Number of regressors.
    pub k_exog: usize,
    /// CUSUM of standardized recursive residuals (`nobs − nobs_diffuse` long).
    pub cusum: Array1<f64>,
    /// CUSUM of squared standardized recursive residuals (ending at 1).
    pub cusum_squares: Array1<f64>,
}

/// Outer product `a bᵀ`.
fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = a[i] * b[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn final_coefficients_equal_full_sample_ols() {
        // RLS final state must equal the OLS fit on all observations.
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 7.0]
        ];
        let y = array![1.1, 2.9, 5.2, 8.8, 11.1, 15.3];
        let res = RecursiveLS::new(y.clone(), x.clone())
            .unwrap()
            .fit()
            .unwrap();

        // Closed-form OLS via normal equations.
        let xtx = x.t().dot(&x);
        let xtxi = solow_linalg::inv(&xtx).unwrap();
        let ols = xtxi.dot(&x.t().dot(&y));
        for j in 0..2 {
            assert!((res.params[j] - ols[j]).abs() < 1e-9, "coef {j}");
        }
        assert_eq!(res.nobs_diffuse, 2);
    }

    #[test]
    fn diffuse_phase_residuals_are_zero_and_cusum_squares_ends_at_one() {
        let x = array![
            [1.0, 0.5],
            [1.0, -0.3],
            [1.0, 1.2],
            [1.0, 0.8],
            [1.0, -1.1],
            [1.0, 0.2]
        ];
        let y = array![0.9, 0.1, 2.0, 1.7, -1.3, 0.6];
        let res = RecursiveLS::new(y, x).unwrap().fit().unwrap();
        // First k residuals are zero (diffuse).
        for t in 0..res.nobs_diffuse {
            assert_eq!(res.resid_recursive[t], 0.0);
        }
        // CUSUM-of-squares is a fraction ending exactly at 1.
        let last = res.cusum_squares[res.cusum_squares.len() - 1];
        assert!((last - 1.0).abs() < 1e-12);
        // llf_obs sums to llf.
        assert!((res.llf_obs.sum() - res.llf).abs() < 1e-12);
    }
}
