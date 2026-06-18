//! Dynamic-factor model with a single common factor, fit by maximum likelihood.
//!
//! Several observed series `y_t` (length `p`) are driven by one latent factor
//! `f_t` that follows an autoregression of order `factor_order`:
//!
//! ```text
//!   y_t = Lambda f_t + eps_t,          eps_t ~ N(0, diag(sigma^2_1..sigma^2_p))
//!   f_t = phi_1 f_{t-1} + ... + phi_o f_{t-o} + eta_t,   eta_t ~ N(0, 1)
//! ```
//!
//! with `Lambda` the `p`-vector of factor loadings. The factor AR is cast into
//! companion form so the state is `(f_t, f_{t-1}, ..., f_{t-o+1})`; the factor
//! disturbance variance is fixed to one for identification (the loadings absorb
//! the scale). The model is started from its stationary distribution (the
//! discrete-Lyapunov solution) and estimated with the multivariate Kalman
//! filter in [`crate::mvkalman`].
//!
//! Estimation maximizes the exact Gaussian log-likelihood. The loadings are
//! free, each idiosyncratic variance is the square of its coordinate, and the
//! factor AR is mapped through the Monahan stationary transform so the iterates
//! always stay stationary.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::solve;
use solow_optimize::{approx_fprime, minimize_bfgs};

use crate::mvkalman::MvStateSpace;

/// A dynamic-factor model bound to a multivariate panel.
#[derive(Clone, Debug)]
pub struct DynamicFactor {
    /// Observations, `n x p` (one observation vector per row).
    endog: Array2<f64>,
    /// Factor autoregressive order.
    factor_order: usize,
}

/// A fitted dynamic-factor model.
#[derive(Clone, Debug)]
pub struct DynamicFactorResults {
    /// Estimated parameters ordered `[loading_1..loading_p, sigma2_1..sigma2_p, ar_1..ar_o]`.
    pub params: Array1<f64>,
    /// Maximized log-likelihood.
    pub llf: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,
    /// Hannan-Quinn information criterion.
    pub hqic: f64,
    /// Whether the optimizer's function-value test was satisfied.
    pub converged: bool,
    /// Number of observations.
    pub nobs: usize,
}

impl DynamicFactor {
    /// Build a single-factor model for the `n x p` panel `endog` with the given
    /// factor autoregressive order.
    pub fn new(endog: Array2<f64>, factor_order: usize) -> Result<Self> {
        if endog.nrows() == 0 || endog.ncols() == 0 {
            return Err(Error::Value("empty panel".into()));
        }
        if factor_order == 0 {
            return Err(Error::Value("factor_order must be >= 1".into()));
        }
        Ok(DynamicFactor {
            endog,
            factor_order,
        })
    }

    /// Number of observed series `p`.
    fn k_endog(&self) -> usize {
        self.endog.ncols()
    }

    /// Total number of estimated parameters: `p` loadings + `p` variances + `o` AR.
    fn k_params(&self) -> usize {
        2 * self.k_endog() + self.factor_order
    }

    /// Evaluate the log-likelihood at the given natural parameters.
    pub fn loglike(&self, params: &Array1<f64>) -> Option<f64> {
        let ss = self.build_state_space(params).ok()?;
        match ss.loglike(&self.endog, 0) {
            Ok(ll) if ll.is_finite() => Some(ll),
            _ => None,
        }
    }

    /// Fit by maximum likelihood. Loadings are free; idiosyncratic variances are
    /// squares of their coordinates; the factor AR uses the Monahan stationary
    /// map. The objective `-loglike` is minimized with BFGS.
    pub fn fit(&self) -> Result<DynamicFactorResults> {
        let p = self.k_endog();
        let o = self.factor_order;
        let k = self.k_params();
        let nobs = self.endog.nrows();

        let neg_ll = |u: &Array1<f64>| -> f64 {
            let params = self.transform_params(u);
            match self.loglike(&params) {
                Some(ll) if ll.is_finite() => -ll,
                _ => f64::INFINITY,
            }
        };
        let grad = |u: &Array1<f64>| approx_fprime(u, neg_ll);

        // Start: loadings from a simple regression scale, variances at the
        // per-series variance, AR at zero (Monahan: 0 -> 0).
        let mut start = Array1::<f64>::zeros(k);
        for j in 0..p {
            let col = self.endog.column(j);
            let mean = col.sum() / nobs as f64;
            let var = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / nobs as f64;
            // loading start (unconstrained, raw): a modest positive value.
            start[j] = var.sqrt().max(1e-2);
            // variance start (unconstrained = sqrt of natural): half the series var.
            start[p + j] = (0.5 * var).max(1e-4).sqrt();
        }

        let mut u_hat = start.clone();
        let mut f_prev = neg_ll(&u_hat);
        let mut converged = false;
        for _ in 0..600 {
            let res = minimize_bfgs(&u_hat, neg_ll, grad, 25, 1e-12)?;
            u_hat = res.x;
            let f_now = res.fval;
            if res.converged || (f_prev - f_now).abs() <= 1e-13 * (1.0 + f_now.abs()) {
                converged = true;
                break;
            }
            f_prev = f_now;
        }

        let params = self.transform_params(&u_hat);
        let llf = self
            .loglike(&params)
            .ok_or_else(|| Error::Convergence("log-likelihood undefined at optimum".into()))?;

        let kf = k as f64;
        let n = nobs as f64;
        let aic = -2.0 * llf + 2.0 * kf;
        let bic = -2.0 * llf + kf * n.ln();
        let hqic = -2.0 * llf + 2.0 * kf * n.ln().ln();

        let _ = o;
        Ok(DynamicFactorResults {
            params,
            llf,
            aic,
            bic,
            hqic,
            converged,
            nobs,
        })
    }

    /// Map unconstrained optimizer coordinates to natural parameters.
    fn transform_params(&self, u: &Array1<f64>) -> Array1<f64> {
        let p = self.k_endog();
        let o = self.factor_order;
        let mut out = Array1::<f64>::zeros(u.len());
        // Loadings: identity.
        for j in 0..p {
            out[j] = u[j];
        }
        // Variances: square.
        for j in 0..p {
            out[p + j] = u[p + j] * u[p + j];
        }
        // Factor AR: Monahan stationary map.
        let ar_block = u.slice(ndarray::s![2 * p..2 * p + o]).to_owned();
        let ar = constrain_stationary(&ar_block);
        for j in 0..o {
            out[2 * p + j] = ar[j];
        }
        out
    }

    /// Build the multivariate state space for the given natural parameters.
    pub fn build_state_space(&self, params: &Array1<f64>) -> Result<MvStateSpace> {
        let p = self.k_endog();
        let o = self.factor_order;
        let m = o; // companion state dimension for a single factor.

        let loadings = params.slice(ndarray::s![0..p]).to_owned();
        let variances = params.slice(ndarray::s![p..2 * p]).to_owned();
        let ar = params.slice(ndarray::s![2 * p..2 * p + o]).to_owned();

        // Design: Z[i, 0] = loading_i; other columns zero.
        let mut design = Array2::<f64>::zeros((p, m));
        for i in 0..p {
            design[[i, 0]] = loadings[i];
        }

        // Observation covariance: diagonal of idiosyncratic variances.
        let mut obs_cov = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            obs_cov[[i, i]] = variances[i];
        }

        // Transition: companion form. First row holds the AR coefficients; a
        // sub-diagonal of ones shifts the lagged factors.
        let mut transition = Array2::<f64>::zeros((m, m));
        for (j, &phi) in ar.iter().enumerate() {
            transition[[0, j]] = phi;
        }
        for i in 1..m {
            transition[[i, i - 1]] = 1.0;
        }

        // Selection: the unit factor disturbance enters the first state only.
        let mut selection = Array2::<f64>::zeros((m, 1));
        selection[[0, 0]] = 1.0;
        let state_cov = Array2::from_elem((1, 1), 1.0);

        // Stationary initialization: P = T P T' + R Q R'.
        let rqr = selection.dot(&state_cov).dot(&selection.t());
        let init_cov = solve_discrete_lyapunov(&transition, &rqr)?;
        let init_state = Array1::<f64>::zeros(m);

        Ok(MvStateSpace {
            transition,
            selection,
            state_cov,
            design,
            obs_cov,
            init_state,
            init_cov,
        })
    }
}

/// Monahan (1984) map from unconstrained reals to stationary AR coefficients,
/// using the same convention as the reference factor-AR transform.
fn constrain_stationary(u: &Array1<f64>) -> Array1<f64> {
    let n = u.len();
    if n == 0 {
        return Array1::<f64>::zeros(0);
    }
    let r: Vec<f64> = u.iter().map(|&v| v / (1.0 + v * v).sqrt()).collect();
    let mut y = vec![vec![0.0_f64; n]; n];
    for k in 0..n {
        for i in 0..k {
            y[k][i] = y[k - 1][i] + r[k] * y[k - 1][k - i - 1];
        }
        y[k][k] = r[k];
    }
    Array1::from_iter((0..n).map(|i| -y[n - 1][i]))
}

/// Solve `P = T P T' + C` via `vec(P) = (I - T ⊗ T)^{-1} vec(C)`.
fn solve_discrete_lyapunov(t: &Array2<f64>, c: &Array2<f64>) -> Result<Array2<f64>> {
    let m = t.nrows();
    let m2 = m * m;
    let mut a = Array2::<f64>::eye(m2);
    for i in 0..m {
        for j in 0..m {
            let tij = t[[i, j]];
            if tij == 0.0 {
                continue;
            }
            for pr in 0..m {
                for q in 0..m {
                    a[[i * m + pr, j * m + q]] -= tij * t[[pr, q]];
                }
            }
        }
    }
    let mut vec_c = Array1::<f64>::zeros(m2);
    for i in 0..m {
        for pr in 0..m {
            vec_c[i * m + pr] = c[[i, pr]];
        }
    }
    let vec_p = solve(&a, &vec_c)?;
    let mut pcov = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            pcov[[i, j]] = vec_p[i * m + j];
        }
    }
    for i in 0..m {
        for j in (i + 1)..m {
            let v = 0.5 * (pcov[[i, j]] + pcov[[j, i]]);
            pcov[[i, j]] = v;
            pcov[[j, i]] = v;
        }
    }
    Ok(pcov)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn state_space_order1_shapes() {
        let y = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let m = DynamicFactor::new(y, 1).unwrap();
        // params: [load1, load2, sig1, sig2, ar1]
        let ss = m
            .build_state_space(&array![1.5, -0.5, 2.0, 3.0, 0.4])
            .unwrap();
        // Design is p x m (2 x 1).
        assert_eq!(ss.design.dim(), (2, 1));
        assert_eq!(ss.design[[0, 0]], 1.5);
        assert_eq!(ss.design[[1, 0]], -0.5);
        assert_eq!(ss.obs_cov, array![[2.0, 0.0], [0.0, 3.0]]);
        assert_eq!(ss.transition, array![[0.4]]);
    }

    #[test]
    fn state_space_order2_companion() {
        let y = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let m = DynamicFactor::new(y, 2).unwrap();
        // params: [load1, load2, sig1, sig2, ar1, ar2]
        let ss = m
            .build_state_space(&array![1.0, 2.0, 1.0, 1.0, 0.5, -0.3])
            .unwrap();
        assert_eq!(ss.transition, array![[0.5, -0.3], [1.0, 0.0]]);
        assert_eq!(ss.design.dim(), (2, 2));
        assert_eq!(ss.design[[0, 0]], 1.0);
        assert_eq!(ss.design[[1, 0]], 2.0);
        // Lagged-factor column of the design is zero.
        assert_eq!(ss.design[[0, 1]], 0.0);
    }

    #[test]
    fn constrain_matches_known_value() {
        // u = 0.5 -> -0.5/sqrt(1.25) = -0.4472135955.
        let c = constrain_stationary(&array![0.5]);
        assert!((c[0] - (-0.447_213_595_5)).abs() < 1e-9, "{}", c[0]);
    }
}
