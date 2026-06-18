//! Unobserved-components (structural time-series) models, fit by maximum
//! likelihood.
//!
//! A structural model decomposes a univariate series into unobserved level,
//! trend and seasonal components plus an irregular term. The two level
//! specifications implemented here are
//!
//! * **local level** — a random-walk level plus irregular noise,
//!
//!   ```text
//!     y_t   = mu_t + eps_t,        eps_t ~ N(0, sigma^2_irregular)
//!     mu_t  = mu_{t-1} + xi_t,     xi_t  ~ N(0, sigma^2_level)
//!   ```
//!
//! * **local linear trend** — a random-walk level with a random-walk slope,
//!
//!   ```text
//!     y_t    = mu_t + eps_t
//!     mu_t   = mu_{t-1} + beta_{t-1} + xi_t
//!     beta_t = beta_{t-1} + zeta_t,  zeta_t ~ N(0, sigma^2_trend)
//!   ```
//!
//! An optional **stochastic seasonal** of period `s` adds a dummy-variable
//! seasonal whose `s-1` coefficients sum to a zero-mean disturbance with
//! variance `sigma^2_seasonal`.
//!
//! The nonstationary states are started with an *approximate-diffuse*
//! covariance (a large multiple of the identity) and the first
//! `loglike_burn` (= number of nonstationary states) observations are excluded
//! from the log-likelihood, matching the reference implementation. Estimation
//! maximizes the exact Gaussian log-likelihood from the scalar [`crate::kalman`]
//! filter; each variance is parametrized as the square of an unconstrained real
//! so it stays nonnegative.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_optimize::{approx_fprime, minimize_bfgs};

use crate::kalman::StateSpace;

/// The level specification of an [`UnobservedComponents`] model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Random-walk level (one state, one nonstationary).
    LocalLevel,
    /// Random-walk level with random-walk slope (two states, two nonstationary).
    LocalLinearTrend,
}

/// Specification of an unobserved-components model.
#[derive(Clone, Copy, Debug)]
pub struct UcSpec {
    /// Level/trend specification.
    pub level: Level,
    /// Stochastic-seasonal period (`0` for no seasonal).
    pub seasonal: usize,
}

impl UcSpec {
    /// A model with the given level and no seasonal component.
    pub fn new(level: Level) -> Self {
        UcSpec { level, seasonal: 0 }
    }

    /// Add a stochastic seasonal of period `s`.
    pub fn with_seasonal(mut self, s: usize) -> Self {
        self.seasonal = s;
        self
    }

    /// Number of latent states.
    fn k_states(&self) -> usize {
        let level = match self.level {
            Level::LocalLevel => 1,
            Level::LocalLinearTrend => 2,
        };
        let seas = if self.seasonal > 0 {
            self.seasonal - 1
        } else {
            0
        };
        level + seas
    }

    /// Number of nonstationary states (the `loglike_burn`).
    fn burn(&self) -> usize {
        let level = match self.level {
            Level::LocalLevel => 1,
            Level::LocalLinearTrend => 2,
        };
        let seas = if self.seasonal > 0 {
            self.seasonal - 1
        } else {
            0
        };
        level + seas
    }

    /// Number of estimated variance parameters.
    ///
    /// Order: `[sigma2.irregular, sigma2.level, (sigma2.trend), (sigma2.seasonal)]`.
    fn k_params(&self) -> usize {
        let mut k = 2; // irregular + level
        if self.level == Level::LocalLinearTrend {
            k += 1; // trend
        }
        if self.seasonal > 0 {
            k += 1; // seasonal
        }
        k
    }
}

/// Approximate-diffuse initial variance for the nonstationary states.
const DIFFUSE_VARIANCE: f64 = 1e6;

/// A fitted unobserved-components model.
#[derive(Clone, Debug)]
pub struct UcResults {
    /// Estimated variance parameters, in the order described by [`UcSpec`].
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

/// An unobserved-components model bound to an observed series.
#[derive(Clone, Debug)]
pub struct UnobservedComponents {
    endog: Array1<f64>,
    spec: UcSpec,
}

impl UnobservedComponents {
    /// Build a model for `endog` with the given specification.
    pub fn new(endog: Array1<f64>, spec: UcSpec) -> Result<Self> {
        if endog.is_empty() {
            return Err(Error::Value("empty series".into()));
        }
        if spec.seasonal == 1 {
            return Err(Error::Value("seasonal period must be >= 2".into()));
        }
        Ok(UnobservedComponents { endog, spec })
    }

    /// Evaluate the log-likelihood at the given natural variance parameters.
    pub fn loglike(&self, params: &Array1<f64>) -> Option<f64> {
        if params.iter().any(|&v| v < 0.0) {
            return None;
        }
        let ss = self.build_state_space(params);
        let out = ss.filter(&self.endog, self.spec.burn());
        if out.loglike.is_finite() {
            Some(out.loglike)
        } else {
            None
        }
    }

    /// Fit by maximum likelihood, optimizing `-loglike` with BFGS over the
    /// unconstrained space where each variance is the square of its coordinate.
    pub fn fit(&self) -> Result<UcResults> {
        let spec = self.spec;
        let k = spec.k_params();
        let nobs = self.endog.len();

        let neg_ll = |u: &Array1<f64>| -> f64 {
            let p = u.mapv(|v| v * v);
            match self.loglike(&p) {
                Some(ll) if ll.is_finite() => -ll,
                _ => f64::INFINITY,
            }
        };
        let grad = |u: &Array1<f64>| approx_fprime(u, neg_ll);

        // Start every variance at the sample variance of the data (a robust,
        // reference-like start).
        let var = sample_var(&self.endog).max(1e-4);
        let start = Array1::from_elem(k, var.sqrt());

        // Restart BFGS in bursts, stopping on an ftol rule (the finite-diff
        // gradient cannot reach the optimizer's gtol).
        let mut u_hat = start.clone();
        let mut f_prev = neg_ll(&u_hat);
        let mut converged = false;
        for _ in 0..400 {
            let res = minimize_bfgs(&u_hat, neg_ll, grad, 25, 1e-12)?;
            u_hat = res.x;
            let f_now = res.fval;
            if res.converged || (f_prev - f_now).abs() <= 1e-13 * (1.0 + f_now.abs()) {
                converged = true;
                break;
            }
            f_prev = f_now;
        }

        let params = u_hat.mapv(|v| v * v);
        let llf = self
            .loglike(&params)
            .ok_or_else(|| Error::Convergence("log-likelihood undefined at optimum".into()))?;

        let kf = k as f64;
        let n = nobs as f64;
        let aic = -2.0 * llf + 2.0 * kf;
        let bic = -2.0 * llf + kf * n.ln();
        let hqic = -2.0 * llf + 2.0 * kf * n.ln().ln();

        Ok(UcResults {
            params,
            llf,
            aic,
            bic,
            hqic,
            converged,
            nobs,
        })
    }

    /// Build the scalar state space for the given natural parameters.
    ///
    /// The variance parameters appear in the order
    /// `[irregular, level, (trend), (seasonal)]`.
    pub fn build_state_space(&self, params: &Array1<f64>) -> StateSpace {
        let spec = self.spec;
        let m = spec.k_states();

        let mut idx = 0;
        let sigma2_irregular = params[idx];
        idx += 1;
        let sigma2_level = params[idx];
        idx += 1;
        let sigma2_trend = if spec.level == Level::LocalLinearTrend {
            let v = params[idx];
            idx += 1;
            Some(v)
        } else {
            None
        };
        let sigma2_seasonal = if spec.seasonal > 0 {
            Some(params[idx])
        } else {
            None
        };

        // Transition T, design Z, and the columns of the selection R that carry
        // state disturbances.
        let mut transition = Array2::<f64>::zeros((m, m));
        let mut design = Array1::<f64>::zeros(m);

        // Level block.
        match spec.level {
            Level::LocalLevel => {
                transition[[0, 0]] = 1.0;
                design[0] = 1.0;
            }
            Level::LocalLinearTrend => {
                transition[[0, 0]] = 1.0;
                transition[[0, 1]] = 1.0;
                transition[[1, 1]] = 1.0;
                design[0] = 1.0;
            }
        }
        let level_states = match spec.level {
            Level::LocalLevel => 1,
            Level::LocalLinearTrend => 2,
        };

        // Seasonal block (dummy variable form): s-1 states. The first seasonal
        // state appears in the design; the transition is
        //   gamma_t = -(gamma_{t-1} + ... + gamma_{t-s+1}) + omega_t.
        if spec.seasonal > 0 {
            let s = spec.seasonal;
            let base = level_states;
            // First seasonal-state row: -1 across all seasonal states.
            for j in 0..(s - 1) {
                transition[[base, base + j]] = -1.0;
            }
            // Shift rows: gamma_{i} <- gamma_{i-1}.
            for i in 1..(s - 1) {
                transition[[base + i, base + i - 1]] = 1.0;
            }
            design[base] = 1.0;
        }

        // Selection / state covariance: place each disturbance variance on the
        // states that receive a shock. The order of disturbance columns is
        // [level, (trend), (seasonal-first)].
        let mut shock_states: Vec<usize> = Vec::new();
        let mut shock_vars: Vec<f64> = Vec::new();
        // Level disturbance acts on state 0.
        shock_states.push(0);
        shock_vars.push(sigma2_level);
        if let Some(st) = sigma2_trend {
            // Trend disturbance acts on the slope state (index 1).
            shock_states.push(1);
            shock_vars.push(st);
        }
        if let Some(ss) = sigma2_seasonal {
            // Seasonal disturbance acts on the first seasonal state.
            shock_states.push(level_states);
            shock_vars.push(ss);
        }

        let r = shock_states.len();
        let mut selection = Array2::<f64>::zeros((m, r));
        let mut state_cov = Array2::<f64>::zeros((r, r));
        for (col, (&st, &var)) in shock_states.iter().zip(shock_vars.iter()).enumerate() {
            selection[[st, col]] = 1.0;
            state_cov[[col, col]] = var;
        }

        // Approximate-diffuse initialization: all states are nonstationary.
        let init_state = Array1::<f64>::zeros(m);
        let init_cov = Array2::<f64>::eye(m) * DIFFUSE_VARIANCE;

        let _ = idx;
        StateSpace {
            transition,
            selection,
            state_cov,
            design,
            obs_cov: sigma2_irregular,
            init_state,
            init_cov,
        }
    }
}

/// Population sample variance used for the variance start.
fn sample_var(y: &Array1<f64>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let mean = y.sum() / n as f64;
    y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn local_level_matrices() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let m = UnobservedComponents::new(y, UcSpec::new(Level::LocalLevel)).unwrap();
        let ss = m.build_state_space(&array![0.5, 1.3]);
        assert_eq!(ss.transition, array![[1.0]]);
        assert_eq!(ss.design, array![1.0]);
        assert_eq!(ss.obs_cov, 0.5);
        assert_eq!(ss.state_cov, array![[1.3]]);
        assert_eq!(ss.init_cov[[0, 0]], DIFFUSE_VARIANCE);
    }

    #[test]
    fn local_linear_trend_matrices() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let m = UnobservedComponents::new(y, UcSpec::new(Level::LocalLinearTrend)).unwrap();
        let ss = m.build_state_space(&array![0.5, 1.3, 0.7]);
        assert_eq!(ss.transition, array![[1.0, 1.0], [0.0, 1.0]]);
        assert_eq!(ss.design, array![1.0, 0.0]);
        // Selection picks states 0 (level) and 1 (slope).
        assert_eq!(ss.selection, array![[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(ss.state_cov, array![[1.3, 0.0], [0.0, 0.7]]);
    }

    #[test]
    fn seasonal_transition_structure() {
        // local level + seasonal period 4 -> 1 + 3 = 4 states.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spec = UcSpec::new(Level::LocalLevel).with_seasonal(4);
        let m = UnobservedComponents::new(y, spec).unwrap();
        let ss = m.build_state_space(&array![0.5, 1.3, 0.2]);
        let expect = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        assert_eq!(ss.transition, expect);
        assert_eq!(ss.design, array![1.0, 1.0, 0.0, 0.0]);
    }
}
