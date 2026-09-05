//! First-order `k`-regime Markov switching regression.
//!
//! ```text
//! y_t = a_{S_t} + x_t' b_{S_t} + e_t,   e_t ~ N(0, sigma^2_{S_t})
//! ```
//!
//! The intercept and (optional) regression coefficients switch across regimes;
//! the error variance optionally switches as well. Estimation maximises the
//! Hamilton-filter log-likelihood.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

use crate::filter::{hamilton_filter, kim_smoother};
use crate::switching::{
    maximize, ols, steady_state, transform_transition, transition_matrix, transition_param_names,
    untransform_transition, var_pop, MarkovResults,
};

/// A first-order `k`-regime Markov switching regression awaiting estimation.
#[derive(Clone, Debug)]
pub struct MarkovRegression {
    endog: Array1<f64>,
    /// Design matrix including the leading column of ones (the intercept).
    exog: Array2<f64>,
    k_regimes: usize,
    k_exog: usize,
    switching_variance: bool,
    maxiter: usize,
}

impl MarkovRegression {
    /// Build a model with a switching intercept (and switching variance flag).
    ///
    /// A constant column is prepended internally; `endog` is the response.
    pub fn new(endog: Array1<f64>, k_regimes: usize, switching_variance: bool) -> Result<Self> {
        let n = endog.len();
        let exog = Array2::<f64>::ones((n, 1));
        Self::with_exog(endog, exog, k_regimes, switching_variance)
    }

    /// Build a model with an explicit design matrix `exog` (which must already
    /// include any intercept column). All coefficients switch across regimes.
    pub fn with_exog(
        endog: Array1<f64>,
        exog: Array2<f64>,
        k_regimes: usize,
        switching_variance: bool,
    ) -> Result<Self> {
        if k_regimes < 2 {
            return Err(Error::Value("k_regimes must be >= 2".into()));
        }
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        let k_exog = exog.ncols();
        Ok(MarkovRegression {
            endog,
            exog,
            k_regimes,
            k_exog,
            switching_variance,
            maxiter: 2000,
        })
    }

    /// Number of transition parameters: `k * (k - 1)`.
    fn k_trans(&self) -> usize {
        self.k_regimes * (self.k_regimes - 1)
    }

    /// Number of mean/exog parameters: `k * k_exog`.
    fn k_mean(&self) -> usize {
        self.k_regimes * self.k_exog
    }

    /// Number of variance parameters.
    fn k_var(&self) -> usize {
        if self.switching_variance {
            self.k_regimes
        } else {
            1
        }
    }

    /// Total free parameters.
    fn k_params(&self) -> usize {
        self.k_trans() + self.k_mean() + self.k_var()
    }

    /// Human-readable names matching the reference layout.
    fn param_names(&self) -> Vec<String> {
        let k = self.k_regimes;
        let mut names = transition_param_names(k);
        // Mean/exog (per regime). The first column is the intercept; any
        // additional exog columns are named `x1`, `x2`, ...
        let exog_name = |c: usize| {
            if c == 0 {
                "const".to_string()
            } else {
                format!("x{c}")
            }
        };
        for i in 0..k {
            for c in 0..self.k_exog {
                names.push(format!("{}[{i}]", exog_name(c)));
            }
        }
        // Variance.
        if self.switching_variance {
            for i in 0..k {
                names.push(format!("sigma2[{i}]"));
            }
        } else {
            names.push("sigma2".to_string());
        }
        names
    }

    /// Map unconstrained optimiser parameters to constrained model parameters.
    fn transform(&self, u: &Array1<f64>) -> Array1<f64> {
        let k = self.k_regimes;
        let mut c = u.clone();
        transform_transition(&mut c, u, k);
        // Mean/exog: identity (already copied).
        // Variance: square.
        let var_start = self.k_trans() + self.k_mean();
        for idx in var_start..self.k_params() {
            c[idx] = u[idx] * u[idx];
        }
        c
    }

    /// Inverse transform (constrained -> unconstrained).
    fn untransform(&self, c: &Array1<f64>) -> Array1<f64> {
        let k = self.k_regimes;
        let mut u = c.clone();
        untransform_transition(&mut u, c, k);
        let var_start = self.k_trans() + self.k_mean();
        for idx in var_start..self.k_params() {
            u[idx] = c[idx].sqrt();
        }
        u
    }

    /// Conditional log-likelihoods `log f(y_t | S_t = i)`, shaped `(k, nobs)`.
    fn conditional_loglik(&self, c: &Array1<f64>) -> Array2<f64> {
        let k = self.k_regimes;
        let n = self.endog.len();
        let mean_start = self.k_trans();
        let var_start = self.k_trans() + self.k_mean();
        let mut cll = Array2::<f64>::zeros((k, n));
        let two_pi = std::f64::consts::TAU;
        for i in 0..k {
            // Regime-i coefficients.
            let beta_start = mean_start + i * self.k_exog;
            let variance = if self.switching_variance {
                c[var_start + i]
            } else {
                c[var_start]
            };
            for t in 0..n {
                let mut yhat = 0.0;
                for ci in 0..self.k_exog {
                    yhat += self.exog[[t, ci]] * c[beta_start + ci];
                }
                let resid = self.endog[t] - yhat;
                cll[[i, t]] = -0.5 * resid * resid / variance - 0.5 * (two_pi * variance).ln();
            }
        }
        cll
    }

    /// Negative log-likelihood at the unconstrained parameter vector.
    fn neg_loglike(&self, u: &Array1<f64>) -> Result<f64> {
        let c = self.transform(u);
        let c_s = c
            .as_slice()
            .ok_or_else(|| Error::Value("params must be contiguous".into()))?;
        let p = transition_matrix(c_s, self.k_regimes);
        let init = steady_state(&p)?;
        let cll = self.conditional_loglik(&c);
        let out = hamilton_filter(&init, &p, &cll, 0);
        Ok(-out.llf)
    }

    /// Reasonable starting parameters in constrained space, mirroring the
    /// reference: equal transition probabilities, OLS interpolated across
    /// regimes, variance(s) seeded from the OLS residual variance.
    fn start_params(&self) -> Result<Array1<f64>> {
        let k = self.k_regimes;
        let mut c = Array1::<f64>::zeros(self.k_params());
        // Equal transition probabilities: each column uniform 1/k.
        for j in 0..k {
            for i in 0..(k - 1) {
                c[j * (k - 1) + i] = 1.0 / k as f64;
            }
        }
        // OLS beta and residual variance.
        let beta = ols(&self.exog, &self.endog)?;
        let yhat = self.exog.dot(&beta);
        let resid = &self.endog - &yhat;
        let variance = var_pop(&resid);
        let mean_start = self.k_trans();
        for i in 0..k {
            let frac = i as f64 / k as f64;
            for ci in 0..self.k_exog {
                c[mean_start + i * self.k_exog + ci] = beta[ci] * frac;
            }
        }
        let var_start = self.k_trans() + self.k_mean();
        if self.switching_variance {
            for i in 0..k {
                // linspace(variance/10, variance, k).
                let t = if k == 1 {
                    0.0
                } else {
                    i as f64 / (k as f64 - 1.0)
                };
                c[var_start + i] = variance / 10.0 + t * (variance - variance / 10.0);
            }
        } else {
            c[var_start] = variance;
        }
        Ok(c)
    }

    /// Fit by maximum likelihood (BFGS on the negative Hamilton log-likelihood).
    pub fn fit(&self) -> Result<MarkovResults> {
        self.fit_from(None)
    }

    /// Fit, optionally starting the optimiser from a constrained parameter guess.
    pub fn fit_from(&self, start_constrained: Option<Array1<f64>>) -> Result<MarkovResults> {
        let start_c = match start_constrained {
            Some(c) => c,
            None => self.start_params()?,
        };
        let start_u = self.untransform(&start_c);

        let (xopt, _negll, converged) = maximize(
            &start_u,
            |u| self.neg_loglike(u).unwrap_or(1e10),
            self.maxiter,
        );

        let c = self.transform(&xopt);
        let c_s = c
            .as_slice()
            .ok_or_else(|| Error::Value("params must be contiguous".into()))?;
        let p = transition_matrix(c_s, self.k_regimes);
        let init = steady_state(&p)?;
        let cll = self.conditional_loglik(&c);
        let out = hamilton_filter(&init, &p, &cll, 0);
        let smoothed = kim_smoother(&p, &out, self.k_regimes, 0);

        Ok(MarkovResults::new(
            c,
            self.param_names(),
            out.llf,
            self.endog.len(),
            p,
            init,
            out.filtered_marginal,
            smoothed,
            converged,
        ))
    }
}
