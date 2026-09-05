//! `k`-regime Markov switching autoregression of a given order.
//!
//! ```text
//! y_t = a_{S_t} + sum_{j=1}^p phi_{j,S_t} (y_{t-j} - a_{S_{t-j}}) + e_t,
//!     e_t ~ N(0, sigma^2_{S_t})
//! ```
//!
//! The mean `a`, the autoregressive coefficients `phi`, and optionally the
//! variance switch across regimes. The model conditions on the first `order`
//! observations. Estimation maximises the order-dependent Hamilton-filter
//! log-likelihood.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

use crate::filter::{hamilton_filter, kim_smoother};
use crate::switching::{
    maximize, ols, steady_state, transform_transition, transition_matrix, transition_param_names,
    untransform_transition, var_pop, MarkovResults,
};

/// A `k`-regime Markov switching autoregression awaiting estimation.
///
/// The mean is a switching constant (`trend = 'c'`). The autoregressive
/// coefficients and the variance may optionally switch.
#[derive(Clone, Debug)]
pub struct MarkovAutoregression {
    /// Full endogenous series (length `nobs + order`).
    endog: Array1<f64>,
    k_regimes: usize,
    order: usize,
    switching_ar: bool,
    switching_variance: bool,
    maxiter: usize,
}

impl MarkovAutoregression {
    /// Build a model. `endog` is the full series; `order >= 1`.
    pub fn new(
        endog: Array1<f64>,
        k_regimes: usize,
        order: usize,
        switching_ar: bool,
        switching_variance: bool,
    ) -> Result<Self> {
        if k_regimes < 2 {
            return Err(Error::Value("k_regimes must be >= 2".into()));
        }
        if order < 1 {
            return Err(Error::Value("order must be >= 1".into()));
        }
        if endog.len() <= order {
            return Err(Error::Value("need more observations than the order".into()));
        }
        Ok(MarkovAutoregression {
            endog,
            k_regimes,
            order,
            switching_ar,
            switching_variance,
            maxiter: 2000,
        })
    }

    fn nobs(&self) -> usize {
        self.endog.len() - self.order
    }

    fn k_trans(&self) -> usize {
        self.k_regimes * (self.k_regimes - 1)
    }

    /// Mean parameters: one switching constant per regime.
    fn k_mean(&self) -> usize {
        self.k_regimes
    }

    fn k_ar(&self) -> usize {
        if self.switching_ar {
            self.k_regimes * self.order
        } else {
            self.order
        }
    }

    fn k_var(&self) -> usize {
        if self.switching_variance {
            self.k_regimes
        } else {
            1
        }
    }

    fn k_params(&self) -> usize {
        self.k_trans() + self.k_mean() + self.k_ar() + self.k_var()
    }

    fn mean_start(&self) -> usize {
        self.k_trans()
    }

    // The reference parameter order is [transition, mean, variance, AR], so the
    // variance block precedes the autoregressive block.
    fn var_start(&self) -> usize {
        self.k_trans() + self.k_mean()
    }

    fn ar_start(&self) -> usize {
        self.k_trans() + self.k_mean() + self.k_var()
    }

    fn param_names(&self) -> Vec<String> {
        let k = self.k_regimes;
        let mut names = transition_param_names(k);
        for i in 0..k {
            names.push(format!("const[{i}]"));
        }
        // Variance precedes autoregressive in the reference layout.
        if self.switching_variance {
            for i in 0..k {
                names.push(format!("sigma2[{i}]"));
            }
        } else {
            names.push("sigma2".to_string());
        }
        if self.switching_ar {
            for i in 0..k {
                for j in 0..self.order {
                    names.push(format!("ar.L{}[{i}]", j + 1));
                }
            }
        } else {
            for j in 0..self.order {
                names.push(format!("ar.L{}", j + 1));
            }
        }
        names
    }

    /// AR coefficient(s) for regime `i` from the constrained parameter vector.
    fn ar_coeffs<'a>(&self, c: &'a Array1<f64>, i: usize) -> &'a [f64] {
        let start = if self.switching_ar {
            self.ar_start() + i * self.order
        } else {
            self.ar_start()
        };
        // SAFETY: owned contiguous constrained parameter vector, so `as_slice()`
        // is always `Some`; the empty-slice fallback would only be reached for a
        // non-contiguous input, which this internal caller never produces.
        &c.as_slice().unwrap_or(&[])[start..start + self.order]
    }

    fn transform(&self, u: &Array1<f64>) -> Array1<f64> {
        let k = self.k_regimes;
        let mut c = u.clone();
        transform_transition(&mut c, u, k);
        // Mean: identity.
        // AR: Monahan stationarity transform per regime block.
        let n_ar_blocks = if self.switching_ar { k } else { 1 };
        for b in 0..n_ar_blocks {
            let start = self.ar_start() + b * self.order;
            let uu: Vec<f64> = (0..self.order).map(|j| u[start + j]).collect();
            let cc = constrain_stationary(&uu);
            for (j, &v) in cc.iter().enumerate() {
                c[start + j] = v;
            }
        }
        // Variance: square.
        for idx in self.var_start()..(self.var_start() + self.k_var()) {
            c[idx] = u[idx] * u[idx];
        }
        c
    }

    fn untransform(&self, c: &Array1<f64>) -> Array1<f64> {
        let k = self.k_regimes;
        let mut u = c.clone();
        untransform_transition(&mut u, c, k);
        let n_ar_blocks = if self.switching_ar { k } else { 1 };
        for b in 0..n_ar_blocks {
            let start = self.ar_start() + b * self.order;
            let cc: Vec<f64> = (0..self.order).map(|j| c[start + j]).collect();
            let uu = unconstrain_stationary(&cc);
            for (j, &v) in uu.iter().enumerate() {
                u[start + j] = v;
            }
        }
        for idx in self.var_start()..(self.var_start() + self.k_var()) {
            u[idx] = c[idx].sqrt();
        }
        u
    }

    /// Conditional log-likelihoods `log f(y_t | S_t, S_{t-1}, ..., S_{t-order})`,
    /// shaped `(k^(order+1), nobs)`, indexed by the joint state with `S_t` most
    /// significant.
    fn conditional_loglik(&self, c: &Array1<f64>) -> Array2<f64> {
        let k = self.k_regimes;
        let order = self.order;
        let nobs = self.nobs();
        let ns = k.pow((order + 1) as u32);
        let mean_start = self.mean_start();
        let var_start = self.var_start();
        let two_pi = std::f64::consts::TAU;
        let mut cll = Array2::<f64>::zeros((ns, nobs));

        for state in 0..ns {
            // Decode joint-state digits: d[0] = S_t, d[m] = S_{t-m}.
            let mut digits = vec![0usize; order + 1];
            let mut idx = state;
            for pos in (0..=order).rev() {
                digits[pos] = idx % k;
                idx /= k;
            }
            let s_t = digits[0];
            let mean_t = c[mean_start + s_t];
            let phi = self.ar_coeffs(c, s_t);
            let variance = if self.switching_variance {
                c[var_start + s_t]
            } else {
                c[var_start]
            };
            for t in 0..nobs {
                // In-sample observation index in the full series.
                let full_t = t + order;
                // Predicted mean conditional on the regime path:
                //   y_t = a_{S_t} + sum_j phi_j (y_{t-j} - a_{S_{t-j}}).
                let mut pred = mean_t;
                for j in 1..=order {
                    let s_lag = digits[j];
                    let mean_lag = c[mean_start + s_lag];
                    pred += phi[j - 1] * (self.endog[full_t - j] - mean_lag);
                }
                let resid = self.endog[full_t] - pred;
                cll[[state, t]] = -0.5 * resid * resid / variance - 0.5 * (two_pi * variance).ln();
            }
        }
        cll
    }

    fn neg_loglike(&self, u: &Array1<f64>) -> Result<f64> {
        let c = self.transform(u);
        let c_s = c
            .as_slice()
            .ok_or_else(|| Error::Value("params must be contiguous".into()))?;
        let p = transition_matrix(c_s, self.k_regimes);
        let init = steady_state(&p)?;
        let cll = self.conditional_loglik(&c);
        let out = hamilton_filter(&init, &p, &cll, self.order);
        Ok(-out.llf)
    }

    /// Starting parameters: OLS of `y_t` on `[1, y_{t-1}, ..., y_{t-p}]`
    /// interpolated across regimes, mirroring the reference.
    fn start_params(&self) -> Result<Array1<f64>> {
        let k = self.k_regimes;
        let order = self.order;
        let nobs = self.nobs();
        let mut c = Array1::<f64>::zeros(self.k_params());
        for j in 0..k {
            for i in 0..(k - 1) {
                c[j * (k - 1) + i] = 1.0 / k as f64;
            }
        }
        // Build design [1, y_{t-1}, ..., y_{t-p}].
        let mut x = Array2::<f64>::zeros((nobs, 1 + order));
        let mut yv = Array1::<f64>::zeros(nobs);
        for t in 0..nobs {
            let full_t = t + order;
            yv[t] = self.endog[full_t];
            x[[t, 0]] = 1.0;
            for j in 1..=order {
                x[[t, j]] = self.endog[full_t - j];
            }
        }
        let beta = ols(&x, &yv)?;
        let resid = &yv - &x.dot(&beta);
        let variance = var_pop(&resid);
        // const term in the reference start params uses beta[:k_exog] (the
        // intercept), interpolated; AR uses beta[k_exog:].
        let intercept = beta[0];
        let mean_start = self.mean_start();
        for i in 0..k {
            let frac = i as f64 / k as f64;
            c[mean_start + i] = intercept * frac;
        }
        if self.switching_ar {
            for i in 0..k {
                let frac = i as f64 / k as f64;
                for j in 0..order {
                    c[self.ar_start() + i * order + j] = beta[1 + j] * frac;
                }
            }
        } else {
            for j in 0..order {
                c[self.ar_start() + j] = beta[1 + j];
            }
        }
        let var_start = self.var_start();
        if self.switching_variance {
            for i in 0..k {
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

    /// Fit by maximum likelihood.
    pub fn fit(&self) -> Result<MarkovResults> {
        self.fit_from(None)
    }

    /// Fit, optionally starting from a constrained parameter guess.
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
        let out = hamilton_filter(&init, &p, &cll, self.order);
        let smoothed = kim_smoother(&p, &out, self.k_regimes, self.order);

        Ok(MarkovResults::new(
            c,
            self.param_names(),
            out.llf,
            self.nobs(),
            p,
            init,
            out.filtered_marginal,
            smoothed,
            converged,
        ))
    }
}

/// Monahan / Barndorff-Nielsen-Schou partial-autocorrelation transform mapping
/// unconstrained reals to stationary AR coefficients, matching scipy /
/// `constrain_stationary_univariate`.
///
/// Step 1: map each unconstrained `u_i` to a partial autocorrelation
/// `r_i = u_i / sqrt(1 + u_i^2)`. Step 2: Durbin–Levinson recursion turns the
/// PACFs into AR coefficients. The reference returns the AR polynomial with the
/// sign convention `y_t = phi_1 y_{t-1} + ...`, which for the single-lag case
/// yields `phi = -r = -u / sqrt(1 + u^2)`.
fn constrain_stationary(u: &[f64]) -> Vec<f64> {
    let n = u.len();
    // Partial autocorrelations in (-1, 1).
    let r: Vec<f64> = u.iter().map(|&x| x / (1.0 + x * x).sqrt()).collect();
    // Monahan's recursion: y[k, i] = y[k-1, i] + r[k] * y[k-1, k-i-1].
    let mut y = vec![vec![0.0f64; n]; n];
    for k in 0..n {
        for i in 0..k {
            y[k][i] = y[k - 1][i] + r[k] * y[k - 1][k - i - 1];
        }
        y[k][k] = r[k];
    }
    // The reference returns -y[n-1, :].
    y[n - 1].iter().map(|&v| -v).collect()
}

/// Inverse of [`constrain_stationary`].
fn unconstrain_stationary(c: &[f64]) -> Vec<f64> {
    let n = c.len();
    let mut y = vec![vec![0.0f64; n]; n];
    // y[n-1, :] = -constrained.
    for i in 0..n {
        y[n - 1][i] = -c[i];
    }
    for k in (1..n).rev() {
        for i in 0..k {
            y[k - 1][i] = (y[k][i] - y[k][k] * y[k][k - i - 1]) / (1.0 - y[k][k] * y[k][k]);
        }
    }
    // r = diagonal; x = r / sqrt(1 - r^2).
    (0..n)
        .map(|k| {
            let r = y[k][k];
            r / (1.0 - r * r).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod ar_transform_tests {
    use super::{constrain_stationary, unconstrain_stationary};

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() <= tol, "got {x}, want {y}");
        }
    }

    #[test]
    fn order1_matches_reference() {
        // constrain([u]) == -u / sqrt(1+u^2).
        for &u in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let c = constrain_stationary(&[u]);
            let want = -u / (1.0 + u * u).sqrt();
            assert!((c[0] - want).abs() < 1e-12);
            let back = unconstrain_stationary(&c);
            assert!((back[0] - u).abs() < 1e-9);
        }
    }

    #[test]
    fn order2_matches_reference() {
        // scipy: constrain([0.7, -0.3]) = [-0.40867915, 0.28734789].
        let c = constrain_stationary(&[0.7, -0.3]);
        close(&c, &[-0.40867915, 0.28734789], 1e-7);
        let back = unconstrain_stationary(&c);
        close(&back, &[0.7, -0.3], 1e-9);
    }
}
