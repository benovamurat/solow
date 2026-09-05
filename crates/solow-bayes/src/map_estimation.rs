//! Maximum a posteriori (MAP) estimation for the Bayesian mixed GLM.
//!
//! Where [`BayesMixedGlm::fit_vb`](crate::BayesMixedGlm::fit_vb) approximates the
//! whole posterior by a factored Gaussian (variational Bayes), the MAP fit just
//! locates the posterior **mode**: the parameter vector that maximizes the joint
//! log-density `log p(y, fe, vc, vcp)`. This is a deterministic optimization (no
//! sampling, no random variational draws), so it is golden-verifiable.
//!
//! The optimized parameter vector is the stacked `[fep, vcp, vc]`:
//!
//! * `fep` — the `k_fep` fixed-effects coefficients,
//! * `vcp` — the `k_vcp` variance-component parameters (log standard deviations
//!   of the random effects),
//! * `vc` — the `k_vc` random-effect realizations.
//!
//! There are **no** posterior standard-deviation parameters here (unlike the VB
//! fit); the mode is a point in the same `dim()`-dimensional space as a single
//! variational-mean vector. The parameterization and ordering match the
//! reference's `genmod.bayes_mixed_glm` `logposterior` / `fit_map`.
//!
//! ## Log-posterior
//!
//! With linear predictor `eta_i = x_i · fep + z_i · vc` and mean
//! `mu_i = g⁻¹(eta_i)` (logistic for binomial, exp for Poisson), the joint
//! log-density is
//!
//! ```text
//! log p = Σ_i loglik_family(y_i, mu_i)
//!         − ½ Σ_j vc_j² / exp(vcp[ident[j]])²  − Σ_j vcp[ident[j]]   (p(vc | vcp))
//!         − ½ Σ_k vcp_k² / vcp_p²                                    (p(vcp))
//!         − ½ Σ_k fep_k² / fe_p²                                     (p(fep))
//! ```
//!
//! The family log-likelihood includes its normalizing constant, exactly as the
//! reference's `family.loglike` does:
//!
//! * binomial (0/1 data): `y log mu + (1 − y) log(1 − mu)`
//!   (computed stably as `y·eta − log(1 + exp(eta))`),
//! * Poisson: `y log mu − mu − log Γ(y + 1)`
//!   (computed as `y·eta − exp(eta) − ln_gamma(y + 1)`).

use crate::{BayesMixedGlm, Family};
use ndarray::Array1;
use solow_core::error::{Error, Result};
use solow_optimize::minimize_bfgs;

/// Result of a MAP (posterior-mode) fit, returned by
/// [`BayesMixedGlm::fit_map`].
#[derive(Clone, Debug)]
pub struct MapResult {
    /// The full parameter vector at the mode, stacked `[fep, vcp, vc]` (length
    /// `dim()`). This is the MAP estimate.
    pub params: Array1<f64>,
    /// Fixed-effects coefficients at the mode (the `fep` block of `params`).
    pub fe: Array1<f64>,
    /// Variance-component parameters at the mode (the `vcp` block; log standard
    /// deviations of the random effects).
    pub vcp: Array1<f64>,
    /// Random-effect realizations at the mode (the `vc` block of `params`).
    pub vc: Array1<f64>,
    /// The log-posterior value `log p(y, fe, vc, vcp)` at the mode.
    pub logposterior: f64,
    /// Whether the optimizer met its convergence test.
    pub converged: bool,
    /// Optimizer iterations performed.
    pub iters: usize,
    /// Final gradient norm reported by the optimizer.
    pub grad_norm: f64,
}

impl BayesMixedGlm {
    /// The joint log-density `log p(y, fe, vc, vcp)` at the stacked parameter
    /// vector `params = [fep, vcp, vc]`. Mirrors the reference `logposterior`.
    pub fn log_posterior(&self, params: &[f64]) -> f64 {
        let (fep, vcp, vc) = self.unpack(params);
        let n = self.endog().len();

        // Linear predictor eta_i = x_i·fep + z_i·vc.
        let mut ll = 0.0;
        for i in 0..n {
            let mut eta = 0.0;
            for (k, &b) in fep.iter().enumerate() {
                eta += self.exog()[[i, k]] * b;
            }
            for (k, &b) in vc.iter().enumerate() {
                eta += self.exog_vc()[[i, k]] * b;
            }
            ll += self.family_loglik_term(self.endog()[i], eta);
        }

        // p(vc | vcp): -½ Σ vc²/s² - Σ vcp[ident] with s = exp(vcp[ident]).
        for (j, &vcj) in vc.iter().enumerate() {
            let v = vcp[self.ident_at(j)];
            let s = v.exp();
            ll -= 0.5 * vcj * vcj / (s * s);
            ll -= v;
        }
        // p(vcp): -½ Σ vcp²/vcp_p².
        for &v in vcp {
            ll -= 0.5 * v * v / (self.vcp_p() * self.vcp_p());
        }
        // p(fep): -½ Σ fep²/fe_p².
        for &b in fep {
            ll -= 0.5 * b * b / (self.fe_p() * self.fe_p());
        }
        ll
    }

    /// Gradient of [`BayesMixedGlm::log_posterior`] with respect to the stacked
    /// `[fep, vcp, vc]` vector. Mirrors the reference `logposterior_grad`.
    pub fn log_posterior_grad(&self, params: &[f64]) -> Array1<f64> {
        let (fep, vcp, vc) = self.unpack(params);
        let n = self.endog().len();
        let dim = self.dim();

        // score_factor_i = endog_i - mu_i (canonical link for both families).
        let mut score = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut eta = 0.0;
            for (k, &b) in fep.iter().enumerate() {
                eta += self.exog()[[i, k]] * b;
            }
            for (k, &b) in vc.iter().enumerate() {
                eta += self.exog_vc()[[i, k]] * b;
            }
            let mu = match self.family() {
                Family::Binomial => sigmoid(eta),
                Family::Poisson => eta.exp(),
            };
            score[i] = self.endog()[i] - mu;
        }

        let mut g = Array1::<f64>::zeros(dim);
        let (a, b) = (self.k_fep(), self.k_fep() + self.k_vcp());

        // d/dfep: exog^T score - fep / fe_p².
        for (k, &fk) in fep.iter().enumerate() {
            let mut acc = 0.0;
            for i in 0..n {
                acc += score[i] * self.exog()[[i, k]];
            }
            acc -= fk / (self.fe_p() * self.fe_p());
            g[k] = acc;
        }

        // d/dvc: exog_vc^T score - vc / s² (with s = exp(vcp[ident])).
        for (k, &vck) in vc.iter().enumerate() {
            let mut acc = 0.0;
            for i in 0..n {
                acc += score[i] * self.exog_vc()[[i, k]];
            }
            let v = vcp[self.ident_at(k)];
            let s = v.exp();
            acc -= vck / (s * s);
            g[b + k] = acc;
        }

        // d/dvcp: Σ_{j: ident[j]=l} (vc_j²/s² - 1) - vcp_l / vcp_p².
        for (k, &v) in vcp.iter().enumerate() {
            g[a + k] = -v / (self.vcp_p() * self.vcp_p());
        }
        for (j, &vcj) in vc.iter().enumerate() {
            let l = self.ident_at(j);
            let s = vcp[l].exp();
            g[a + l] += vcj * vcj / (s * s) - 1.0;
        }

        g
    }

    /// Per-observation family log-likelihood `loglik(y, mu)` expressed through the
    /// linear predictor `eta` (so `mu = g⁻¹(eta)`). Includes the normalizing
    /// constant, matching the reference `family.loglike`.
    fn family_loglik_term(&self, y: f64, eta: f64) -> f64 {
        match self.family() {
            // y log mu + (1-y) log(1-mu) = y·eta - log(1 + exp(eta)).
            Family::Binomial => y * eta - log1pexp(eta),
            // y log mu - mu - log Γ(y+1) = y·eta - exp(eta) - lgamma(y+1).
            Family::Poisson => y * eta - eta.exp() - ln_gamma(y + 1.0),
        }
    }

    /// Find the posterior **mode** (MAP estimate) by maximizing
    /// [`BayesMixedGlm::log_posterior`].
    ///
    /// Internally this minimizes `-log_posterior` with the analytic gradient via
    /// BFGS, exactly as the reference `fit_map` does. The optimization is fully
    /// deterministic.
    ///
    /// * `start` — starting parameter vector, length `dim()` (`[fep, vcp, vc]`).
    ///   If `None`, the reference's deterministic-friendly start is used:
    ///   `fep = 0`, `vcp = 1`, `vc = 0` (the reference draws `vc` from a normal;
    ///   here we use zeros so the fit is reproducible).
    pub fn fit_map(
        &self,
        start: Option<Array1<f64>>,
        maxiter: usize,
        gtol: f64,
    ) -> Result<MapResult> {
        let dim = self.dim();
        let start = match start {
            Some(s) => {
                if s.len() != dim {
                    return Err(Error::Shape("start has wrong length".into()));
                }
                s
            }
            None => {
                let mut s = Array1::<f64>::zeros(dim);
                let a = self.k_fep();
                for k in 0..self.k_vcp() {
                    s[a + k] = 1.0;
                }
                s
            }
        };

        // SAFETY: the optimizer evaluates these closures on owned, standard-layout
        // arrays, so `as_slice()` is always `Some`; the empty-slice fallback is unreachable.
        let f = |x: &Array1<f64>| -> f64 { -self.log_posterior(x.as_slice().unwrap_or(&[])) };
        let grad = |x: &Array1<f64>| -> Array1<f64> {
            -self.log_posterior_grad(x.as_slice().unwrap_or(&[]))
        };

        let res = minimize_bfgs(&start, f, grad, maxiter, gtol)?;
        let x = res.x;
        let x_s = x
            .as_slice()
            .ok_or_else(|| Error::Value("params must be contiguous".into()))?;
        let (fep, vcp, vc) = self.unpack(x_s);
        let fe = Array1::from_vec(fep.to_vec());
        let vcp = Array1::from_vec(vcp.to_vec());
        let vc = Array1::from_vec(vc.to_vec());
        let logposterior = self.log_posterior(x_s);

        Ok(MapResult {
            params: x,
            fe,
            vcp,
            vc,
            logposterior,
            converged: res.converged,
            iters: res.iters,
            grad_norm: res.grad_norm,
        })
    }
}

/// Numerically stable `log(1 + exp(x))`.
fn log1pexp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically stable logistic sigmoid `1 / (1 + exp(-x))`.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `log Γ(x)` via the Lanczos approximation (x > 0). Used for the Poisson
/// log-likelihood normalizing constant `log Γ(y + 1)`.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos g=7, n=9 coefficients.
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Family;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};
    use solow_optimize::approx_fprime;

    /// A small Poisson grouped design for gradient / convergence checks.
    fn poisson_model() -> BayesMixedGlm {
        let n_groups = 4usize;
        let per = 3usize;
        let n = n_groups * per;
        let mut exog = Array2::<f64>::zeros((n, 2));
        let mut exog_vc = Array2::<f64>::zeros((n, n_groups));
        for i in 0..n {
            exog[[i, 0]] = 1.0;
            exog[[i, 1]] = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
            exog_vc[[i, i / per]] = 1.0;
        }
        let endog = array![1., 2., 1., 0., 1., 3., 2., 1., 4., 2., 1., 0.];
        let ident = vec![0usize; n_groups];
        BayesMixedGlm::new(Family::Poisson, endog, exog, exog_vc, ident, 0.5, 2.0).unwrap()
    }

    fn binom_model() -> BayesMixedGlm {
        let n_groups = 6usize;
        let per = 4usize;
        let n = n_groups * per;
        let x1: Vec<f64> = (0..n)
            .map(|i| -1.5 + 3.0 * i as f64 / (n as f64 - 1.0))
            .collect();
        let mut exog = Array2::<f64>::zeros((n, 2));
        let mut exog_vc = Array2::<f64>::zeros((n, n_groups));
        for i in 0..n {
            exog[[i, 0]] = 1.0;
            exog[[i, 1]] = x1[i];
            exog_vc[[i, i / per]] = 1.0;
        }
        let endog = array![
            0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0.,
            1., 1.
        ];
        let ident = vec![0usize; n_groups];
        BayesMixedGlm::new(Family::Binomial, endog, exog, exog_vc, ident, 0.5, 2.0).unwrap()
    }

    /// ln_gamma matches known values.
    #[test]
    fn ln_gamma_known_values() {
        assert_abs_diff_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-12);
        // Γ(5) = 24, ln = ln(24).
        assert_abs_diff_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1e-10);
        // Γ(6) = 120.
        assert_abs_diff_eq!(ln_gamma(6.0), 120.0_f64.ln(), epsilon = 1e-10);
    }

    /// Analytic log-posterior gradient agrees with finite differences (binomial).
    #[test]
    fn binom_grad_matches_finite_difference() {
        let m = binom_model();
        let dim = m.dim();
        let mut x = Array1::<f64>::zeros(dim);
        for k in 0..dim {
            x[k] = 0.1 * (k as f64 - 3.0);
        }
        let obj = |x: &Array1<f64>| -m.log_posterior(x.as_slice().unwrap());
        let fd = approx_fprime(&x, obj);
        let g = m.log_posterior_grad(x.as_slice().unwrap());
        for k in 0..dim {
            assert_abs_diff_eq!(-g[k], fd[k], epsilon = 1e-5);
        }
    }

    /// Analytic log-posterior gradient agrees with finite differences (Poisson).
    #[test]
    fn poisson_grad_matches_finite_difference() {
        let m = poisson_model();
        let dim = m.dim();
        let mut x = Array1::<f64>::zeros(dim);
        for k in 0..dim {
            x[k] = 0.05 * (k as f64 - 2.0);
        }
        let obj = |x: &Array1<f64>| -m.log_posterior(x.as_slice().unwrap());
        let fd = approx_fprime(&x, obj);
        let g = m.log_posterior_grad(x.as_slice().unwrap());
        for k in 0..dim {
            assert_abs_diff_eq!(-g[k], fd[k], epsilon = 1e-5);
        }
    }

    /// The MAP fit reaches a stationary point of the log-posterior. (As with the
    /// reference's BFGS, the optimizer may report non-convergence even when the
    /// gradient is effectively zero, so we test the gradient directly.)
    #[test]
    fn poisson_fit_map_reaches_mode() {
        let m = poisson_model();
        let res = m.fit_map(None, 5_000, 1e-10).unwrap();
        // Gradient of the log-posterior vanishes at the mode.
        let g = m.log_posterior_grad(res.params.as_slice().unwrap());
        let gnorm = g.dot(&g).sqrt();
        assert!(gnorm < 1e-6, "gradient norm {gnorm} too large");
    }
}
