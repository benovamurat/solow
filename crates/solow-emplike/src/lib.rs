//! # solow-emplike
//!
//! Empirical-likelihood (EL) inference on descriptive statistics of a univariate
//! sample, following Owen (2001) and validated against the reference
//! `emplike.descriptive` module.
//!
//! The entry point is [`DescStat`], which exposes
//!
//! * [`DescStat::test_mean`] — EL test of a hypothesized mean `mu0`,
//! * [`DescStat::ci_mean`] — EL confidence interval for the mean,
//! * [`DescStat::test_var`] — EL test of a hypothesized variance `sig2_0`.
//!
//! ## Method
//!
//! For a hypothesized mean `mu0`, the empirical-likelihood weights `w_i` maximize
//! `sum log(w_i)` subject to `sum w_i = 1` and `sum w_i (x_i - mu0) = 0`. The dual
//! solution is `w_i = 1 / (n (1 + eta (x_i - mu0)))`, where the Lagrange
//! multiplier `eta` is the root of `sum (x_i - mu0) / (1 + eta (x_i - mu0)) = 0`.
//! The EL test statistic is `-2 sum log(n w_i)`, asymptotically chi-squared with
//! one degree of freedom.
//!
//! The variance test profiles out a nuisance mean: for each candidate mean it
//! solves a two-parameter EL dual by a modified Newton iteration, then minimizes
//! the resulting `-2 logELR` over the nuisance mean.

mod root;

use root::{brentq, fminbound};
use solow_distributions::{chi2_ppf, chi2_sf};

/// Empirical-likelihood inference for a univariate sample.
#[derive(Clone, Debug)]
pub struct DescStat {
    endog: Vec<f64>,
    nobs: usize,
}

/// Result of an EL hypothesis test: the `-2 logELR` statistic and its p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestResult {
    /// The `-2 log` empirical-likelihood-ratio statistic.
    pub stat: f64,
    /// The asymptotic p-value, from the chi-squared survival function.
    pub pvalue: f64,
}

impl DescStat {
    /// Build a [`DescStat`] from a sample. Panics if fewer than two points.
    pub fn new(endog: &[f64]) -> Self {
        assert!(endog.len() >= 2, "need at least two observations");
        DescStat {
            endog: endog.to_vec(),
            nobs: endog.len(),
        }
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.nobs
    }

    fn min(&self) -> f64 {
        self.endog.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.endog.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// EL test of a hypothesized mean `mu0`.
    ///
    /// Returns the `-2 logELR` statistic and the chi-squared(1) p-value.
    pub fn test_mean(&self, mu0: f64) -> TestResult {
        let n = self.nobs as f64;
        let endog = &self.endog;

        // Bracket for the Lagrange multiplier, matching the reference.
        let eta_min = (1.0 - 1.0 / n) / (mu0 - self.max());
        let eta_max = (1.0 - 1.0 / n) / (mu0 - self.min());

        let find_eta = |eta: f64| -> f64 {
            endog
                .iter()
                .map(|&xi| (xi - mu0) / (1.0 + eta * (xi - mu0)))
                .sum()
        };
        let eta_star = brentq(find_eta, eta_min, eta_max);

        let llr: f64 = endog
            .iter()
            .map(|&xi| {
                let w = (1.0 / n) / (1.0 + eta_star * (xi - mu0));
                (n * w).ln()
            })
            .sum::<f64>()
            * -2.0;

        TestResult {
            stat: llr,
            pvalue: chi2_sf(llr, 1.0),
        }
    }

    /// EL confidence interval for the mean at significance level `sig`
    /// (e.g. `0.05` for a 95% interval), using the "gamma" root-finding method.
    ///
    /// Returns `(low, high)`.
    pub fn ci_mean(&self, sig: f64) -> (f64, f64) {
        self.ci_mean_opts(sig, 1e-8, -1e10, 1e10)
    }

    /// [`Self::ci_mean`] with explicit search parameters, mirroring the reference
    /// defaults (`epsilon = 1e-8`, `gamma_low = -1e10`, `gamma_high = 1e10`).
    pub fn ci_mean_opts(
        &self,
        sig: f64,
        epsilon: f64,
        gamma_low: f64,
        gamma_high: f64,
    ) -> (f64, f64) {
        let endog = &self.endog;
        let n = self.nobs as f64;
        let r0 = chi2_ppf(1.0 - sig, 1.0);

        // sum(log(n*w(gamma))) test function: -2 sum log(n w) - r0, with
        // w(gamma) = (x - gamma)^-1 / sum((x - gamma)^-1).
        let find_gamma = |gamma: f64| -> f64 {
            let denom: f64 = endog.iter().map(|&xi| 1.0 / (xi - gamma)).sum();
            let llr: f64 = endog
                .iter()
                .map(|&xi| {
                    let w = (1.0 / (xi - gamma)) / denom;
                    (n * w).ln()
                })
                .sum();
            -2.0 * llr - r0
        };

        let gamma_star_l = brentq(find_gamma, gamma_low, self.min() - epsilon);
        let gamma_star_u = brentq(find_gamma, self.max() + epsilon, gamma_high);

        let mu_at = |gamma: f64| -> f64 {
            let denom: f64 = endog.iter().map(|&xi| 1.0 / (xi - gamma)).sum();
            endog
                .iter()
                .map(|&xi| ((1.0 / (xi - gamma)) / denom) * xi)
                .sum()
        };

        (mu_at(gamma_star_l), mu_at(gamma_star_u))
    }

    /// EL test of a hypothesized variance `sig2_0`.
    ///
    /// Profiles out the nuisance mean by minimizing the `-2 logELR` over the
    /// mean, then returns that statistic and the chi-squared(1) p-value.
    pub fn test_var(&self, sig2_0: f64) -> TestResult {
        let mu_min = self.min();
        let mu_max = self.max();
        let mut opt_var = |nuisance_mu: f64| self.opt_var(nuisance_mu, sig2_0);
        let (_xf, llr) = fminbound(&mut opt_var, mu_min, mu_max);
        TestResult {
            stat: llr,
            pvalue: chi2_sf(llr, 1.0),
        }
    }

    /// `-2 logELR` for the variance test at a fixed nuisance mean.
    fn opt_var(&self, nuisance_mu: f64, sig2_0: f64) -> f64 {
        let n = self.nobs as f64;
        // Estimating equations: [x - mu, (x - mu)^2 - sig2_0].
        let est: Vec<[f64; 2]> = self
            .endog
            .iter()
            .map(|&x| {
                let m = x - nuisance_mu;
                [m, m * m - sig2_0]
            })
            .collect();

        let eta = modif_newton(&est, self.nobs);

        let llr: f64 = est
            .iter()
            .map(|e| {
                let denom = 1.0 + eta[0] * e[0] + eta[1] * e[1];
                let w = (1.0 / n) / denom;
                (n * w).ln()
            })
            .sum();
        -2.0 * llr
    }
}

/// Modified Newton solver for the two-parameter EL dual, faithfully mirroring the
/// reference `_modif_newton` / `_fit_newton` (weights `1/n`, `tol = 1e-8`,
/// `ridge_factor = 1e-10`, `maxiter = 50`).
///
/// Minimizes `-sum log_star(eta)` over the Lagrange multiplier `eta` in R^2.
fn modif_newton(est: &[[f64; 2]], nobs: usize) -> [f64; 2] {
    let n = nobs as f64;
    let weights = 1.0 / n; // every observation weight is 1/n
    let sum_w = 1.0; // sum of weights
    let inv_n = 1.0 / n;
    let start = inv_n;
    let mut params = [start, start];
    let mut oldparams = [f64::INFINITY, f64::INFINITY];
    let tol = 1e-8;
    let ridge = 1e-10;
    let maxiter = 50;

    let mut iters = 0;
    while iters < maxiter
        && ((params[0] - oldparams[0]).abs() > tol || (params[1] - oldparams[1]).abs() > tol)
    {
        // Negative gradient and negative Hessian of sum(log_star).
        // _fit_newton minimizes f = -sum(log_star), so score = -grad, hess = -hess.
        let (grad, hess) = grad_hess(est, &params, weights, sum_w, n);
        // score = -grad ; H = -hess ; regularize H diagonal with ridge.
        let h = [
            [-hess[0][0] + ridge, -hess[0][1]],
            [-hess[1][0], -hess[1][1] + ridge],
        ];
        let s = [-grad[0], -grad[1]];

        // Solve H * step = s (2x2 closed form).
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        let step = [
            (h[1][1] * s[0] - h[0][1] * s[1]) / det,
            (-h[1][0] * s[0] + h[0][0] * s[1]) / det,
        ];

        oldparams = params;
        params = [oldparams[0] - step[0], oldparams[1] - step[1]];
        iters += 1;
    }
    params
}

/// Gradient and Hessian of `sum(log_star(eta))` for the 2-parameter problem,
/// including the small-`data_star` linearization branch from the reference.
fn grad_hess(
    est: &[[f64; 2]],
    eta: &[f64; 2],
    weights: f64,
    sum_w: f64,
    nobs: f64,
) -> ([f64; 2], [[f64; 2]; 2]) {
    let mut grad = [0.0_f64; 2];
    let mut hess = [[0.0_f64; 2]; 2];
    let thr = 1.0 / nobs;

    for e in est {
        let ds = sum_w + eta[0] * e[0] + eta[1] * e[1];

        // First-derivative factor (data_star_prime).
        let dsp = if ds < thr {
            nobs * (2.0 - nobs * ds)
        } else {
            1.0 / ds
        };
        let gfac = weights * dsp;
        grad[0] += gfac * e[0];
        grad[1] += gfac * e[1];

        // Second-derivative factor (data_star_doub_prime).
        let dspp = if ds < thr {
            -nobs * nobs
        } else {
            -1.0 / (ds * ds)
        };
        let hfac = weights * dspp;
        hess[0][0] += hfac * e[0] * e[0];
        hess[0][1] += hfac * e[0] * e[1];
        hess[1][0] += hfac * e[1] * e[0];
        hess[1][1] += hfac * e[1] * e[1];
    }
    (grad, hess)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn ramp() -> Vec<f64> {
        (1..=10).map(|i| i as f64).collect()
    }

    #[test]
    fn test_mean_at_sample_mean_is_zero() {
        let d = DescStat::new(&ramp());
        // sample mean is 5.5; statistic should be (numerically) zero, p ~ 1.
        let r = d.test_mean(5.5);
        assert_abs_diff_eq!(r.stat, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r.pvalue, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_mean_is_symmetric_about_mean() {
        let d = DescStat::new(&ramp());
        // Symmetric data: mu0 = 5 and mu0 = 6 are mirror images about 5.5.
        let a = d.test_mean(5.0);
        let b = d.test_mean(6.0);
        assert_abs_diff_eq!(a.stat, b.stat, epsilon = 1e-9);
    }

    #[test]
    fn ci_mean_brackets_sample_mean() {
        let d = DescStat::new(&ramp());
        let (lo, hi) = d.ci_mean(0.05);
        assert!(lo < 5.5 && 5.5 < hi, "CI [{lo}, {hi}] must contain 5.5");
        // The statistic at each endpoint equals chi2_ppf(0.95, 1).
        let crit = chi2_ppf(0.95, 1.0);
        assert_abs_diff_eq!(d.test_mean(lo).stat, crit, epsilon = 1e-6);
        assert_abs_diff_eq!(d.test_mean(hi).stat, crit, epsilon = 1e-6);
    }

    #[test]
    fn test_var_at_sample_variance_is_zero() {
        let d = DescStat::new(&ramp());
        // population variance of 1..10 is 8.25.
        let r = d.test_var(8.25);
        assert_abs_diff_eq!(r.stat, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn pvalue_matches_chi2_sf() {
        let d = DescStat::new(&ramp());
        let r = d.test_mean(4.0);
        assert_abs_diff_eq!(r.pvalue, chi2_sf(r.stat, 1.0), epsilon = 1e-12);
    }
}
