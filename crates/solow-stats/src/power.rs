//! Statistical power for one-sample t-tests and two-sample normal tests.
//!
//! Mirrors the reference `TTestPower` and `NormalIndPower`. Both expose
//! [`power`](TTestPower::power) and a [`solve_power`](TTestPower::solve_power)
//! that inverts the power equation for the sample size at a target power.

use crate::noncentral::{nct_cdf, nct_sf};
use crate::weightstats::Alternative;
use solow_distributions::{norm_cdf, norm_isf, norm_ppf, norm_sf, t_isf, t_ppf};

/// Convert the alternative to the per-tail significance used by power formulas.
fn alpha_tail(alpha: f64, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::TwoSided => alpha / 2.0,
        Alternative::Larger | Alternative::Smaller => alpha,
    }
}

/// Power of a one-sample (or paired) t-test.
///
/// `effect_size` is the standardized mean (Cohen's d), `nobs` the sample size,
/// and `df` defaults to `nobs − 1`. The power integrates the noncentral
/// t-distribution with noncentrality `d·√nobs`. Mirrors the reference
/// `ttest_power`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TTestPower;

impl TTestPower {
    /// Power at the given configuration. `df = None` uses `nobs − 1`.
    pub fn power(
        &self,
        effect_size: f64,
        nobs: f64,
        alpha: f64,
        df: Option<f64>,
        alternative: Alternative,
    ) -> f64 {
        let d = effect_size;
        let df = df.unwrap_or(nobs - 1.0);
        let alpha_ = alpha_tail(alpha, alternative);
        let nc = d * nobs.sqrt();
        let mut pow_ = 0.0;
        if matches!(alternative, Alternative::TwoSided | Alternative::Larger) {
            let crit_upp = t_isf(alpha_, df);
            pow_ += nct_sf(crit_upp, df, nc);
        }
        if matches!(alternative, Alternative::TwoSided | Alternative::Smaller) {
            let crit_low = t_ppf(alpha_, df);
            pow_ += nct_cdf(crit_low, df, nc);
        }
        pow_
    }

    /// Solve for the sample size `nobs` that achieves `power` at the given
    /// `effect_size` and `alpha`. Inverts the monotone power-in-`nobs` curve by
    /// bracketed bisection (with `df = nobs − 1`).
    pub fn solve_power(
        &self,
        effect_size: f64,
        alpha: f64,
        power: f64,
        alternative: Alternative,
    ) -> f64 {
        let f = |n: f64| self.power(effect_size, n, alpha, None, alternative) - power;
        solve_nobs(f, 2.000_001, 1.0e7)
    }
}

/// Power of a two-sample z-test for independent samples (normal approximation).
///
/// `effect_size` is the standardized mean difference, `nobs1` the size of the
/// first sample, and `ratio` the size of sample two relative to sample one
/// (`nobs2 = ratio·nobs1`; `ratio = 0` gives the one-sample test). Mirrors the
/// reference `NormalIndPower` with `ddof = 0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalIndPower;

impl NormalIndPower {
    /// Power at the given configuration.
    pub fn power(
        &self,
        effect_size: f64,
        nobs1: f64,
        alpha: f64,
        ratio: f64,
        alternative: Alternative,
    ) -> f64 {
        let ddof = 0.0;
        let nobs = if ratio > 0.0 {
            let nobs2 = nobs1 * ratio;
            1.0 / (1.0 / (nobs1 - ddof) + 1.0 / (nobs2 - ddof))
        } else {
            nobs1 - ddof
        };
        normal_power(effect_size, nobs, alpha, alternative)
    }

    /// Solve for `nobs1` achieving `power` at the given configuration.
    pub fn solve_power(
        &self,
        effect_size: f64,
        alpha: f64,
        power: f64,
        ratio: f64,
        alternative: Alternative,
    ) -> f64 {
        let f = |n: f64| self.power(effect_size, n, alpha, ratio, alternative) - power;
        solve_nobs(f, 1.000_001, 1.0e7)
    }
}

/// Power of a normally distributed test statistic. Mirrors `normal_power`.
fn normal_power(effect_size: f64, nobs: f64, alpha: f64, alternative: Alternative) -> f64 {
    let d = effect_size;
    let alpha_ = alpha_tail(alpha, alternative);
    let mut pow_ = 0.0;
    if matches!(alternative, Alternative::TwoSided | Alternative::Larger) {
        let crit = norm_isf(alpha_);
        pow_ += norm_sf(crit - d * nobs.sqrt());
    }
    if matches!(alternative, Alternative::TwoSided | Alternative::Smaller) {
        let crit = norm_ppf(alpha_);
        pow_ += norm_cdf(crit - d * nobs.sqrt());
    }
    pow_
}

/// Find the sample size where `f` (power minus target) crosses zero.
///
/// Power increases monotonically in `nobs`, so a simple bracket-and-bisect is
/// robust and converges to machine precision.
fn solve_nobs<F: Fn(f64) -> f64>(f: F, lo0: f64, hi0: f64) -> f64 {
    let mut lo = lo0;
    let mut hi = hi0;
    let flo = f(lo);
    let fhi = f(hi);
    // Expect a sign change across the bracket.
    if flo * fhi > 0.0 {
        // No crossing in range; return the endpoint nearest zero.
        return if flo.abs() < fhi.abs() { lo } else { hi };
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let fm = f(mid);
        if fm == 0.0 {
            return mid;
        }
        if (flo < 0.0) == (fm < 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-12 * (1.0 + hi) {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttest_power_in_unit_interval() {
        let tt = TTestPower;
        let p = tt.power(0.5, 30.0, 0.05, None, Alternative::TwoSided);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn solve_power_roundtrips() {
        let tt = TTestPower;
        let n = tt.solve_power(0.5, 0.05, 0.8, Alternative::TwoSided);
        let p = tt.power(0.5, n, 0.05, None, Alternative::TwoSided);
        assert!((p - 0.8).abs() < 1e-6, "{p}");
    }

    #[test]
    fn normal_power_solve_roundtrips() {
        let nip = NormalIndPower;
        let n = nip.solve_power(0.5, 0.05, 0.8, 1.0, Alternative::TwoSided);
        let p = nip.power(0.5, n, 0.05, 1.0, Alternative::TwoSided);
        assert!((p - 0.8).abs() < 1e-9, "{p}");
    }
}
