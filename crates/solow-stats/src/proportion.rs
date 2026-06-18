//! Tests and confidence intervals for binomial proportions.
//!
//! Provides the one- and two-sample proportion z-test
//! ([`proportions_ztest`]) and several proportion confidence-interval methods
//! ([`proportion_confint`]). Mirrors the reference `proportions_ztest` and
//! `proportion_confint`.

use crate::weightstats::Alternative;
use solow_distributions::special::betaincinv;
use solow_distributions::{norm_cdf, norm_isf, norm_sf};

/// One- or two-sample test for proportions based on the normal approximation.
///
/// `count` and `nobs` are equal-length slices of successes and trials. For a
/// single element this is the one-sample test against `value` (the null
/// proportion). For two elements it tests `prop[0] − prop[1] = value`. The
/// variance uses the pooled proportion. Returns `(zstat, pvalue)`. Mirrors the
/// reference `proportions_ztest`.
pub fn proportions_ztest(
    count: &[f64],
    nobs: &[f64],
    value: f64,
    alternative: Alternative,
) -> (f64, f64) {
    assert_eq!(count.len(), nobs.len(), "count and nobs length mismatch");
    let k = count.len();
    assert!(
        k == 1 || k == 2,
        "only one- and two-sample tests are supported"
    );

    let prop: Vec<f64> = count.iter().zip(nobs).map(|(&c, &n)| c / n).collect();
    let diff = if k == 1 {
        prop[0] - value
    } else {
        prop[0] - prop[1] - value
    };

    let count_sum: f64 = count.iter().sum();
    let nobs_sum: f64 = nobs.iter().sum();
    let p_pooled = count_sum / nobs_sum;
    let nobs_fact: f64 = nobs.iter().map(|&n| 1.0 / n).sum();
    let var = p_pooled * (1.0 - p_pooled) * nobs_fact;
    let std_diff = var.sqrt();

    let zstat = diff / std_diff;
    let pvalue = match alternative {
        Alternative::TwoSided => norm_sf(zstat.abs()) * 2.0,
        Alternative::Larger => norm_sf(zstat),
        Alternative::Smaller => norm_cdf(zstat),
    };
    (zstat, pvalue)
}

/// Confidence-interval method for a binomial proportion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfintMethod {
    /// Asymptotic normal (Wald) interval (clipped to `[0, 1]`).
    Normal,
    /// Agresti–Coull interval (clipped to `[0, 1]`).
    AgrestiCoull,
    /// Wilson score interval.
    Wilson,
    /// Clopper–Pearson exact (Beta) interval.
    Beta,
    /// Jeffreys Bayesian interval.
    Jeffreys,
}

/// Inverse-CDF of the Beta(a, b) distribution at probability `p`.
fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    betaincinv(a, b, p)
}

/// Survival-function inverse (isf) of the Beta(a, b) distribution.
fn beta_isf(p: f64, a: f64, b: f64) -> f64 {
    beta_ppf(1.0 - p, a, b)
}

/// `(1 − alpha)` confidence interval for a binomial proportion.
///
/// `count` successes in `nobs` trials. Returns `(lower, upper)`. Mirrors the
/// reference `proportion_confint` for the supported `method`s.
pub fn proportion_confint(count: f64, nobs: f64, alpha: f64, method: ConfintMethod) -> (f64, f64) {
    let q = count / nobs;
    let alpha_2 = 0.5 * alpha;
    let crit = norm_isf(alpha / 2.0);

    let (mut lo, mut hi) = match method {
        ConfintMethod::Normal => {
            let std = (q * (1.0 - q) / nobs).sqrt();
            let dist = crit * std;
            (q - dist, q + dist)
        }
        ConfintMethod::AgrestiCoull => {
            let nobs_c = nobs + crit * crit;
            let q_c = (count + crit * crit / 2.0) / nobs_c;
            let std_c = (q_c * (1.0 - q_c) / nobs_c).sqrt();
            let dist = crit * std_c;
            (q_c - dist, q_c + dist)
        }
        ConfintMethod::Wilson => {
            let crit2 = crit * crit;
            let denom = 1.0 + crit2 / nobs;
            let center = (q + crit2 / (2.0 * nobs)) / denom;
            let mut dist = crit * (q * (1.0 - q) / nobs + crit2 / (4.0 * nobs * nobs)).sqrt();
            dist /= denom;
            (center - dist, center + dist)
        }
        ConfintMethod::Beta => {
            let mut ci_low = beta_ppf(alpha_2, count, nobs - count + 1.0);
            let mut ci_upp = beta_isf(alpha_2, count + 1.0, nobs - count);
            if q == 0.0 {
                ci_low = 0.0;
            }
            if q == 1.0 {
                ci_upp = 1.0;
            }
            (ci_low, ci_upp)
        }
        ConfintMethod::Jeffreys => {
            // beta.interval(1 - alpha, count + .5, nobs - count + .5)
            let a = count + 0.5;
            let b = nobs - count + 0.5;
            (beta_ppf(alpha_2, a, b), beta_ppf(1.0 - alpha_2, a, b))
        }
    };

    if matches!(method, ConfintMethod::Normal | ConfintMethod::AgrestiCoull) {
        lo = lo.clamp(0.0, 1.0);
        hi = hi.clamp(0.0, 1.0);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_sample_ztest_sign() {
        let (z, p) = proportions_ztest(&[45.0], &[100.0], 0.5, Alternative::TwoSided);
        assert!(z < 0.0);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn normal_confint_brackets_estimate() {
        let (lo, hi) = proportion_confint(45.0, 100.0, 0.05, ConfintMethod::Normal);
        assert!(lo < 0.45 && hi > 0.45);
    }
}
