//! Two-sample Poisson rate comparison.
//!
//! [`test_poisson_2indep`] tests the equality (ratio or difference) of two
//! independent Poisson intensity rates, mirroring the reference
//! `rates.test_poisson_2indep`. The closed-form score / Wald / log / sqrt
//! statistics referenced to the normal distribution are implemented for both
//! the ratio and difference comparisons, along with the exact-conditional and
//! conditional mid-p tests based on the binomial distribution. The simulation
//! /grid `etest` variants are out of scope.

use crate::weightstats::Alternative;
use solow_distributions::special::{betainc, lgamma};
use solow_distributions::{norm_cdf, norm_sf};

/// Comparison target for [`test_poisson_2indep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    /// Test the ratio `rate1 / rate2` against `value` (default `value = 1`).
    Ratio,
    /// Test the difference `rate1 - rate2` against `value` (default `value = 0`).
    Diff,
}

/// Test statistic / p-value method for [`test_poisson_2indep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonMethod {
    /// Wald test, variance from the observed rates (`ratio` and `diff`).
    Wald,
    /// Wald test with a continuity-corrected variance (`diff` only).
    WaldCcv,
    /// Score test, variance from the null-constrained estimate (`ratio`, `diff`).
    Score,
    /// Wald test on the log-ratio (`ratio` only).
    WaldLog,
    /// Score test on the log-ratio (`ratio` only).
    ScoreLog,
    /// Variance-stabilising square-root transformation test (`ratio` only).
    Sqrt,
    /// Exact conditional test based on the binomial distribution (`ratio` only).
    ExactCond,
    /// Mid-p value of the exact conditional test (`ratio` only).
    CondMidp,
}

/// Result of [`test_poisson_2indep`].
#[derive(Debug, Clone, Copy)]
pub struct PoissonResult {
    /// Test statistic. `NaN` for the binomial (`ExactCond` / `CondMidp`) tests.
    pub statistic: f64,
    /// p-value of the test.
    pub pvalue: f64,
    /// Estimated rate of sample 1, `count1 / exposure1`.
    pub rate1: f64,
    /// Estimated rate of sample 2, `count2 / exposure2`.
    pub rate2: f64,
    /// Observed rate ratio `rate1 / rate2`.
    pub ratio: f64,
    /// Observed rate difference `rate1 - rate2`.
    pub diff: f64,
}

/// log of the binomial coefficient `C(n, k)`.
fn ln_binom(n: f64, k: f64) -> f64 {
    lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)
}

/// Binomial probability mass `P(X = k)` for `X ~ Binom(n, p)`.
fn binom_pmf(k: f64, n: f64, p: f64) -> f64 {
    if k < 0.0 || k > n {
        return 0.0;
    }
    if p <= 0.0 {
        return if k == 0.0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    (ln_binom(n, k) + k * p.ln() + (n - k) * (1.0 - p).ln()).exp()
}

/// Binomial CDF `P(X <= k)` for `X ~ Binom(n, p)` via the regularised
/// incomplete beta function: `P(X <= k) = I_{1-p}(n - k, k + 1)`.
fn binom_cdf(k: f64, n: f64, p: f64) -> f64 {
    let k = k.floor();
    if k < 0.0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }
    betainc(n - k, k + 1.0, 1.0 - p)
}

/// Binomial survival function `P(X > k) = 1 - P(X <= k)`.
fn binom_sf(k: f64, n: f64, p: f64) -> f64 {
    1.0 - binom_cdf(k, n, p)
}

/// Implicit binary search used by the two-sided binomial test: returns the
/// index `i` in `[lo, hi]` such that `a(i) <= d < a(i+1)`, where `a` is assumed
/// monotone increasing over the range. Mirrors the reference helper.
fn binary_search_binom(a: &dyn Fn(f64) -> f64, d: f64, mut lo: f64, mut hi: f64) -> f64 {
    while lo < hi {
        let mid = lo + ((hi - lo) / 2.0).floor();
        let midval = a(mid);
        if midval < d {
            lo = mid + 1.0;
        } else if midval > d {
            hi = mid - 1.0;
        } else {
            return mid;
        }
    }
    if a(lo) <= d {
        lo
    } else {
        lo - 1.0
    }
}

/// Two-sided exact binomial p-value using the "minlike" method, reproducing
/// `scipy.stats.binomtest(k, n, p).pvalue`.
fn binom_test_two_sided(k: f64, n: f64, p: f64) -> f64 {
    let d = binom_pmf(k, n, p);
    let rerr = 1.0 + 1e-7;
    let pval = if k == p * n {
        1.0
    } else if k < p * n {
        // Search the upper tail (mode .. n) for terms <= d*rerr.
        let neg_pmf = |x: f64| -binom_pmf(x, n, p);
        let ix = binary_search_binom(&neg_pmf, -d * rerr, (p * n).ceil(), n);
        let y = n - ix
            + if d * rerr == binom_pmf(ix, n, p) {
                1.0
            } else {
                0.0
            };
        binom_cdf(k, n, p) + binom_sf(n - y, n, p)
    } else {
        // Search the lower tail (0 .. mode) for terms <= d*rerr.
        let pmf = |x: f64| binom_pmf(x, n, p);
        let ix = binary_search_binom(&pmf, d * rerr, 0.0, (p * n).floor());
        let y = ix + 1.0;
        binom_cdf(y - 1.0, n, p) + binom_sf(k - 1.0, n, p)
    };
    pval.min(1.0)
}

/// Binomial p-value used by the conditional Poisson tests for a given
/// alternative; `count` successes in `total` trials under success probability
/// `prop`.
fn binom_test(count: f64, total: f64, prop: f64, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::TwoSided => binom_test_two_sided(count, total, prop),
        Alternative::Larger => binom_sf(count - 1.0, total, prop),
        Alternative::Smaller => binom_cdf(count, total, prop),
    }
}

/// p-value of a normal (z) test statistic for the given alternative.
fn z_pvalue(stat: f64, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::TwoSided => norm_sf(stat.abs()) * 2.0,
        Alternative::Larger => norm_sf(stat),
        Alternative::Smaller => norm_cdf(stat),
    }
}

/// Test the equality of two independent Poisson rates.
///
/// `count1`/`exposure1` and `count2`/`exposure2` are the event counts and total
/// exposures of the two samples; `value` is the null ratio (default `1.0` for
/// [`Compare::Ratio`]) or difference (default `0.0` for [`Compare::Diff`]). The
/// `method` selects the test statistic (see [`PoissonMethod`]). Mirrors the
/// reference `test_poisson_2indep`. Panics if a method is not valid for the
/// chosen comparison.
#[allow(clippy::too_many_arguments)]
pub fn test_poisson_2indep(
    count1: f64,
    exposure1: f64,
    count2: f64,
    exposure2: f64,
    value: Option<f64>,
    method: PoissonMethod,
    compare: Compare,
    alternative: Alternative,
) -> PoissonResult {
    let (y1, n1, y2, n2) = (count1, exposure1, count2, exposure2);
    let d = n2 / n1;
    let rate1 = y1 / n1;
    let rate2 = y2 / n2;

    let (stat, pvalue) = match compare {
        Compare::Ratio => {
            let r = value.unwrap_or(1.0);
            let r_d = r / d; // r1 * n1 / (r2 * n2)
            match method {
                PoissonMethod::Score => {
                    let stat = (y1 - y2 * r_d) / ((y1 + y2) * r_d).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                PoissonMethod::Wald => {
                    let stat = (y1 - y2 * r_d) / (y1 + y2 * r_d * r_d).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                PoissonMethod::ScoreLog => {
                    let stat =
                        ((y1 / y2).ln() - r_d.ln()) / ((2.0 + 1.0 / r_d + r_d) / (y1 + y2)).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                PoissonMethod::WaldLog => {
                    let stat = ((y1 / y2).ln() - r_d.ln()) / (1.0 / y1 + 1.0 / y2).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                PoissonMethod::Sqrt => {
                    let stat = 2.0 * ((y1 + 3.0 / 8.0).sqrt() - ((y2 + 3.0 / 8.0) * r_d).sqrt())
                        / (1.0 + r_d).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                PoissonMethod::ExactCond => {
                    let bp = r_d / (1.0 + r_d);
                    let y_total = y1 + y2;
                    (f64::NAN, binom_test(y1, y_total, bp, alternative))
                }
                PoissonMethod::CondMidp => {
                    let bp = r_d / (1.0 + r_d);
                    let y_total = y1 + y2;
                    let p =
                        binom_test(y1, y_total, bp, alternative) - 0.5 * binom_pmf(y1, y_total, bp);
                    (f64::NAN, p)
                }
                PoissonMethod::WaldCcv => {
                    panic!("waldccv is only defined for compare = diff");
                }
            }
        }
        Compare::Diff => {
            let v = value.unwrap_or(0.0);
            match method {
                PoissonMethod::Wald => {
                    let stat = (rate1 - rate2 - v) / (rate1 / n1 + rate2 / n2).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                PoissonMethod::WaldCcv => {
                    let stat = (rate1 - rate2 - v)
                        / ((y1 + 0.5) / (n1 * n1) + (y2 + 0.5) / (n2 * n2)).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                PoissonMethod::Score => {
                    let count_pooled = y1 + y2;
                    let rate_pooled = count_pooled / (n1 + n2);
                    let dt = rate_pooled - v;
                    let r2_cmle = 0.5 * (dt + (dt * dt + 4.0 * v * y2 / (n1 + n2)).sqrt());
                    let r1_cmle = r2_cmle + v;
                    let stat = (rate1 - rate2 - v) / (r1_cmle / n1 + r2_cmle / n2).sqrt();
                    (stat, z_pvalue(stat, alternative))
                }
                _ => panic!("method is not valid for compare = diff"),
            }
        }
    };

    PoissonResult {
        statistic: stat,
        pvalue,
        rate1,
        rate2,
        ratio: rate1 / rate2,
        diff: rate1 - rate2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_ratio_runs() {
        let r = test_poisson_2indep(
            60.0,
            51477.5,
            30.0,
            54308.7,
            None,
            PoissonMethod::Score,
            Compare::Ratio,
            Alternative::TwoSided,
        );
        assert!(r.statistic > 0.0);
        assert!((0.0..=1.0).contains(&r.pvalue));
    }

    #[test]
    fn binom_cdf_matches_pmf_sum() {
        // CDF(3) should equal sum of PMF(0..=3).
        let (n, p) = (10.0, 0.3);
        let direct: f64 = (0..=3).map(|k| binom_pmf(k as f64, n, p)).sum();
        assert!((binom_cdf(3.0, n, p) - direct).abs() < 1e-12);
    }
}
