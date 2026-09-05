//! Weighted descriptive statistics and two-sample location tests.

use ndarray::Array1;
use solow_distributions::{norm_cdf, norm_sf, t_cdf, t_sf};

/// Alternative hypothesis for a location test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternative {
    /// `H1: parameter != value` (default).
    TwoSided,
    /// `H1: parameter > value`.
    Larger,
    /// `H1: parameter < value`.
    Smaller,
}

/// Result of a t- or z-test: statistic, p-value, and degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct TTestResult {
    /// Test statistic.
    pub statistic: f64,
    /// p-value of the test.
    pub pvalue: f64,
    /// Degrees of freedom (for a z-test this is `+inf` / unused).
    pub df: f64,
}

/// Weighted descriptive statistics for a 1-D sample.
///
/// Wraps a data vector and (optional) case weights and exposes the weighted
/// mean, (co)variance with a configurable degrees-of-freedom correction, and a
/// one-sample t-test of the mean. Mirrors the reference `DescrStatsW`.
#[derive(Debug, Clone)]
pub struct DescrStatsW {
    data: Array1<f64>,
    weights: Array1<f64>,
    ddof: f64,
}

impl DescrStatsW {
    /// Construct from `data` and optional `weights` (defaulting to all-ones),
    /// with degrees-of-freedom correction `ddof` (default convention `0`).
    pub fn new(data: Array1<f64>, weights: Option<Array1<f64>>, ddof: f64) -> Self {
        let weights = weights.unwrap_or_else(|| Array1::ones(data.len()));
        assert_eq!(
            data.len(),
            weights.len(),
            "data and weights length mismatch"
        );
        DescrStatsW {
            data,
            weights,
            ddof,
        }
    }

    /// Sum of the weights (the effective number of observations).
    pub fn sum_weights(&self) -> f64 {
        self.weights.sum()
    }

    /// Number of observations, equal to the sum of weights.
    pub fn nobs(&self) -> f64 {
        self.sum_weights()
    }

    /// Weighted sum of the data.
    pub fn sum(&self) -> f64 {
        self.data.dot(&self.weights)
    }

    /// Weighted mean of the data.
    pub fn mean(&self) -> f64 {
        self.sum() / self.sum_weights()
    }

    /// Weighted sum of squares of the demeaned data.
    pub fn sumsquares(&self) -> f64 {
        let m = self.mean();
        self.data
            .iter()
            .zip(self.weights.iter())
            .map(|(&x, &w)| w * (x - m) * (x - m))
            .sum()
    }

    /// Variance with denominator `sum_weights - ddof_override`.
    pub fn var_ddof(&self, ddof: f64) -> f64 {
        self.sumsquares() / (self.sum_weights() - ddof)
    }

    /// Variance with the instance's default `ddof`.
    pub fn var(&self) -> f64 {
        self.var_ddof(self.ddof)
    }

    /// Standard deviation with denominator `sum_weights - ddof_override`.
    pub fn std_ddof(&self, ddof: f64) -> f64 {
        self.var_ddof(ddof).sqrt()
    }

    /// Standard deviation with the instance's default `ddof`.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Standard error of the weighted mean.
    ///
    /// Uses the `ddof`-adjusted standard deviation rescaled to the population
    /// form and divided by `sqrt(sum_weights - 1)`, exactly as the reference.
    pub fn std_mean(&self) -> f64 {
        let mut std = self.std();
        if self.ddof != 0.0 {
            std *= ((self.sum_weights() - self.ddof) / self.sum_weights()).sqrt();
        }
        std / (self.sum_weights() - 1.0).sqrt()
    }

    /// One-sample t-test that the (weighted) mean equals `value`.
    ///
    /// Returns the statistic `(mean - value) / std_mean`, its t-distribution
    /// p-value with `sum_weights - 1` degrees of freedom, and that d.o.f.
    pub fn ttest_mean(&self, value: f64, alternative: Alternative) -> TTestResult {
        let tstat = (self.mean() - value) / self.std_mean();
        let dof = self.sum_weights() - 1.0;
        let pvalue = match alternative {
            Alternative::TwoSided => t_sf(tstat.abs(), dof) * 2.0,
            Alternative::Larger => t_sf(tstat, dof),
            Alternative::Smaller => t_cdf(tstat, dof),
        };
        TTestResult {
            statistic: tstat,
            pvalue,
            df: dof,
        }
    }
}

/// Whether a two-sample test assumes pooled (equal) or unequal variances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseVar {
    /// Equal-variance (pooled) assumption.
    Pooled,
    /// Welch / Satterthwaite unequal-variance assumption.
    Unequal,
}

/// Two-sample independent t-test of `mean(x1) - mean(x2) == value`.
///
/// With `UseVar::Pooled` the pooled-variance Student t-test is used (d.o.f.
/// `n1 + n2 - 2`); with `UseVar::Unequal` the Welch test with Satterthwaite
/// d.o.f. is used. Mirrors the reference `ttest_ind` (unweighted samples).
pub fn ttest_ind(
    x1: &Array1<f64>,
    x2: &Array1<f64>,
    alternative: Alternative,
    usevar: UseVar,
    value: f64,
) -> TTestResult {
    let d1 = DescrStatsW::new(x1.clone(), None, 0.0);
    let d2 = DescrStatsW::new(x2.clone(), None, 0.0);
    let n1 = d1.nobs();
    let n2 = d2.nobs();
    let ss1 = d1.sumsquares();
    let ss2 = d2.sumsquares();
    // `_var` in the reference is the population variance (ddof = 0).
    let var1 = ss1 / n1;
    let var2 = ss2 / n2;

    let (stdm, dof) = match usevar {
        UseVar::Pooled => {
            let var_pooled = (ss1 + ss2) / (n1 - 1.0 + n2 - 1.0);
            let stdm = (var_pooled * (1.0 / n1 + 1.0 / n2)).sqrt();
            (stdm, n1 - 1.0 + n2 - 1.0)
        }
        UseVar::Unequal => {
            let sem1 = var1 / (n1 - 1.0);
            let sem2 = var2 / (n2 - 1.0);
            let semsum = sem1 + sem2;
            let stdm = semsum.sqrt();
            let z1 = (sem1 / semsum).powi(2) / (n1 - 1.0);
            let z2 = (sem2 / semsum).powi(2) / (n2 - 1.0);
            let dof = 1.0 / (z1 + z2);
            (stdm, dof)
        }
    };

    let tstat = (d1.mean() - d2.mean() - value) / stdm;
    let pvalue = match alternative {
        Alternative::TwoSided => t_sf(tstat.abs(), dof) * 2.0,
        Alternative::Larger => t_sf(tstat, dof),
        Alternative::Smaller => t_cdf(tstat, dof),
    };
    TTestResult {
        statistic: tstat,
        pvalue,
        df: dof,
    }
}

/// One-sample z-test that `mean(x1) == value`.
///
/// Uses the population variance of `x1` with a `ddof` correction (default
/// convention `ddof = 1`) for the standard error, referenced to the standard
/// normal distribution. Returns the z-statistic and p-value (`df` is `+inf`).
/// Mirrors the one-sample branch of the reference `ztest`.
pub fn ztest(x1: &Array1<f64>, value: f64, alternative: Alternative, ddof: f64) -> TTestResult {
    let n1 = x1.len() as f64;
    let mean = x1.sum() / n1;
    // numpy var(0): population variance, divided by n.
    let var0 = x1.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n1;
    let var = var0 / (n1 - ddof);
    let std_diff = var.sqrt();
    let zstat = (mean - value) / std_diff;
    let pvalue = match alternative {
        Alternative::TwoSided => norm_sf(zstat.abs()) * 2.0,
        Alternative::Larger => norm_sf(zstat),
        Alternative::Smaller => norm_cdf(zstat),
    };
    TTestResult {
        statistic: zstat,
        pvalue,
        df: f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn unweighted_mean_matches_plain_mean() {
        let d = DescrStatsW::new(array![1.0, 2.0, 3.0, 4.0], None, 0.0);
        assert!((d.mean() - 2.5).abs() < 1e-12);
        assert!((d.sum_weights() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_mean_basic() {
        let d = DescrStatsW::new(array![1.0, 3.0], Some(array![1.0, 3.0]), 0.0);
        // (1*1 + 3*3) / 4 = 2.5
        assert!((d.mean() - 2.5).abs() < 1e-12);
    }
}
