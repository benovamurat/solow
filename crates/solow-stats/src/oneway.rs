//! One-way analysis of variance for `k` independent samples.
//!
//! Mirrors the reference `anova_oneway` / `anova_generic`, supporting the
//! classic equal-variance F-test, Welch's unequal-variance test, and the
//! Brown–Forsythe variant. [`f_oneway`] is the classic equal-variance test in
//! the familiar SciPy form.

use solow_distributions::f_sf;

/// How heteroscedasticity is treated in [`anova_oneway`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseVar {
    /// Equal variances assumed (standard one-way ANOVA).
    Equal,
    /// Unequal variances (Welch's ANOVA with Satterthwaite d.o.f.).
    Unequal,
    /// Brown–Forsythe variant with Mehrotra-corrected degrees of freedom.
    BrownForsythe,
}

/// Result of a one-way ANOVA.
#[derive(Debug, Clone, Copy)]
pub struct OnewayResult {
    /// F-distributed test statistic.
    pub statistic: f64,
    /// p-value of the test.
    pub pvalue: f64,
    /// Numerator degrees of freedom.
    pub df_num: f64,
    /// Denominator degrees of freedom.
    pub df_denom: f64,
}

/// Sample mean.
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample variance with `ddof = 1`.
fn var_unbiased(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = mean(x);
    x.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)
}

/// One-way ANOVA from summary statistics. Mirrors `anova_generic`.
pub fn anova_generic(
    means: &[f64],
    variances: &[f64],
    nobs: &[f64],
    use_var: UseVar,
    welch_correction: bool,
) -> OnewayResult {
    let n_groups = means.len();
    let ng = n_groups as f64;
    let nobs_t: f64 = nobs.iter().sum();

    // Group weights depend on the variance assumption.
    let weights: Vec<f64> = match use_var {
        UseVar::Unequal => nobs.iter().zip(variances).map(|(&n, &v)| n / v).collect(),
        UseVar::Equal | UseVar::BrownForsythe => nobs.to_vec(),
    };
    let w_total: f64 = weights.iter().sum();
    let w_rel: Vec<f64> = weights.iter().map(|&w| w / w_total).collect();
    let meanw_t: f64 = w_rel.iter().zip(means).map(|(&w, &m)| w * m).sum();

    let mut statistic: f64 = weights
        .iter()
        .zip(means)
        .map(|(&w, &m)| w * (m - meanw_t) * (m - meanw_t))
        .sum::<f64>()
        / (ng - 1.0);
    let mut df_num = ng - 1.0;
    let df_denom;

    match use_var {
        UseVar::Unequal => {
            let tmp: f64 = w_rel
                .iter()
                .zip(nobs)
                .map(|(&wr, &n)| (1.0 - wr) * (1.0 - wr) / (n - 1.0))
                .sum::<f64>()
                / (ng * ng - 1.0);
            if welch_correction {
                statistic /= 1.0 + 2.0 * (ng - 2.0) * tmp;
            }
            df_denom = 1.0 / (3.0 * tmp);
        }
        UseVar::Equal => {
            let tmp: f64 = nobs
                .iter()
                .zip(variances)
                .map(|(&n, &v)| (n - 1.0) * v)
                .sum::<f64>()
                / (nobs_t - ng);
            statistic /= tmp;
            df_denom = nobs_t - ng;
        }
        UseVar::BrownForsythe => {
            let tmp: f64 = nobs
                .iter()
                .zip(variances)
                .map(|(&n, &v)| (1.0 - n / nobs_t) * v)
                .sum();
            statistic = nobs
                .iter()
                .zip(means)
                .map(|(&n, &m)| n * (m - meanw_t) * (m - meanw_t))
                .sum::<f64>()
                / tmp;
            df_denom = tmp * tmp
                / nobs
                    .iter()
                    .zip(variances)
                    .map(|(&n, &v)| {
                        let f = 1.0 - n / nobs_t;
                        f * f * v * v / (n - 1.0)
                    })
                    .sum::<f64>();
            // Mehrotra-corrected numerator d.o.f.
            let sum_v2: f64 = variances.iter().map(|&v| v * v).sum();
            let sum_nv: f64 = nobs
                .iter()
                .zip(variances)
                .map(|(&n, &v)| n / nobs_t * v)
                .sum();
            let sum_nv2: f64 = nobs
                .iter()
                .zip(variances)
                .map(|(&n, &v)| n / nobs_t * v * v)
                .sum();
            df_num = tmp * tmp / (sum_v2 + sum_nv * sum_nv - 2.0 * sum_nv2);
        }
    }

    let pvalue = f_sf(statistic, df_num, df_denom);
    OnewayResult {
        statistic,
        pvalue,
        df_num,
        df_denom,
    }
}

/// One-way ANOVA from raw samples. Mirrors `anova_oneway` (no trimming).
pub fn anova_oneway(groups: &[Vec<f64>], use_var: UseVar, welch_correction: bool) -> OnewayResult {
    let means: Vec<f64> = groups.iter().map(|g| mean(g)).collect();
    let variances: Vec<f64> = groups.iter().map(|g| var_unbiased(g)).collect();
    let nobs: Vec<f64> = groups.iter().map(|g| g.len() as f64).collect();
    anova_generic(&means, &variances, &nobs, use_var, welch_correction)
}

/// Classic one-way ANOVA F-test (equal variances), as in SciPy's `f_oneway`.
///
/// Returns `(statistic, pvalue)`. Equivalent to
/// `anova_oneway(.., UseVar::Equal, ..)`.
pub fn f_oneway(groups: &[Vec<f64>]) -> (f64, f64) {
    let res = anova_oneway(groups, UseVar::Equal, true);
    (res.statistic, res.pvalue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_variance_matches_scipy_form() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let (f, p) = f_oneway(&groups);
        assert!(f > 0.0);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn df_num_is_k_minus_one() {
        let groups = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let res = anova_oneway(&groups, UseVar::Equal, true);
        assert_eq!(res.df_num, 2.0);
    }
}
