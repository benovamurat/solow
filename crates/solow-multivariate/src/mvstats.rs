//! Shared helpers for the four classical multivariate test statistics.
//!
//! Both [`crate::manova`] and [`crate::cancorr`] reduce to a set of eigenvalues
//! `λ_i` of `inv(E + H) · H` (equivalently, the squared canonical correlations)
//! from which Wilks' lambda, Pillai's trace, the Hotelling-Lawley trace and
//! Roy's greatest root are computed together with their F-approximations and
//! p-values. This module follows the reference's `multivariate_stats` exactly.

use solow_distributions::f_sf;

/// One row of a multivariate statistics table: the statistic value, the
/// numerator and denominator degrees of freedom of its F-approximation, the
/// F-value and the upper-tail p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MvStat {
    /// The statistic value.
    pub value: f64,
    /// Numerator degrees of freedom of the F-approximation.
    pub num_df: f64,
    /// Denominator degrees of freedom of the F-approximation.
    pub den_df: f64,
    /// The approximate F-value.
    pub f_value: f64,
    /// Upper-tail p-value `P(F > f_value)`.
    pub pr_f: f64,
}

/// The four classical multivariate test statistics for a single hypothesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultivariateStats {
    /// Wilks' lambda (product of `1 / (1 + λ_i)` in eigenvalue-of-`inv(E)H`
    /// form; here `prod(1 - e_i)` with `e_i` the eigenvalues of `inv(E+H)H`).
    pub wilks_lambda: MvStat,
    /// Pillai's trace.
    pub pillai_trace: MvStat,
    /// Hotelling-Lawley trace.
    pub hotelling_lawley: MvStat,
    /// Roy's greatest root.
    pub roy_greatest_root: MvStat,
}

/// Compute the four multivariate statistics from the eigenvalues of
/// `inv(E + H) · H`.
///
/// * `eigenvals` — eigenvalues of `inv(E + H) · H` (each in `[0, 1)`);
/// * `r_err_sscp` — rank of `E + H` (`p`);
/// * `r_contrast` — rank of the contrast matrix `T` (`q`);
/// * `df_resid` — residual degrees of freedom (`v`).
///
/// Mirrors the reference `multivariate.multivariate_ols.multivariate_stats`.
pub fn multivariate_stats(
    eigenvals: &[f64],
    r_err_sscp: usize,
    r_contrast: usize,
    df_resid: f64,
) -> MultivariateStats {
    let tolerance = 1e-8;
    let v = df_resid;
    let p = r_err_sscp as f64;
    let q = r_contrast as f64;
    let s = p.min(q);

    // Keep eigenvalues above tolerance (the `e_i`), and their `λ_i = e/(1-e)`.
    let eigv2: Vec<f64> = eigenvals
        .iter()
        .copied()
        .filter(|&x| x > tolerance)
        .collect();
    let eigv1: Vec<f64> = eigv2.iter().map(|&e| e / (1.0 - e)).collect();

    let m = (((p - q).abs()) - 1.0) / 2.0;
    let n = (v - p - 1.0) / 2.0;

    // ---- Wilks' lambda. ----
    let wilks_value: f64 = eigv2.iter().map(|&e| 1.0 - e).product();
    let r = v - (p - q + 1.0) / 2.0;
    let u = (p * q - 2.0) / 4.0;
    let mut df1 = p * q;
    let t = if p * p + q * q - 5.0 > 0.0 {
        ((p * p * q * q - 4.0) / (p * p + q * q - 5.0)).sqrt()
    } else {
        1.0
    };
    let df2 = r * t - 2.0 * u;
    let lmd = wilks_value.powf(1.0 / t);
    let f = (1.0 - lmd) / lmd * df2 / df1;
    let wilks = MvStat {
        value: wilks_value,
        num_df: df1,
        den_df: df2,
        f_value: f,
        pr_f: f_sf(f, df1, df2),
    };

    // ---- Pillai's trace. ----
    let pillai_value: f64 = eigv2.iter().sum();
    df1 = s * (2.0 * m + s + 1.0);
    let df2 = s * (2.0 * n + s + 1.0);
    let f = df2 / df1 * pillai_value / (s - pillai_value);
    let pillai = MvStat {
        value: pillai_value,
        num_df: df1,
        den_df: df2,
        f_value: f,
        pr_f: f_sf(f, df1, df2),
    };

    // ---- Hotelling-Lawley trace. ----
    let hl_value: f64 = eigv1.iter().sum();
    let (df1, df2, f) = if n > 0.0 {
        let b = (p + 2.0 * n) * (q + 2.0 * n) / 2.0 / (2.0 * n + 1.0) / (n - 1.0);
        let df1 = p * q;
        let df2 = 4.0 + (p * q + 2.0) / (b - 1.0);
        let c = (df2 - 2.0) / 2.0 / n;
        let f = df2 / df1 * hl_value / c;
        (df1, df2, f)
    } else {
        let df1 = s * (2.0 * m + s + 1.0);
        let df2 = s * (s * n + 1.0);
        let f = df2 / df1 / s * hl_value;
        (df1, df2, f)
    };
    let hotelling = MvStat {
        value: hl_value,
        num_df: df1,
        den_df: df2,
        f_value: f,
        pr_f: f_sf(f, df1, df2),
    };

    // ---- Roy's greatest root. ----
    let roy_value: f64 = eigv1.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rmax = p.max(q);
    let df1 = rmax;
    let df2 = v - rmax + q;
    let f = df2 / df1 * roy_value;
    let roy = MvStat {
        value: roy_value,
        num_df: df1,
        den_df: df2,
        f_value: f,
        pr_f: f_sf(f, df1, df2),
    };

    MultivariateStats {
        wilks_lambda: wilks,
        pillai_trace: pillai,
        hotelling_lawley: hotelling,
        roy_greatest_root: roy,
    }
}
