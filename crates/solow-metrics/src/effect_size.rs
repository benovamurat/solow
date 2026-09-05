//! Effect-size metrics — a piece of solow's world-class inference
//! surface that has no direct the reference equivalent.
//!
//! * `cohens_d` — the standardised mean difference (Cohen 1988).
//! * `hedges_g` — the small-sample-bias-corrected variant of Cohen's d.
//! * `glass_delta` — control-group-only standardisation.
//! * `eta_squared`, `omega_squared` — variance-explained effect sizes
//!   from one-way ANOVA outputs.
//! * `cliffs_delta` — non-parametric ordinal effect size (Cliff 1993).
//! * `cramers_v` — categorical association from a chi-square statistic.
//!
//! All functions are `#![forbid(unsafe_code)]` and free of allocations
//! beyond a single output scalar.

use solow_core::{Error, Result};

/// Cohen's d — `(μ₁ − μ₂) / σ_pooled`.
pub fn cohens_d(x1: &[f64], x2: &[f64]) -> Result<f64> {
    if x1.len() < 2 || x2.len() < 2 {
        return Err(Error::Value("cohens_d: both groups must have ≥ 2 samples".into()));
    }
    let (m1, s1) = mean_var(x1);
    let (m2, s2) = mean_var(x2);
    let n1 = x1.len() as f64;
    let n2 = x2.len() as f64;
    let pooled = (((n1 - 1.0) * s1 + (n2 - 1.0) * s2) / (n1 + n2 - 2.0)).sqrt();
    Ok((m1 - m2) / pooled.max(1e-30))
}

/// Hedges' g — Cohen's d corrected for small-sample bias.
pub fn hedges_g(x1: &[f64], x2: &[f64]) -> Result<f64> {
    let d = cohens_d(x1, x2)?;
    let n = (x1.len() + x2.len()) as f64;
    let correction = 1.0 - 3.0 / (4.0 * n - 9.0);
    Ok(d * correction)
}

/// Glass's Δ — `(μ₁ − μ₂) / σ_control` (the second group's std alone).
pub fn glass_delta(treatment: &[f64], control: &[f64]) -> Result<f64> {
    if control.len() < 2 {
        return Err(Error::Value("glass_delta: control needs ≥ 2 samples".into()));
    }
    let (mt, _) = mean_var(treatment);
    let (mc, vc) = mean_var(control);
    let sc = vc.sqrt();
    Ok((mt - mc) / sc.max(1e-30))
}

/// η² (eta squared) — `SS_between / SS_total` for a one-way ANOVA.
pub fn eta_squared(ss_between: f64, ss_total: f64) -> Result<f64> {
    if ss_total <= 0.0 {
        return Err(Error::Value("eta_squared: SS_total must be > 0".into()));
    }
    Ok(ss_between / ss_total)
}

/// ω² (omega squared) — a less biased variance-explained estimate.
pub fn omega_squared(ss_between: f64, ss_within: f64, df_between: f64, ms_within: f64) -> Result<f64> {
    let denom = ss_between + ss_within + ms_within;
    if denom <= 0.0 {
        return Err(Error::Value("omega_squared: total denominator must be > 0".into()));
    }
    Ok((ss_between - df_between * ms_within) / denom)
}

/// Cliff's δ — probability that a random draw from `a` exceeds one from
/// `b` minus the reverse. Ranges in `[−1, +1]`.
pub fn cliffs_delta(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::Value("cliffs_delta: both groups must be non-empty".into()));
    }
    let mut gt = 0_isize;
    let mut lt = 0_isize;
    for &ai in a {
        for &bj in b {
            if ai > bj {
                gt += 1;
            } else if ai < bj {
                lt += 1;
            }
        }
    }
    let n = (a.len() * b.len()) as f64;
    Ok((gt - lt) as f64 / n)
}

/// Cramér's V — categorical association strength derived from a
/// chi-square statistic on a contingency table.
pub fn cramers_v(chi2: f64, n: usize, min_dim: usize) -> Result<f64> {
    if n == 0 || min_dim < 2 {
        return Err(Error::Value("cramers_v: n must be > 0 and min_dim ≥ 2".into()));
    }
    Ok((chi2 / (n as f64 * (min_dim - 1) as f64)).sqrt())
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohens_d_recovers_a_large_effect() {
        let a = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
        let b = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let d = cohens_d(&a, &b).unwrap();
        assert!(d > 3.0);
    }

    #[test]
    fn hedges_g_is_slightly_smaller_than_cohens_d_for_small_n() {
        let a = vec![10.0_f64, 11.0, 12.0];
        let b = vec![0.0_f64, 1.0, 2.0];
        let d = cohens_d(&a, &b).unwrap();
        let g = hedges_g(&a, &b).unwrap();
        assert!(g.abs() < d.abs());
    }

    #[test]
    fn cliffs_delta_ranges_within_negative_one_and_one() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let delta = cliffs_delta(&a, &b).unwrap();
        assert!((-1.0..=1.0).contains(&delta));
        assert!(delta < 0.0, "a is strictly less than b");
    }

    #[test]
    fn cramers_v_stays_in_zero_one_range() {
        let v = cramers_v(50.0, 200, 3).unwrap();
        assert!((0.0..=1.0).contains(&v));
    }
}
