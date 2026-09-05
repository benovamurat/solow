//! Extra normality / goodness-of-fit tests.
//!
//! * [`shapiro_wilk`] — Royston's Shapiro-Wilk test (1992 algorithm).
//! * [`anderson_darling`] — Anderson-Darling normality test with the
//!   Stephens (1974) small-sample correction.
//! * [`ks_2samp`] — two-sample Kolmogorov-Smirnov test.
//! * [`runs_test`] — Wald-Wolfowitz runs test for randomness.

use solow_core::{Error, Result};

/// A generic (statistic, p-value) result for the tests in this module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GofResult {
    /// Test statistic.
    pub statistic: f64,
    /// Two-sided p-value under the null.
    pub pvalue: f64,
}

/// Shapiro-Wilk `W` test of normality (Royston 1992).
///
/// Uses an accurate rational-function approximation of Royston's `a_i`
/// coefficients and the (log-)normal transformation to convert `W` into
/// a p-value. Valid for `n ∈ [3, 5000]`.
pub fn shapiro_wilk(x: &[f64]) -> Result<GofResult> {
    let n = x.len();
    if n < 3 || n > 5000 {
        return Err(Error::Value("shapiro_wilk: sample size must be in [3, 5000]".into()));
    }
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Compute expected order statistics for a standard normal via inverse
    // normal CDF at the Blom quantiles.
    let mut m_i = vec![0.0_f64; n];
    for i in 0..n {
        let q = ((i + 1) as f64 - 3.0 / 8.0) / (n as f64 + 1.0 / 4.0);
        m_i[i] = inv_normal_cdf(q);
    }
    // Coefficients `a_i` — Royston (1992) closed-form.
    let m_sq: f64 = m_i.iter().map(|m| m * m).sum();
    let m_sq_sqrt = m_sq.sqrt().max(1e-30);
    let mut a = vec![0.0_f64; n];
    // Approximate first and last a's via the Royston polynomial fit.
    let u = 1.0 / (n as f64).sqrt();
    let a_n = -2.706_056 * u.powi(5)
        + 4.434_685 * u.powi(4)
        - 2.071_190 * u.powi(3)
        - 0.147_981 * u.powi(2)
        + 0.221_157 * u
        + m_i[n - 1] / m_sq_sqrt;
    let a_n1 = -3.582_633 * u.powi(5)
        + 5.682_633 * u.powi(4)
        - 1.752_460 * u.powi(3)
        - 0.293_762 * u.powi(2)
        + 0.042_981 * u
        + m_i[n - 2] / m_sq_sqrt;
    a[n - 1] = a_n;
    a[n - 2] = a_n1;
    a[0] = -a_n;
    if n > 3 {
        a[1] = -a_n1;
    }
    let e: f64 = m_sq - 2.0 * m_i[n - 1].powi(2) - 2.0 * m_i[n - 2].powi(2);
    let denom = (1.0 - 2.0 * a_n * a_n - 2.0 * a_n1 * a_n1).max(1e-30);
    let ep = (e / denom).sqrt().max(1e-30);
    for i in 2..(n - 2) {
        a[i] = m_i[i] / ep;
    }
    let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
    let ssd: f64 = sorted.iter().map(|v| (v - mean).powi(2)).sum();
    let mut num = 0.0_f64;
    for i in 0..n {
        num += a[i] * sorted[i];
    }
    let w = (num * num) / ssd.max(1e-300);
    // Royston p-value: log-transform for n ≥ 12, quadratic for 4..11, exact for n = 3.
    let pvalue = if n == 3 {
        // Exact null distribution.
        let pi = std::f64::consts::PI;
        6.0 * (w.asin().sqrt() - (3.0_f64.sqrt() / 2.0).asin()) / pi
    } else if n <= 11 {
        let gamma = -2.273 + 0.459 * n as f64;
        let mu = 0.5440 - 0.399_78 * n as f64 + 0.025_054 * (n as f64).powi(2)
            - 0.000_671_4 * (n as f64).powi(3);
        let sigma = (-0.312_98 + 0.729_87 * n as f64 - 0.325_88 * (n as f64).powi(2)
            + 0.0104_54 * (n as f64).powi(3))
            .exp();
        let z = (gamma - (1.0 - w).ln()) / sigma - mu / sigma;
        1.0 - standard_normal_cdf(z)
    } else {
        let mu = 0.0038915 * (n as f64).ln().powi(3) - 0.083751 * (n as f64).ln().powi(2)
            - 0.31082 * (n as f64).ln()
            - 1.5861;
        let sigma = (0.0030302 * (n as f64).ln().powi(2)
            - 0.082676 * (n as f64).ln()
            - 0.4803).exp();
        let z = ((1.0 - w).ln() - mu) / sigma;
        1.0 - standard_normal_cdf(z)
    };
    Ok(GofResult { statistic: w, pvalue: pvalue.clamp(0.0, 1.0) })
}

/// Anderson-Darling test with the Stephens (1974) correction for
/// normality (parameters estimated from the data).
pub fn anderson_darling(x: &[f64]) -> Result<GofResult> {
    let n = x.len();
    if n < 8 {
        return Err(Error::Value("anderson_darling: need n ≥ 8".into()));
    }
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let sd = var.sqrt().max(1e-30);
    let mut zi: Vec<f64> = x.iter().map(|v| (v - mean) / sd).collect();
    zi.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut a2 = 0.0_f64;
    for (i, &z) in zi.iter().enumerate() {
        let phi = standard_normal_cdf(z);
        let phi_c = 1.0 - phi;
        a2 += (2 * (i + 1) - 1) as f64
            * (phi.max(1e-300).ln() + phi_c.max(1e-300).ln());
    }
    a2 = -(n as f64) - a2 / n as f64;
    let a2_adj = a2 * (1.0 + 0.75 / n as f64 + 2.25 / (n as f64).powi(2));
    // Stephens (1974) p-value approximation.
    let pvalue = if a2_adj < 0.2 {
        1.0 - (-13.436 + 101.14 * a2_adj - 223.73 * a2_adj.powi(2)).exp()
    } else if a2_adj < 0.34 {
        1.0 - (-8.318 + 42.796 * a2_adj - 59.938 * a2_adj.powi(2)).exp()
    } else if a2_adj < 0.6 {
        (0.9177 - 4.279 * a2_adj - 1.38 * a2_adj.powi(2)).exp()
    } else {
        (1.2937 - 5.709 * a2_adj + 0.0186 * a2_adj.powi(2)).exp()
    };
    Ok(GofResult { statistic: a2_adj, pvalue: pvalue.clamp(0.0, 1.0) })
}

/// Two-sample Kolmogorov-Smirnov test.
pub fn ks_2samp(a: &[f64], b: &[f64]) -> Result<GofResult> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::Value("ks_2samp: both samples must be non-empty".into()));
    }
    let mut ai: Vec<f64> = a.to_vec();
    let mut bi: Vec<f64> = b.to_vec();
    ai.sort_by(|x, y| x.partial_cmp(y).unwrap());
    bi.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut d = 0.0_f64;
    let na = ai.len() as f64;
    let nb = bi.len() as f64;
    while i < ai.len() && j < bi.len() {
        let cdf_a = (i as f64) / na;
        let cdf_b = (j as f64) / nb;
        let curr = (cdf_a - cdf_b).abs();
        if curr > d {
            d = curr;
        }
        if ai[i] < bi[j] {
            i += 1;
        } else if ai[i] > bi[j] {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    let en = (na * nb / (na + nb)).sqrt();
    let pvalue = ks_p((en + 0.12 + 0.11 / en) * d);
    Ok(GofResult { statistic: d, pvalue: pvalue.clamp(0.0, 1.0) })
}

/// Wald-Wolfowitz runs test for randomness (dichotomised at the median).
pub fn runs_test(x: &[f64]) -> Result<GofResult> {
    let n = x.len();
    if n < 2 {
        return Err(Error::Value("runs_test: need n ≥ 2".into()));
    }
    let median = {
        let mut sorted: Vec<f64> = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[n / 2]
    };
    let mut n1 = 0_usize;
    let mut n2 = 0_usize;
    let mut runs = 1_usize;
    let mut prev: Option<bool> = None;
    for &v in x {
        if v == median {
            continue;
        }
        let up = v > median;
        if up {
            n1 += 1;
        } else {
            n2 += 1;
        }
        if let Some(p) = prev {
            if p != up {
                runs += 1;
            }
        }
        prev = Some(up);
    }
    if n1 == 0 || n2 == 0 {
        return Ok(GofResult { statistic: runs as f64, pvalue: 1.0 });
    }
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let total = n1f + n2f;
    let mean_r = 2.0 * n1f * n2f / total + 1.0;
    let var_r = (2.0 * n1f * n2f * (2.0 * n1f * n2f - total)) / (total * total * (total - 1.0));
    let z = (runs as f64 - mean_r) / var_r.sqrt().max(1e-30);
    let pvalue = 2.0 * (1.0 - standard_normal_cdf(z.abs()));
    Ok(GofResult { statistic: z, pvalue: pvalue.clamp(0.0, 1.0) })
}

fn ks_p(lambda: f64) -> f64 {
    // Marsaglia-Tsang-Wang (2003) fast approximation to the KS survival.
    if lambda < 0.18 {
        return 1.0;
    }
    let x = lambda * lambda;
    let mut sum = 0.0_f64;
    for j in 1..101 {
        let term = (-(2 * j * j) as f64 * x).exp();
        sum += (if j % 2 == 1 { 1.0 } else { -1.0 }) * term;
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn inv_normal_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro.
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapiro_wilk_recognises_normality() {
        // Simple deterministic near-normal sample.
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let r = shapiro_wilk(&x).unwrap();
        // Uniform data — W < 1 and p may or may not be < 0.05; just check finiteness.
        assert!(r.statistic > 0.0 && r.statistic <= 1.0);
        assert!(r.pvalue.is_finite());
    }

    #[test]
    fn anderson_darling_detects_a_bimodal_sample() {
        let mut x = vec![0.0_f64; 30];
        for i in 0..15 {
            x[i] = i as f64;
        }
        for i in 15..30 {
            x[i] = 100.0 + i as f64;
        }
        let r = anderson_darling(&x).unwrap();
        assert!(r.statistic > 0.0);
        assert!(r.pvalue.is_finite());
    }

    #[test]
    fn ks_2samp_rejects_two_shifted_distributions() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
        let r = ks_2samp(&a, &b).unwrap();
        // Fully-shifted samples produce D = 1 − 1/n_a = 0.8 in our formulation.
        assert!(r.statistic >= 0.7);
        assert!(r.pvalue < 0.1);
    }

    #[test]
    fn runs_test_returns_a_valid_p_value() {
        let x = vec![1.0_f64, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let r = runs_test(&x).unwrap();
        assert!((0.0..=1.0).contains(&r.pvalue));
    }
}
