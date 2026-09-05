//! Homogeneity-of-variance tests.
//!
//! * [`levene`] — Levene's (1960) robust variance test, with the choice
//!   of centre (mean or median for the Brown-Forsythe variant).
//! * [`bartlett`] — Bartlett's classic normal-theory variance test.
//! * [`fligner`] — Fligner-Killeen rank-based variance test.

use solow_core::{Error, Result};

/// Test-statistic + p-value pair returned by variance tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarianceTestResult {
    /// Statistic (F for Levene, χ² for Bartlett/Fligner).
    pub statistic: f64,
    /// Two-sided p-value under the null of equal variances.
    pub pvalue: f64,
    /// Degrees-of-freedom pair `(dfn, dfd)` where applicable.
    pub df: (f64, f64),
}

/// Centering strategy for Levene's test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeveneCenter {
    /// Deviations from the group mean (classical Levene).
    Mean,
    /// Deviations from the group median (Brown-Forsythe).
    Median,
}

/// Levene's test / Brown-Forsythe variant.
pub fn levene(groups: &[Vec<f64>], center: LeveneCenter) -> Result<VarianceTestResult> {
    if groups.len() < 2 {
        return Err(Error::Value("levene: need ≥ 2 groups".into()));
    }
    let k = groups.len();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total < k + 1 {
        return Err(Error::Value("levene: too few samples".into()));
    }
    let mut group_z: Vec<Vec<f64>> = Vec::with_capacity(k);
    for g in groups {
        let c = match center {
            LeveneCenter::Mean => g.iter().sum::<f64>() / g.len() as f64,
            LeveneCenter::Median => median(g),
        };
        group_z.push(g.iter().map(|v| (v - c).abs()).collect());
    }
    // ANOVA on the |z| values.
    let grand: f64 = group_z.iter().flatten().sum::<f64>() / n_total as f64;
    let mut ss_between = 0.0_f64;
    let mut ss_within = 0.0_f64;
    for g in &group_z {
        let mg: f64 = g.iter().sum::<f64>() / g.len() as f64;
        ss_between += g.len() as f64 * (mg - grand).powi(2);
        for &v in g {
            ss_within += (v - mg).powi(2);
        }
    }
    let dfn = (k - 1) as f64;
    let dfd = (n_total - k) as f64;
    let f = (ss_between / dfn) / (ss_within / dfd).max(1e-300);
    let pvalue = f_survival(f, dfn, dfd);
    Ok(VarianceTestResult { statistic: f, pvalue, df: (dfn, dfd) })
}

/// Bartlett's test — classical normal-theory χ² test on log-variance
/// ratios.
pub fn bartlett(groups: &[Vec<f64>]) -> Result<VarianceTestResult> {
    if groups.len() < 2 {
        return Err(Error::Value("bartlett: need ≥ 2 groups".into()));
    }
    let k = groups.len() as f64;
    let mut ni = Vec::with_capacity(groups.len());
    let mut si2 = Vec::with_capacity(groups.len());
    for g in groups {
        if g.len() < 2 {
            return Err(Error::Value("bartlett: each group must have ≥ 2 samples".into()));
        }
        let n = g.len() as f64;
        let m = g.iter().sum::<f64>() / n;
        let v = g.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        ni.push(n);
        si2.push(v);
    }
    let n_total: f64 = ni.iter().sum();
    let sp2: f64 =
        ni.iter().zip(si2.iter()).map(|(n, v)| (n - 1.0) * v).sum::<f64>() / (n_total - k);
    let numer = (n_total - k) * sp2.ln()
        - ni.iter().zip(si2.iter()).map(|(n, v)| (n - 1.0) * v.ln()).sum::<f64>();
    let one_over_n_minus_1: f64 = ni.iter().map(|n| 1.0 / (n - 1.0)).sum();
    let one_over_total: f64 = 1.0 / (n_total - k);
    let c = 1.0 + 1.0 / (3.0 * (k - 1.0)) * (one_over_n_minus_1 - one_over_total);
    let chi2 = numer / c;
    let dfn = k - 1.0;
    let pvalue = chi2_survival(chi2, dfn);
    Ok(VarianceTestResult { statistic: chi2, pvalue, df: (dfn, 0.0) })
}

/// Fligner-Killeen (1976) rank-based variance test.
pub fn fligner(groups: &[Vec<f64>]) -> Result<VarianceTestResult> {
    if groups.len() < 2 {
        return Err(Error::Value("fligner: need ≥ 2 groups".into()));
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    // Combine and centre by group median.
    let mut deviations: Vec<(f64, usize)> = Vec::with_capacity(n_total);
    for (i, g) in groups.iter().enumerate() {
        let m = median(g);
        for &v in g {
            deviations.push(((v - m).abs(), i));
        }
    }
    deviations.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // Rank in ascending order; standard normal scores of the ranks.
    let mut a_scores = vec![0.0_f64; n_total];
    for i in 0..n_total {
        let rank = (i + 1) as f64;
        let quantile = 0.5 * (rank / (n_total as f64 + 1.0) + 1.0);
        a_scores[i] = inv_normal_cdf(quantile);
    }
    let mean_a: f64 = a_scores.iter().sum::<f64>() / n_total as f64;
    let var_a: f64 = a_scores.iter().map(|a| (a - mean_a).powi(2)).sum::<f64>() / n_total as f64;
    // Per-group mean of the transformed scores.
    let mut group_sum = vec![0.0_f64; groups.len()];
    let mut group_n = vec![0.0_f64; groups.len()];
    for (i, (_, gi)) in deviations.iter().enumerate() {
        group_sum[*gi] += a_scores[i];
        group_n[*gi] += 1.0;
    }
    let mut chi2 = 0.0_f64;
    for j in 0..groups.len() {
        let mj = group_sum[j] / group_n[j];
        chi2 += group_n[j] * (mj - mean_a).powi(2);
    }
    chi2 /= var_a.max(1e-300);
    let dfn = (groups.len() - 1) as f64;
    let pvalue = chi2_survival(chi2, dfn);
    Ok(VarianceTestResult { statistic: chi2, pvalue, df: (dfn, 0.0) })
}

fn median(x: &[f64]) -> f64 {
    let mut v: Vec<f64> = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    } else {
        v[n / 2]
    }
}

fn inv_normal_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro rational approximation.
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

fn f_survival(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    let x = d2 / (d2 + d1 * f);
    regularised_incomplete_beta(x, d2 / 2.0, d1 / 2.0)
}

fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    1.0 - lower_regularised_gamma(df / 2.0, x / 2.0)
}

fn lower_regularised_gamma(s: f64, x: f64) -> f64 {
    if x < 0.0 || s <= 0.0 {
        return 0.0;
    }
    if x < s + 1.0 {
        gamma_series(s, x)
    } else {
        1.0 - gamma_continued_fraction(s, x)
    }
}

fn gamma_series(s: f64, x: f64) -> f64 {
    let mut sum = 1.0 / s;
    let mut term = sum;
    for n in 1..200 {
        term *= x / (s + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 3e-15 {
            break;
        }
    }
    sum * (-x + s * x.ln() - ln_gamma(s)).exp()
}

fn gamma_continued_fraction(s: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - s;
    let mut c = 1.0 / 1e-300;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - s);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = b + an / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 3e-15 {
            break;
        }
    }
    (-x + s * x.ln() - ln_gamma(s)).exp() * h
}

fn regularised_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = ((a * x.ln() + b * (1.0 - x).ln()) - ln_beta).exp() / a;
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(x, a, b)
    } else {
        1.0 - front * betacf(1.0 - x, b, a)
    }
}

fn betacf(x: f64, a: f64, b: f64) -> f64 {
    let mut c = 1.0_f64;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-300 {
        d = 1e-300;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let mf = m as f64;
        let two_m = 2.0 * mf;
        let mut aa = mf * (b - mf) * x / ((qam + two_m) * (a + two_m));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + mf) * (qab + mf) * x / ((a + two_m) * (qap + two_m));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 3e-15 {
            break;
        }
    }
    h
}

fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let cof = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_5e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = cof[0];
        let t = x + g + 0.5;
        for (i, &c) in cof.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levene_detects_variance_difference_between_two_groups() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0_f64, 100.0, 200.0, 300.0, 400.0];
        let r = levene(&[a, b], LeveneCenter::Median).unwrap();
        assert!(r.pvalue < 0.1);
    }

    #[test]
    fn bartlett_detects_variance_difference() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0_f64, 100.0, 200.0, 300.0, 400.0];
        let r = bartlett(&[a, b]).unwrap();
        assert!(r.pvalue < 0.1);
    }

    #[test]
    fn fligner_detects_variance_difference() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0_f64, 100.0, 200.0, 300.0, 400.0, 500.0];
        let r = fligner(&[a, b]).unwrap();
        assert!(r.pvalue < 0.1);
    }
}
