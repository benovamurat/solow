//! Non-parametric two-sample and k-sample tests.
//!
//! * [`mannwhitneyu`] — Mann-Whitney U (Wilcoxon rank-sum).
//! * [`kruskal`] — Kruskal-Wallis one-way ANOVA on ranks.
//! * [`mcnemar`] — McNemar's paired-binary test.
//! * [`chi2_contingency`] — Pearson χ² test of independence.

use solow_core::{Error, Result};

/// A one-degree-of-freedom test statistic + p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestResult {
    /// The test statistic.
    pub statistic: f64,
    /// Two-sided p-value.
    pub pvalue: f64,
}

/// Mann-Whitney U test — two-sided, normal-approximation p-value with
/// tie correction (Mann-Whitney 1947).
pub fn mannwhitneyu(x: &[f64], y: &[f64]) -> Result<TestResult> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::Value("mannwhitneyu: both samples must be non-empty".into()));
    }
    let n1 = x.len() as f64;
    let n2 = y.len() as f64;
    let mut combined: Vec<(f64, u8)> =
        x.iter().map(|&v| (v, 0)).chain(y.iter().map(|&v| (v, 1))).collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = combined.len();
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    let mut ties_sum = 0.0_f64;
    while i < n {
        let mut j = i;
        while j + 1 < n && combined[j + 1].0 == combined[i].0 {
            j += 1;
        }
        let avg = ((i + j) as f64 + 2.0) / 2.0;
        let t = (j - i + 1) as f64;
        if t > 1.0 {
            ties_sum += (t.powi(3) - t) / 12.0;
        }
        for k in i..=j {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    let mut r1 = 0.0_f64;
    for i in 0..n {
        if combined[i].1 == 0 {
            r1 += ranks[i];
        }
    }
    let u1 = r1 - n1 * (n1 + 1.0) / 2.0;
    let u2 = n1 * n2 - u1;
    let u = u1.min(u2);
    let mean = n1 * n2 / 2.0;
    let n_all = n1 + n2;
    let var = n1 * n2 / 12.0 * (n_all + 1.0 - ties_sum / (n_all * (n_all - 1.0) / 12.0));
    let z = (u - mean) / var.sqrt().max(1e-300);
    let pvalue = 2.0 * standard_normal_survival(z.abs());
    Ok(TestResult { statistic: u, pvalue })
}

/// Kruskal-Wallis one-way ANOVA on ranks. Returns `(H, p)` where `H`
/// approximately follows a χ²(k − 1) distribution.
pub fn kruskal(groups: &[Vec<f64>]) -> Result<TestResult> {
    if groups.len() < 2 {
        return Err(Error::Value("kruskal: need ≥ 2 groups".into()));
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total < 3 {
        return Err(Error::Value("kruskal: need ≥ 3 total samples".into()));
    }
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n_total);
    for (gi, g) in groups.iter().enumerate() {
        for &v in g {
            combined.push((v, gi));
        }
    }
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut ranks = vec![0.0_f64; n_total];
    let mut i = 0;
    while i < n_total {
        let mut j = i;
        while j + 1 < n_total && combined[j + 1].0 == combined[i].0 {
            j += 1;
        }
        let avg = ((i + j) as f64 + 2.0) / 2.0;
        for k in i..=j {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    let k = groups.len() as f64;
    let mut rank_sum = vec![0.0_f64; groups.len()];
    let mut group_size = vec![0.0_f64; groups.len()];
    for i in 0..n_total {
        rank_sum[combined[i].1] += ranks[i];
        group_size[combined[i].1] += 1.0;
    }
    let n = n_total as f64;
    let mut h = 0.0_f64;
    for j in 0..groups.len() {
        if group_size[j] > 0.0 {
            h += rank_sum[j] * rank_sum[j] / group_size[j];
        }
    }
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
    let pvalue = chi2_survival(h, k - 1.0);
    Ok(TestResult { statistic: h, pvalue })
}

/// McNemar's test on a 2 × 2 paired-binary table.
pub fn mcnemar(b: usize, c: usize, exact: bool) -> Result<TestResult> {
    let bf = b as f64;
    let cf = c as f64;
    if b + c == 0 {
        return Err(Error::Value("mcnemar: b + c must be > 0".into()));
    }
    if exact {
        // Two-sided exact binomial p on min(b, c) with n = b + c, p = 0.5.
        let n = b + c;
        let k = b.min(c);
        let mut p = 0.0_f64;
        for i in 0..=k {
            p += binomial_pmf(n, i, 0.5);
        }
        let pvalue = (2.0 * p).min(1.0);
        Ok(TestResult { statistic: k as f64, pvalue })
    } else {
        let stat = (bf - cf).powi(2) / (bf + cf);
        let pvalue = chi2_survival(stat, 1.0);
        Ok(TestResult { statistic: stat, pvalue })
    }
}

/// Pearson χ² test of independence on a contingency table (rows ×
/// cols). Returns `(χ², p, dof, expected)`.
pub fn chi2_contingency(observed: &[Vec<f64>]) -> Result<(f64, f64, usize, Vec<Vec<f64>>)> {
    if observed.is_empty() || observed[0].is_empty() {
        return Err(Error::Value("chi2_contingency: empty table".into()));
    }
    let r = observed.len();
    let c = observed[0].len();
    for row in observed {
        if row.len() != c {
            return Err(Error::Value(
                "chi2_contingency: rows have inconsistent widths".into(),
            ));
        }
    }
    let mut row_sums = vec![0.0_f64; r];
    let mut col_sums = vec![0.0_f64; c];
    let mut total = 0.0_f64;
    for i in 0..r {
        for j in 0..c {
            row_sums[i] += observed[i][j];
            col_sums[j] += observed[i][j];
            total += observed[i][j];
        }
    }
    let mut expected = vec![vec![0.0_f64; c]; r];
    for i in 0..r {
        for j in 0..c {
            expected[i][j] = row_sums[i] * col_sums[j] / total.max(1e-300);
        }
    }
    let mut chi2 = 0.0_f64;
    for i in 0..r {
        for j in 0..c {
            let e = expected[i][j];
            if e > 0.0 {
                let d = observed[i][j] - e;
                chi2 += d * d / e;
            }
        }
    }
    let dof = (r - 1) * (c - 1);
    let pvalue = chi2_survival(chi2, dof as f64);
    Ok((chi2, pvalue, dof, expected))
}

fn binomial_pmf(n: usize, k: usize, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    let ln_c = ln_choose(n, k);
    let logp = (ln_c + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp();
    logp
}

fn ln_choose(n: usize, k: usize) -> f64 {
    (1..=k).map(|i| ((n - i + 1) as f64).ln() - (i as f64).ln()).sum()
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

fn standard_normal_survival(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
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
    1.0 - sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mannwhitneyu_flags_a_clear_difference() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
        let r = mannwhitneyu(&a, &b).unwrap();
        assert!(r.pvalue < 0.05);
    }

    #[test]
    fn kruskal_flags_three_group_difference() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![10.0_f64, 11.0, 12.0];
        let c = vec![100.0_f64, 101.0, 102.0];
        let r = kruskal(&[a, b, c]).unwrap();
        assert!(r.pvalue < 0.05);
    }

    #[test]
    fn mcnemar_returns_the_correct_stat_for_a_2x2_table() {
        let r = mcnemar(20, 5, false).unwrap();
        // With b=20, c=5: chi2 = (20-5)^2/25 = 9.
        assert!((r.statistic - 9.0).abs() < 1e-12);
        assert!(r.pvalue < 0.01);
    }

    #[test]
    fn chi2_contingency_returns_a_finite_pvalue() {
        let table = vec![vec![10.0, 20.0, 30.0], vec![6.0, 9.0, 17.0]];
        let (chi2, p, dof, _expected) = chi2_contingency(&table).unwrap();
        assert!(chi2 >= 0.0);
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(dof, 2);
    }
}
