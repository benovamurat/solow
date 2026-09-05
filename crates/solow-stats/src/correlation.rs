//! Pearson, Spearman, and Kendall correlation coefficients with
//! two-sided p-values.

use solow_core::{Error, Result};

/// A correlation coefficient with its two-sided p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationResult {
    /// The estimated coefficient in `[-1, 1]`.
    pub statistic: f64,
    /// Two-sided p-value under the null of no association.
    pub pvalue: f64,
}

/// Pearson product-moment correlation.
pub fn pearsonr(x: &[f64], y: &[f64]) -> Result<CorrelationResult> {
    let n = x.len();
    if n < 3 || y.len() != n {
        return Err(Error::Value("pearsonr: need n ≥ 3 and matched lengths".into()));
    }
    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
    let mut sxy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom < 1e-300 {
        return Err(Error::Value("pearsonr: at least one column has zero variance".into()));
    }
    let r = (sxy / denom).clamp(-1.0, 1.0);
    // Two-sided p from a t(n − 2) distribution: t = r · sqrt((n − 2)/(1 − r²)).
    let pvalue = if r.abs() >= 1.0 - 1e-14 {
        0.0
    } else {
        let df = (n - 2) as f64;
        let t = r * (df / (1.0 - r * r)).sqrt();
        2.0 * student_t_survival(t.abs(), df)
    };
    Ok(CorrelationResult { statistic: r, pvalue })
}

/// Spearman rank correlation.
pub fn spearmanr(x: &[f64], y: &[f64]) -> Result<CorrelationResult> {
    if x.len() != y.len() || x.len() < 3 {
        return Err(Error::Value("spearmanr: need n ≥ 3 and matched lengths".into()));
    }
    let rx = ranks_with_ties(x);
    let ry = ranks_with_ties(y);
    pearsonr(&rx, &ry)
}

/// Kendall τ-b — with ties correction.
pub fn kendalltau(x: &[f64], y: &[f64]) -> Result<CorrelationResult> {
    let n = x.len();
    if y.len() != n || n < 3 {
        return Err(Error::Value("kendalltau: need n ≥ 3 and matched lengths".into()));
    }
    let mut concordant = 0_i64;
    let mut discordant = 0_i64;
    let mut ties_x = 0_i64;
    let mut ties_y = 0_i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let sx = dx.signum();
            let sy = dy.signum();
            if dx == 0.0 && dy == 0.0 {
                // ignore joint ties
            } else if dx == 0.0 {
                ties_x += 1;
            } else if dy == 0.0 {
                ties_y += 1;
            } else if sx == sy {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let n0 = n as f64 * (n as f64 - 1.0) / 2.0;
    let tau_b = (concordant - discordant) as f64
        / (((n0 - ties_x as f64) * (n0 - ties_y as f64)).sqrt().max(1e-300));
    // Two-sided normal-approximation p-value.
    let var = (2.0 * (2.0 * n as f64 + 5.0)) / (9.0 * n as f64 * (n as f64 - 1.0));
    let z = tau_b / var.sqrt();
    let pvalue = 2.0 * standard_normal_survival(z.abs());
    Ok(CorrelationResult { statistic: tau_b, pvalue })
}

fn ranks_with_ties(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = ((i + j) as f64 + 2.0) / 2.0; // 1-based
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn standard_normal_survival(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 approximation (max error ≈ 1.5e-7).
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

fn student_t_survival(t: f64, df: f64) -> f64 {
    // Two-sided-safe survival S(t) = P(T > t) for T ~ t(df).
    // Uses the incomplete-beta identity:
    //   S(t) = 0.5 · I_{df/(df+t²)}(df/2, 1/2)  for t > 0.
    if t <= 0.0 {
        return 0.5;
    }
    let x = df / (df + t * t);
    0.5 * regularised_incomplete_beta(x, df / 2.0, 0.5)
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
        1.0 - front * betacf(1.0 - x, b, a) * (front / front).max(1.0)
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
    // Lanczos approximation.
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
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x)
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
    fn pearsonr_recovers_perfect_positive_correlation() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let r = pearsonr(&x, &y).unwrap();
        assert!((r.statistic - 1.0).abs() < 1e-12);
        assert!(r.pvalue < 1e-6);
    }

    #[test]
    fn spearmanr_handles_ties_correctly() {
        let x = vec![1.0_f64, 2.0, 2.0, 3.0, 4.0];
        let y = vec![1.0_f64, 3.0, 3.0, 5.0, 7.0];
        let r = spearmanr(&x, &y).unwrap();
        assert!(r.statistic > 0.9);
    }

    #[test]
    fn kendalltau_returns_a_value_in_the_valid_range() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0_f64, 4.0, 3.0, 2.0, 1.0];
        let r = kendalltau(&x, &y).unwrap();
        assert!((r.statistic - (-1.0)).abs() < 1e-12);
        assert!(r.pvalue < 0.1);
    }
}
