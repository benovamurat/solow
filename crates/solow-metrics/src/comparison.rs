//! Statistical tests for comparing several models across benchmarks.
//!
//! The classical Demšar (2006) recipe: when comparing `k` models on `m`
//! datasets or CV folds, use the non-parametric **Friedman test** to
//! decide whether the models differ at all, and if it rejects, do the
//! **Nemenyi post-hoc** on the mean ranks (or, for pairwise comparisons,
//! the paired **Wilcoxon signed-rank** test). Everything is
//! distribution-free — no normality assumption on the score.
//!
//! Input is always an `(m × k)` score matrix (higher-is-better in the
//! Friedman/Nemenyi convention; pass `-loss` if you have a loss). This
//! module makes no distributional assumption about the score itself.

use ndarray::ArrayView2;
use solow_core::{Error, Result};

/// Result of a Friedman test.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FriedmanResult {
    /// The Iman-Davenport F-adjusted Friedman statistic (better small-sample
    /// behaviour than the plain χ² form). Follows `F(k - 1, (k - 1)(m - 1))`.
    pub statistic: f64,
    /// Upper-tail F p-value.
    pub p_value: f64,
    /// Mean rank of each model (over the `m` datasets); lower = better on
    /// the Friedman convention that ties get averaged ranks.
    pub mean_ranks: Vec<f64>,
    /// Number of datasets / folds.
    pub m: usize,
    /// Number of models.
    pub k: usize,
}

/// Friedman test on a score matrix.
///
/// `scores` has shape `(m, k)` — one row per dataset/fold, one column
/// per model. Higher entries are better; ties get averaged ranks.
pub fn friedman_test(scores: ArrayView2<'_, f64>) -> Result<FriedmanResult> {
    let (m, k) = (scores.nrows(), scores.ncols());
    if m < 2 || k < 2 {
        return Err(Error::Value(format!(
            "friedman_test: need at least 2 datasets and 2 models (got {m} × {k})"
        )));
    }
    let mut ranks = vec![vec![0.0_f64; k]; m];
    for i in 0..m {
        let row: Vec<(usize, f64)> = (0..k).map(|c| (c, scores[[i, c]])).collect();
        // Rank descending (higher = better rank 1); with average ties.
        let mut order: Vec<(usize, f64)> = row.clone();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut j = 0;
        while j < k {
            let mut jj = j + 1;
            while jj < k && (order[jj].1 - order[j].1).abs() < 1e-12 {
                jj += 1;
            }
            let avg_rank = ((j + 1) as f64 + jj as f64) / 2.0;
            for t in j..jj {
                ranks[i][order[t].0] = avg_rank;
            }
            j = jj;
        }
    }
    let mut r_bar = vec![0.0_f64; k];
    for j in 0..k {
        for i in 0..m {
            r_bar[j] += ranks[i][j];
        }
        r_bar[j] /= m as f64;
    }
    let expected = (k as f64 + 1.0) / 2.0;
    let ss: f64 = r_bar.iter().map(|r| (r - expected).powi(2)).sum();
    let chi2 = 12.0 * (m as f64) / (k as f64 * (k as f64 + 1.0)) * ss;
    // Iman-Davenport F adjustment.
    let denom = (m as f64) * (k as f64 - 1.0) - chi2;
    let f_stat = if denom.abs() < 1e-300 {
        f64::INFINITY
    } else {
        ((m as f64 - 1.0) * chi2) / denom
    };
    let df1 = (k - 1) as f64;
    let df2 = ((k - 1) * (m - 1)) as f64;
    let p_value = f_upper_tail(f_stat, df1, df2);
    Ok(FriedmanResult {
        statistic: f_stat,
        p_value,
        mean_ranks: r_bar,
        m,
        k,
    })
}

/// Nemenyi critical difference for `k` models on `m` datasets at
/// significance level `alpha`.
///
/// Two models differ significantly if `|R̄ᵢ - R̄ⱼ| > CD`. The critical
/// value `q_α` is drawn from the studentized range distribution
/// tabulated in Demšar (2006, Table 5); this function returns the
/// closed-form
/// `CD = q_α · √(k · (k + 1) / (6 · m))`
/// with a small lookup for the common alphas 0.05 and 0.10.
pub fn nemenyi_critical_difference(k: usize, m: usize, alpha: f64) -> Result<f64> {
    if k < 2 || m < 1 {
        return Err(Error::Value(
            "nemenyi_critical_difference: k ≥ 2 and m ≥ 1 required".into(),
        ));
    }
    let q = nemenyi_q(k, alpha)?;
    let cd = q * ((k as f64 * (k as f64 + 1.0)) / (6.0 * m as f64)).sqrt();
    Ok(cd)
}

/// Result of a Wilcoxon signed-rank test between two paired samples.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WilcoxonResult {
    /// The `W = min(W_+, W_-)` statistic on the ranks of non-zero
    /// differences.
    pub statistic: f64,
    /// Normal-approximation p-value (two-sided) with a tie correction
    /// and no continuity correction (SciPy default).
    pub p_value: f64,
    /// Number of non-zero differences used.
    pub n_effective: usize,
}

/// Wilcoxon signed-rank test on paired samples.
///
/// The two-sided p-value uses the normal approximation with tie
/// correction, following `scipy.stats.wilcoxon(..., zero_method='wilcox')`.
/// Zero differences are dropped (Wilcoxon's original convention);
/// callers who prefer Pratt's method must add them back explicitly.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> Result<WilcoxonResult> {
    if a.len() != b.len() {
        return Err(Error::Shape(format!(
            "wilcoxon_signed_rank: a has {} entries but b has {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(Error::Value(
            "wilcoxon_signed_rank: at least one paired sample is required".into(),
        ));
    }
    // Non-zero differences.
    let d: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x - y)
        .filter(|d| d.abs() > 0.0)
        .collect();
    let n = d.len();
    if n < 2 {
        return Err(Error::Value(
            "wilcoxon_signed_rank: need ≥ 2 non-zero differences".into(),
        ));
    }
    // Rank |d| with average ties.
    let mut order: Vec<(usize, f64)> = d.iter().enumerate().map(|(i, x)| (i, x.abs())).collect();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0_f64; n];
    let mut tie_terms: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (order[j].1 - order[i].1).abs() < 1e-12 {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for t in i..j {
            ranks[order[t].0] = avg;
        }
        let t_size = (j - i) as f64;
        if t_size > 1.0 {
            tie_terms.push(t_size * t_size * t_size - t_size);
        }
        i = j;
    }
    let w_plus: f64 = ranks
        .iter()
        .zip(d.iter())
        .filter_map(|(r, x)| if *x > 0.0 { Some(*r) } else { None })
        .sum();
    let w_minus: f64 = ranks
        .iter()
        .zip(d.iter())
        .filter_map(|(r, x)| if *x < 0.0 { Some(*r) } else { None })
        .sum();
    let w = w_plus.min(w_minus);
    let n_f = n as f64;
    let mean = n_f * (n_f + 1.0) / 4.0;
    let mut var = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0;
    let tie_correction: f64 = tie_terms.iter().sum::<f64>() / 48.0;
    var -= tie_correction;
    if var <= 0.0 {
        return Err(Error::Value(
            "wilcoxon_signed_rank: degenerate variance (too many ties)".into(),
        ));
    }
    let z = (w - mean) / var.sqrt();
    let p = 2.0 * standard_normal_sf(z.abs());
    Ok(WilcoxonResult {
        statistic: w,
        p_value: p.clamp(0.0, 1.0),
        n_effective: n,
    })
}

// ---------------------------------------------------------------------------
// Helpers — F upper-tail p-value, standard-normal SF, Nemenyi q table
// ---------------------------------------------------------------------------

fn f_upper_tail(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    if !f.is_finite() {
        // Positive infinity → the upper tail has probability zero.
        return 0.0;
    }
    let x = df2 / (df2 + df1 * f);
    regularized_incomplete_beta(x, 0.5 * df2, 0.5 * df1).clamp(0.0, 1.0)
}

fn standard_normal_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Chebyshev-based Numerical Recipes erfc; accurate to ~1e-7.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Studentized-range critical values for the Nemenyi post-hoc test
/// (from Demšar 2006, Table 5). Only the two most commonly used
/// alphas (0.05 and 0.10) are tabulated; unknown alphas return
/// [`Error::Value`].
fn nemenyi_q(k: usize, alpha: f64) -> Result<f64> {
    // Table columns: k = 2..=10, rows α = 0.05, 0.10.
    let q05: [f64; 9] = [
        1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164,
    ];
    let q10: [f64; 9] = [
        1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920,
    ];
    if !(2..=10).contains(&k) {
        return Err(Error::Value(format!(
            "nemenyi_q: only k in [2, 10] is tabulated (got {k})"
        )));
    }
    let idx = k - 2;
    if (alpha - 0.05).abs() < 1e-9 {
        Ok(q05[idx])
    } else if (alpha - 0.10).abs() < 1e-9 {
        Ok(q10[idx])
    } else {
        Err(Error::Value(format!(
            "nemenyi_q: only alpha = 0.05 or 0.10 is tabulated (got {alpha})"
        )))
    }
}

fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(x, a, b) / a
    } else {
        1.0 - bt * betacf(1.0 - x, b, a) / b
    }
}

fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAXIT: usize = 512;
    const EPS: f64 = 3.0e-16;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEFF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFF[0];
        for (i, &c) in COEFF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}
