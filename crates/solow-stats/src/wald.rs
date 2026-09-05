//! Generic Wald and F tests for linear restrictions on a parameter vector.
//!
//! Both take a fitted parameter vector `θ̂`, its estimated covariance
//! `V̂`, and a linear restriction `R θ = r`. `R` is `(q × k)` (one row
//! per restriction), `r` is a `q`-vector.
//!
//! * [`wald_test`] — asymptotic Wald statistic `(R θ̂ - r)' (R V̂ R')⁻¹
//!   (R θ̂ - r)`, distributed χ²(q) under the null.
//! * [`f_test`] — same numerator, divided by `q`, referenced against the
//!   `F(q, df_denom)` distribution. This is the small-sample-corrected
//!   version usually reported for OLS.
//!
//! Both use a numerically-stable Cholesky-based inversion on `R V̂ R'`
//! and return the statistic together with its degrees of freedom and
//! p-value.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Result of a linear-restriction Wald test.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct WaldResult {
    /// Wald statistic `(R θ̂ - r)' (R V̂ R')⁻¹ (R θ̂ - r)`.
    pub statistic: f64,
    /// Upper-tail χ² p-value under `df` degrees of freedom.
    pub p_value: f64,
    /// Number of restrictions (rows of `R`).
    pub df: usize,
    /// `R θ̂ - r` — the fitted deviation from the restriction.
    pub restriction_gap: Array1<f64>,
}

/// Result of a linear-restriction F test.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FTestResult {
    /// F statistic `(R θ̂ - r)' (R V̂ R')⁻¹ (R θ̂ - r) / q`.
    pub statistic: f64,
    /// Upper-tail F(df_num, df_denom) p-value.
    pub p_value: f64,
    /// Numerator degrees of freedom (number of restrictions).
    pub df_num: usize,
    /// Denominator degrees of freedom.
    pub df_denom: f64,
    /// `R θ̂ - r` — the fitted deviation from the restriction.
    pub restriction_gap: Array1<f64>,
}

/// Wald test on a linear restriction `R θ = r`.
///
/// * `theta` — the fitted parameter vector `θ̂`.
/// * `cov` — its estimated covariance matrix `V̂`.
/// * `r_matrix` — the `(q × k)` restriction matrix.
/// * `r_value` — the `q`-vector `r`.
pub fn wald_test(
    theta: ArrayView1<'_, f64>,
    cov: ArrayView2<'_, f64>,
    r_matrix: ArrayView2<'_, f64>,
    r_value: ArrayView1<'_, f64>,
) -> Result<WaldResult> {
    let k = theta.len();
    let (q, r_cols) = (r_matrix.nrows(), r_matrix.ncols());
    if cov.nrows() != k || cov.ncols() != k {
        return Err(Error::Shape(format!(
            "wald_test: theta has {k} entries but cov is {}x{}",
            cov.nrows(),
            cov.ncols()
        )));
    }
    if r_cols != k {
        return Err(Error::Shape(format!(
            "wald_test: r_matrix has {r_cols} columns but theta has {k} entries"
        )));
    }
    if q == 0 || r_value.len() != q {
        return Err(Error::Shape(format!(
            "wald_test: r_matrix has {q} rows but r_value has {} entries",
            r_value.len()
        )));
    }
    let gap = compute_gap(theta, r_matrix, r_value);
    let stat = quadratic_form(&gap, r_matrix, cov)?;
    let p = chi2_upper_tail(stat, q as f64);
    Ok(WaldResult {
        statistic: stat,
        p_value: p,
        df: q,
        restriction_gap: gap,
    })
}

/// F test on the same restriction `R θ = r`, with `df_denom` denominator
/// degrees of freedom (typically `n - k` for OLS).
pub fn f_test(
    theta: ArrayView1<'_, f64>,
    cov: ArrayView2<'_, f64>,
    r_matrix: ArrayView2<'_, f64>,
    r_value: ArrayView1<'_, f64>,
    df_denom: f64,
) -> Result<FTestResult> {
    if !(df_denom > 0.0 && df_denom.is_finite()) {
        return Err(Error::Value(format!(
            "f_test: df_denom must be finite and > 0 (got {df_denom})"
        )));
    }
    let wald = wald_test(theta, cov, r_matrix, r_value)?;
    let q = wald.df as f64;
    let f_stat = wald.statistic / q;
    Ok(FTestResult {
        statistic: f_stat,
        p_value: f_upper_tail(f_stat, q, df_denom),
        df_num: wald.df,
        df_denom,
        restriction_gap: wald.restriction_gap,
    })
}

fn compute_gap(
    theta: ArrayView1<'_, f64>,
    r_matrix: ArrayView2<'_, f64>,
    r_value: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let q = r_matrix.nrows();
    let mut gap = Array1::<f64>::zeros(q);
    for i in 0..q {
        let mut s = 0.0_f64;
        for j in 0..theta.len() {
            s += r_matrix[[i, j]] * theta[j];
        }
        gap[i] = s - r_value[i];
    }
    gap
}

fn quadratic_form(
    gap: &Array1<f64>,
    r_matrix: ArrayView2<'_, f64>,
    cov: ArrayView2<'_, f64>,
) -> Result<f64> {
    let q = gap.len();
    let k = cov.nrows();
    // R V R' — an (q x q) symmetric matrix. Build by RV then multiply again by R'.
    let mut rv: Array2<f64> = Array2::zeros((q, k));
    for i in 0..q {
        for j in 0..k {
            let mut s = 0.0_f64;
            for a in 0..k {
                s += r_matrix[[i, a]] * cov[[a, j]];
            }
            rv[[i, j]] = s;
        }
    }
    let mut rvr: Vec<Vec<f64>> = vec![vec![0.0; q]; q];
    for i in 0..q {
        for j in 0..q {
            let mut s = 0.0_f64;
            for a in 0..k {
                s += rv[[i, a]] * r_matrix[[j, a]];
            }
            rvr[i][j] = s;
        }
    }
    // Solve (R V R') x = gap by Cholesky-with-fallback Gauss-Jordan.
    let x = solve_symmetric(&mut rvr, gap.as_slice().unwrap())?;
    let mut stat = 0.0_f64;
    for i in 0..q {
        stat += gap[i] * x[i];
    }
    Ok(stat)
}

fn solve_symmetric(m: &mut [Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = Vec::with_capacity(n + 1);
            row.extend_from_slice(&m[i]);
            row.push(rhs[i]);
            row
        })
        .collect();
    for i in 0..n {
        let mut pivot = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Err(Error::Value(
                "wald_test: R V R' is singular; cannot solve the restriction system".into(),
            ));
        }
        if pivot != i {
            a.swap(i, pivot);
        }
        let piv = a[i][i];
        for c in 0..(n + 1) {
            a[i][c] /= piv;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            if factor == 0.0 {
                continue;
            }
            for c in 0..(n + 1) {
                a[r][c] -= factor * a[i][c];
            }
        }
    }
    Ok((0..n).map(|i| a[i][n]).collect())
}

// ---------------------------------------------------------------------------
// χ² and F upper-tail p-values (local — kept private to keep the crate free
// of a heavy stats-distributions dep at this leaf).
// ---------------------------------------------------------------------------

fn chi2_upper_tail(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }
    reg_upper_gamma(0.5 * df, 0.5 * x).clamp(0.0, 1.0)
}

fn f_upper_tail(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    if !f.is_finite() {
        return 0.0;
    }
    let x = df2 / (df2 + df1 * f);
    reg_incomplete_beta(x, 0.5 * df2, 0.5 * df1).clamp(0.0, 1.0)
}

fn reg_upper_gamma(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - lower_series(a, x)
    } else {
        upper_cf(a, x)
    }
}

fn lower_series(a: f64, x: f64) -> f64 {
    const MAXIT: usize = 512;
    const EPS: f64 = 3e-16;
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAXIT {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn upper_cf(a: f64, x: f64) -> f64 {
    const MAXIT: usize = 512;
    const EPS: f64 = 3e-16;
    const FPMIN: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAXIT {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
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
    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn reg_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
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
    const EPS: f64 = 3e-16;
    const FPMIN: f64 = 1e-300;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn wald_zero_when_restriction_is_satisfied() {
        let theta = array![1.0, 2.0, 3.0];
        let cov = Array2::<f64>::eye(3);
        // R = [1, -1, 0], r = -1 → gap = 1 - 2 - (-1) = 0.
        let r_matrix = array![[1.0, -1.0, 0.0]];
        let r_value = array![-1.0];
        let w = wald_test(theta.view(), cov.view(), r_matrix.view(), r_value.view()).unwrap();
        assert!(w.statistic.abs() < 1e-12);
        assert!((w.p_value - 1.0).abs() < 1e-12);
        assert_eq!(w.df, 1);
    }

    #[test]
    fn wald_large_when_restriction_is_far() {
        let theta = array![5.0, 0.0];
        let cov = Array2::<f64>::eye(2);
        let r_matrix = array![[1.0, 0.0]];
        let r_value = array![0.0];
        let w = wald_test(theta.view(), cov.view(), r_matrix.view(), r_value.view()).unwrap();
        // stat = 25 / 1 = 25 → tiny p.
        assert!((w.statistic - 25.0).abs() < 1e-12);
        assert!(w.p_value < 1e-6);
    }

    #[test]
    fn f_test_reduces_to_wald_over_q_for_one_restriction() {
        let theta = array![1.5, 0.0];
        let cov = Array2::<f64>::eye(2);
        let r_matrix = array![[1.0, 0.0]];
        let r_value = array![0.0];
        let w = wald_test(theta.view(), cov.view(), r_matrix.view(), r_value.view()).unwrap();
        let f = f_test(
            theta.view(),
            cov.view(),
            r_matrix.view(),
            r_value.view(),
            100.0,
        )
        .unwrap();
        assert!((f.statistic - w.statistic).abs() < 1e-12);
        assert_eq!(f.df_num, 1);
        assert!((f.df_denom - 100.0).abs() < 1e-12);
    }
}
