//! Forecasting-specific metrics.
//!
//! These are the losses reported for time-series forecasts, above and beyond
//! the point-error metrics in [`crate::regression`]:
//!
//! * [`mase`] — Mean Absolute Scaled Error, the seasonal-naive scaled MAE
//!   introduced by Hyndman and Koehler (2006).
//! * [`rmsse`] — Root Mean Squared Scaled Error, the M5-competition variant.
//! * [`pinball_loss`] — the quantile loss for one probabilistic quantile
//!   forecast.
//! * [`interval_coverage`] — the fraction of prediction intervals that
//!   actually contain the observation.
//! * [`mean_interval_score`] — Winkler's proper score for a two-sided
//!   `(1 - α)` prediction interval.

use ndarray::ArrayView1;
use solow_core::{Error, Result};

fn check_pair(name: &str, y: ArrayView1<'_, f64>, yhat: ArrayView1<'_, f64>) -> Result<()> {
    if y.len() != yhat.len() {
        return Err(Error::Shape(format!(
            "{name}: y_true has {} entries but y_pred has {}",
            y.len(),
            yhat.len()
        )));
    }
    if y.is_empty() {
        return Err(Error::Value(format!(
            "{name}: at least one sample is required"
        )));
    }
    for &v in y.iter().chain(yhat.iter()) {
        if !v.is_finite() {
            return Err(Error::Value(format!("{name}: inputs must be finite")));
        }
    }
    Ok(())
}

/// Mean Absolute Scaled Error.
///
/// `y_train` is the in-sample series used to compute the seasonal-naive
/// benchmark; `seasonality = m` scales by `mean(|y_train[t] - y_train[t - m]|)`
/// (so `m = 1` reproduces the classical MASE and, on strictly seasonal data,
/// `m = 12` gives the monthly-seasonal MASE).
pub fn mase(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    y_train: ArrayView1<'_, f64>,
    seasonality: usize,
) -> Result<f64> {
    check_pair("mase", y_true, y_pred)?;
    if seasonality == 0 {
        return Err(Error::Value("mase: seasonality must be ≥ 1".into()));
    }
    if y_train.len() <= seasonality {
        return Err(Error::Value(format!(
            "mase: y_train has only {} points, needs > seasonality = {}",
            y_train.len(),
            seasonality
        )));
    }
    for &v in y_train.iter() {
        if !v.is_finite() {
            return Err(Error::Value("mase: y_train must be finite".into()));
        }
    }
    let mut denom = 0.0_f64;
    for t in seasonality..y_train.len() {
        denom += (y_train[t] - y_train[t - seasonality]).abs();
    }
    denom /= (y_train.len() - seasonality) as f64;
    if denom == 0.0 {
        return Err(Error::Value(
            "mase: seasonal-naive benchmark has zero mean absolute error (constant y_train)".into(),
        ));
    }
    let mut num = 0.0_f64;
    for (a, b) in y_true.iter().zip(y_pred.iter()) {
        num += (a - b).abs();
    }
    num /= y_true.len() as f64;
    Ok(num / denom)
}

/// Root Mean Squared Scaled Error (the M5-competition scale).
pub fn rmsse(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    y_train: ArrayView1<'_, f64>,
    seasonality: usize,
) -> Result<f64> {
    check_pair("rmsse", y_true, y_pred)?;
    if seasonality == 0 {
        return Err(Error::Value("rmsse: seasonality must be ≥ 1".into()));
    }
    if y_train.len() <= seasonality {
        return Err(Error::Value(format!(
            "rmsse: y_train has only {} points, needs > seasonality = {}",
            y_train.len(),
            seasonality
        )));
    }
    for &v in y_train.iter() {
        if !v.is_finite() {
            return Err(Error::Value("rmsse: y_train must be finite".into()));
        }
    }
    let mut denom = 0.0_f64;
    for t in seasonality..y_train.len() {
        let d = y_train[t] - y_train[t - seasonality];
        denom += d * d;
    }
    denom /= (y_train.len() - seasonality) as f64;
    if denom == 0.0 {
        return Err(Error::Value(
            "rmsse: seasonal-naive benchmark has zero MSE (constant y_train)".into(),
        ));
    }
    let mut num = 0.0_f64;
    for (a, b) in y_true.iter().zip(y_pred.iter()) {
        let d = a - b;
        num += d * d;
    }
    num /= y_true.len() as f64;
    Ok((num / denom).sqrt())
}

/// Pinball loss for a quantile forecast at level `tau ∈ (0, 1)`.
pub fn pinball_loss(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    tau: f64,
) -> Result<f64> {
    check_pair("pinball_loss", y_true, y_pred)?;
    if !(0.0 < tau && tau < 1.0) {
        return Err(Error::Value(format!(
            "pinball_loss: tau must be in (0, 1), got {tau}"
        )));
    }
    let mut sum = 0.0_f64;
    for (&y, &q) in y_true.iter().zip(y_pred.iter()) {
        let d = y - q;
        sum += if d >= 0.0 { tau * d } else { (tau - 1.0) * d };
    }
    Ok(sum / y_true.len() as f64)
}

/// Empirical coverage of a two-sided prediction interval.
pub fn interval_coverage(
    y_true: ArrayView1<'_, f64>,
    y_lower: ArrayView1<'_, f64>,
    y_upper: ArrayView1<'_, f64>,
) -> Result<f64> {
    check_pair("interval_coverage", y_true, y_lower)?;
    check_pair("interval_coverage", y_true, y_upper)?;
    let n = y_true.len();
    let mut hits = 0usize;
    for i in 0..n {
        if y_upper[i] < y_lower[i] {
            return Err(Error::Value(format!(
                "interval_coverage: sample {i} has upper < lower"
            )));
        }
        if y_lower[i] <= y_true[i] && y_true[i] <= y_upper[i] {
            hits += 1;
        }
    }
    Ok(hits as f64 / n as f64)
}

/// Mean Winkler / interval score at nominal miscoverage `alpha ∈ (0, 1)`.
///
/// Lower is better. A perfectly tight, perfectly-covering interval scores
/// `(u - l)`; observations outside `[l, u]` add an `(2/α) · outside` penalty.
pub fn mean_interval_score(
    y_true: ArrayView1<'_, f64>,
    y_lower: ArrayView1<'_, f64>,
    y_upper: ArrayView1<'_, f64>,
    alpha: f64,
) -> Result<f64> {
    check_pair("mean_interval_score", y_true, y_lower)?;
    check_pair("mean_interval_score", y_true, y_upper)?;
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(Error::Value(format!(
            "mean_interval_score: alpha must be in (0, 1), got {alpha}"
        )));
    }
    let n = y_true.len();
    let mut sum = 0.0_f64;
    for i in 0..n {
        let (l, u, y) = (y_lower[i], y_upper[i], y_true[i]);
        if u < l {
            return Err(Error::Value(format!(
                "mean_interval_score: sample {i} has upper < lower"
            )));
        }
        let width = u - l;
        let under = if y < l { (2.0 / alpha) * (l - y) } else { 0.0 };
        let over = if y > u { (2.0 / alpha) * (y - u) } else { 0.0 };
        sum += width + under + over;
    }
    Ok(sum / n as f64)
}

// ---------------------------------------------------------------------------
// Diebold-Mariano forecast-comparison test
// ---------------------------------------------------------------------------

/// Loss function for [`diebold_mariano`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DmLoss {
    /// Squared-error loss `(y - f)²`.
    SquaredError,
    /// Absolute-error loss `|y - f|`.
    AbsoluteError,
}

impl DmLoss {
    fn evaluate(&self, y: f64, f: f64) -> f64 {
        match self {
            DmLoss::SquaredError => {
                let d = y - f;
                d * d
            }
            DmLoss::AbsoluteError => (y - f).abs(),
        }
    }
}

/// Result of a Diebold-Mariano test.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DieboldMarianoResult {
    /// The DM statistic (Harvey-Leybourne-Newbold small-sample corrected
    /// version — this is the modified Diebold-Mariano statistic).
    pub statistic: f64,
    /// Two-sided p-value from the Student-t reference distribution with
    /// `n - 1` degrees of freedom (matching the HLN recommendation for
    /// small samples).
    pub p_value: f64,
    /// Degrees of freedom used to compute [`Self::p_value`].
    pub df: f64,
    /// Mean loss differential `d̄ = mean(L(y, f₁) - L(y, f₂))`.
    /// A positive value indicates `f₂` had lower loss.
    pub mean_loss_diff: f64,
    /// Newey-West long-run variance of the loss differential.
    pub long_run_variance: f64,
    /// Sample size.
    pub n: usize,
}

/// Diebold-Mariano test for equal predictive accuracy of two forecasts
/// `f1`, `f2` of the same target `y`, at forecast horizon `horizon`.
///
/// The reported statistic is the Harvey-Leybourne-Newbold (1997)
/// small-sample-corrected version of Diebold-Mariano (1995): the raw DM
/// statistic is scaled by `sqrt((n + 1 - 2h + h(h - 1)/n) / n)` and its
/// p-value read from a Student-t with `n - 1` degrees of freedom. This is
/// the recommended default over the asymptotic `N(0, 1)` p-value; for
/// `h = 1` the correction is negligible on long series and reduces to DM.
///
/// The long-run variance of the loss differential is estimated by
/// Newey-West with a lag truncation of `horizon - 1` (the standard choice
/// for an `h`-step forecast whose errors are MA(`h - 1`) under the null).
pub fn diebold_mariano(
    y: ArrayView1<'_, f64>,
    f1: ArrayView1<'_, f64>,
    f2: ArrayView1<'_, f64>,
    horizon: usize,
    loss: DmLoss,
) -> Result<DieboldMarianoResult> {
    check_pair("diebold_mariano", y, f1)?;
    check_pair("diebold_mariano", y, f2)?;
    if horizon == 0 {
        return Err(Error::Value("diebold_mariano: horizon must be ≥ 1".into()));
    }
    let n = y.len();
    if n <= horizon {
        return Err(Error::Value(format!(
            "diebold_mariano: n = {n} must exceed horizon = {horizon}"
        )));
    }
    // Loss differential d_t = L(y_t, f1_t) - L(y_t, f2_t).
    let d: Vec<f64> = (0..n)
        .map(|t| loss.evaluate(y[t], f1[t]) - loss.evaluate(y[t], f2[t]))
        .collect();
    let n_f = n as f64;
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;

    // Newey-West long-run variance with lag truncation `horizon - 1`.
    let l_star = horizon.saturating_sub(1);
    // gamma_0
    let mut gamma_sum = d.iter().map(|di| (di - d_bar).powi(2)).sum::<f64>() / n_f;
    // sum_{k=1..=l_star} 2 * (1 - k/(l_star+1)) * gamma_k
    for k in 1..=l_star {
        let mut gk = 0.0_f64;
        for t in k..n {
            gk += (d[t] - d_bar) * (d[t - k] - d_bar);
        }
        gk /= n_f;
        let bartlett = 1.0 - (k as f64) / ((l_star + 1) as f64);
        gamma_sum += 2.0 * bartlett * gk;
    }
    // Guard against a negative estimate produced by Bartlett kernel on a
    // very short series — clip to a small positive.
    let long_run_variance = gamma_sum.max(f64::EPSILON);

    let dm = d_bar / (long_run_variance / n_f).sqrt();
    // Harvey-Leybourne-Newbold small-sample correction.
    let h = horizon as f64;
    let scale = ((n_f + 1.0 - 2.0 * h + h * (h - 1.0) / n_f) / n_f)
        .max(0.0)
        .sqrt();
    let stat_hln = dm * scale;
    let df = n_f - 1.0;
    let p_value = two_sided_student_t_pvalue(stat_hln, df);
    Ok(DieboldMarianoResult {
        statistic: stat_hln,
        p_value,
        df,
        mean_loss_diff: d_bar,
        long_run_variance,
        n,
    })
}

// ---------------------------------------------------------------------------
// Student-t two-sided p-value (from t survival function via incomplete beta)
// ---------------------------------------------------------------------------

fn two_sided_student_t_pvalue(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let sf_one_side = 0.5 * regularized_incomplete_beta(x, 0.5 * df, 0.5);
    (2.0 * sf_one_side).clamp(0.0, 1.0)
}

// Regularized incomplete beta I_x(a, b) via a Lentz continued-fraction
// expansion (Numerical Recipes §6.4). This lives here rather than as a
// dependency on solow-distributions to keep solow-metrics free of a
// heavy statistical-distributions dep — the DM p-value is the only place
// it is needed inside this crate.
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

// Lanczos-approximation log-gamma (g = 7, n = 9); accurate to ~1e-15 for
// real arguments in the reflection-mapped domain.
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
        // Reflection formula
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

// ---------------------------------------------------------------------------
// Giacomini-White (2006) conditional predictive-ability test
// ---------------------------------------------------------------------------

use ndarray::ArrayView2;

/// Result of a Giacomini-White conditional predictive-ability test.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiacominiWhiteResult {
    /// The Wald statistic `n · Z̄' V̂⁻¹ Z̄`. Follows χ²(k) asymptotically
    /// under the null of equal conditional predictive ability, where `k`
    /// is the number of conditioning regressors.
    pub statistic: f64,
    /// One-sided upper-tail χ² p-value.
    pub p_value: f64,
    /// Degrees of freedom (number of conditioning regressors).
    pub df: usize,
    /// Sample size (number of forecast pairs).
    pub n: usize,
}

/// Giacomini-White (2006) conditional predictive-ability test.
///
/// This generalises Diebold-Mariano to a **conditional** null: the two
/// forecasts have equal accuracy *given* the information encoded in a
/// user-supplied matrix of `test_regressors`. The classical unconditional
/// DM test is recovered when `test_regressors` is a single all-ones column
/// (the constant regressor).
///
/// `test_regressors` has shape `(n, k)` — one row per forecast period,
/// one column per conditioning variable. Typical choices are:
/// a constant column (recovers DM), lagged loss differentials (Andrews-
/// style tests), or macro-financial predictors (Giacomini-White original).
///
/// The test statistic is `n · Z̄' V̂⁻¹ Z̄` where `Z_t = h_t · d_t` (test
/// regressors × loss differential), `Z̄` its sample mean, and `V̂` its
/// heteroskedasticity- and autocorrelation-consistent covariance
/// (Newey-West at lag `horizon - 1`). Under the null the statistic is
/// asymptotically χ² with `k` degrees of freedom.
pub fn giacomini_white_test(
    y: ArrayView1<'_, f64>,
    f1: ArrayView1<'_, f64>,
    f2: ArrayView1<'_, f64>,
    test_regressors: ArrayView2<'_, f64>,
    horizon: usize,
    loss: DmLoss,
) -> Result<GiacominiWhiteResult> {
    check_pair("giacomini_white_test", y, f1)?;
    check_pair("giacomini_white_test", y, f2)?;
    if horizon == 0 {
        return Err(Error::Value(
            "giacomini_white_test: horizon must be ≥ 1".into(),
        ));
    }
    let n = y.len();
    if test_regressors.nrows() != n {
        return Err(Error::Shape(format!(
            "giacomini_white_test: test_regressors has {} rows but y has {n}",
            test_regressors.nrows()
        )));
    }
    let k = test_regressors.ncols();
    if k == 0 {
        return Err(Error::Value(
            "giacomini_white_test: test_regressors must have at least one column".into(),
        ));
    }
    if n <= horizon || n <= k {
        return Err(Error::Value(format!(
            "giacomini_white_test: n = {n} must exceed both horizon = {horizon} and k = {k}"
        )));
    }
    let d: Vec<f64> = (0..n)
        .map(|t| loss.evaluate(y[t], f1[t]) - loss.evaluate(y[t], f2[t]))
        .collect();
    // Z_t = h_t * d_t (element-wise scaling of the regressor row by the loss diff).
    let mut z = Vec::with_capacity(n);
    for t in 0..n {
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            row.push(test_regressors[[t, j]] * d[t]);
        }
        z.push(row);
    }
    // Sample mean Z̄.
    let mut z_bar = vec![0.0_f64; k];
    for row in &z {
        for j in 0..k {
            z_bar[j] += row[j];
        }
    }
    for v in z_bar.iter_mut() {
        *v /= n as f64;
    }
    // Newey-West HAC variance of Z_t with Bartlett kernel at lag horizon - 1.
    let l_star = horizon.saturating_sub(1);
    let mut v = vec![vec![0.0_f64; k]; k];
    // Gamma_0
    for row in &z {
        for i in 0..k {
            for j in 0..k {
                v[i][j] += (row[i] - z_bar[i]) * (row[j] - z_bar[j]);
            }
        }
    }
    for i in 0..k {
        for j in 0..k {
            v[i][j] /= n as f64;
        }
    }
    // Bartlett-weighted lagged cross-terms.
    for lag in 1..=l_star {
        let mut g = vec![vec![0.0_f64; k]; k];
        for t in lag..n {
            for i in 0..k {
                for j in 0..k {
                    g[i][j] += (z[t][i] - z_bar[i]) * (z[t - lag][j] - z_bar[j]);
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                g[i][j] /= n as f64;
            }
        }
        let bartlett = 1.0 - (lag as f64) / ((l_star + 1) as f64);
        for i in 0..k {
            for j in 0..k {
                v[i][j] += bartlett * (g[i][j] + g[j][i]);
            }
        }
    }
    // Fast-path: if z_bar and the HAC diagonal are all essentially zero, the
    // two forecasts have identical losses on the sample. The null is trivially
    // not rejected — return statistic 0 rather than trying to invert a zero
    // matrix.
    let z_bar_scale = z_bar.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let v_diag_scale = (0..k).map(|i| v[i][i].abs()).fold(0.0_f64, f64::max);
    if z_bar_scale <= 1e-300 && v_diag_scale <= 1e-300 {
        return Ok(GiacominiWhiteResult {
            statistic: 0.0,
            p_value: 1.0,
            df: k,
            n,
        });
    }
    // Wald: n * z_bar' * V^{-1} * z_bar.
    let v_inv = invert_symmetric(&v)?;
    let mut wz = vec![0.0_f64; k];
    for i in 0..k {
        for j in 0..k {
            wz[i] += v_inv[i][j] * z_bar[j];
        }
    }
    let mut wald = 0.0_f64;
    for i in 0..k {
        wald += z_bar[i] * wz[i];
    }
    let statistic = (n as f64) * wald;
    let p_value = chi2_sf(statistic, k as f64);
    Ok(GiacominiWhiteResult {
        statistic,
        p_value,
        df: k,
        n,
    })
}

// ---------------------------------------------------------------------------
// Small linear-algebra + chi-square SF helpers used by GW
// ---------------------------------------------------------------------------

/// Gauss-Jordan inversion of a small symmetric positive-definite matrix
/// (up to a modest `k`). Falls back to an error when the pivot vanishes,
/// which happens on a rank-deficient conditioning set.
fn invert_symmetric(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let k = m.len();
    let mut a: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row = Vec::with_capacity(2 * k);
            row.extend_from_slice(&m[i]);
            for j in 0..k {
                row.push(if i == j { 1.0 } else { 0.0 });
            }
            row
        })
        .collect();
    for i in 0..k {
        // Partial pivot on |a[.][i]|.
        let mut pivot = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..k {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Err(Error::Value(
                "giacomini_white_test: HAC covariance matrix is singular".into(),
            ));
        }
        if pivot != i {
            a.swap(i, pivot);
        }
        let piv = a[i][i];
        for c in 0..(2 * k) {
            a[i][c] /= piv;
        }
        for r in 0..k {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            if factor == 0.0 {
                continue;
            }
            for c in 0..(2 * k) {
                a[r][c] -= factor * a[i][c];
            }
        }
    }
    Ok((0..k).map(|i| a[i][k..(2 * k)].to_vec()).collect())
}

/// Upper-tail χ²(df) survival function `P(X > x)` via the regularised upper
/// incomplete gamma. Accurate to ~1e-12 for the arguments we encounter here.
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    regularized_upper_incomplete_gamma(0.5 * df, 0.5 * x).clamp(0.0, 1.0)
}

fn regularized_upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - regularized_lower_incomplete_gamma_series(a, x)
    } else {
        regularized_upper_incomplete_gamma_cf(a, x)
    }
}

fn regularized_lower_incomplete_gamma_series(a: f64, x: f64) -> f64 {
    const MAXIT: usize = 512;
    const EPS: f64 = 3.0e-16;
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

fn regularized_upper_incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    const MAXIT: usize = 512;
    const EPS: f64 = 3.0e-16;
    const FPMIN: f64 = 1.0e-300;
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
