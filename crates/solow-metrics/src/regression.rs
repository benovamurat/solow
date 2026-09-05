//! Regression metrics — losses and scores for continuous targets.
//!
//! Every function takes ndarray views over `f64` and an optional sample-weight
//! vector. Sample weights (when present) must be non-negative and cannot all be
//! zero; a shape mismatch, a NaN, or an out-of-domain input is returned as an
//! [`Error::Value`] or [`Error::Shape`] rather than silently propagating.
//!
//! The formulas match the canonical the reference definitions so a Solow user
//! can read a the reference example and translate call-for-call.

use ndarray::ArrayView1;
use solow_core::{numeric::NeumaierSum, Error, Result};

/// Compact report bundling the most common regression metrics in one call.
///
/// `RegressionReport::compute(y_true, y_pred, None)` is the standard "how did
/// this model do" summary — mean squared error, RMSE, mean absolute error,
/// median absolute error, R², explained variance, and max error, all computed
/// in a single traversal-friendly call.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionReport {
    /// Mean squared error, `mean((y - ŷ)²)`.
    pub mse: f64,
    /// Root mean squared error, `sqrt(mse)`.
    pub rmse: f64,
    /// Mean absolute error, `mean(|y - ŷ|)`.
    pub mae: f64,
    /// Median absolute error, `median(|y - ŷ|)` (unweighted).
    pub medae: f64,
    /// R² coefficient of determination.
    pub r2: f64,
    /// Explained-variance score, `1 - Var(y - ŷ) / Var(y)`.
    pub explained_variance: f64,
    /// Maximum absolute residual.
    pub max_error: f64,
    /// Number of samples used.
    pub n: usize,
}

impl RegressionReport {
    /// Compute the standard regression report on `y_true` / `y_pred` with an
    /// optional non-negative sample-weight vector.
    pub fn compute(
        y_true: ArrayView1<'_, f64>,
        y_pred: ArrayView1<'_, f64>,
        sample_weight: Option<ArrayView1<'_, f64>>,
    ) -> Result<Self> {
        check_pair("regression report", y_true, y_pred)?;
        if let Some(w) = sample_weight {
            check_weights("regression report", w, y_true.len())?;
        }
        Ok(RegressionReport {
            mse: mean_squared_error(y_true, y_pred, sample_weight)?,
            rmse: root_mean_squared_error(y_true, y_pred, sample_weight)?,
            mae: mean_absolute_error(y_true, y_pred, sample_weight)?,
            medae: median_absolute_error(y_true, y_pred)?,
            r2: r2_score(y_true, y_pred, sample_weight)?,
            explained_variance: explained_variance_score(y_true, y_pred, sample_weight)?,
            max_error: max_error(y_true, y_pred)?,
            n: y_true.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// Shared checks
// ---------------------------------------------------------------------------

fn check_pair(name: &str, y_true: ArrayView1<'_, f64>, y_pred: ArrayView1<'_, f64>) -> Result<()> {
    if y_true.len() != y_pred.len() {
        return Err(Error::Shape(format!(
            "{name}: y_true has {} entries but y_pred has {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    if y_true.is_empty() {
        return Err(Error::Value(format!(
            "{name}: at least one sample is required"
        )));
    }
    for &v in y_true.iter().chain(y_pred.iter()) {
        if !v.is_finite() {
            return Err(Error::Value(format!(
                "{name}: inputs must be finite (found NaN or infinity)"
            )));
        }
    }
    Ok(())
}

fn check_weights(name: &str, w: ArrayView1<'_, f64>, n: usize) -> Result<()> {
    if w.len() != n {
        return Err(Error::Shape(format!(
            "{name}: sample_weight has {} entries but y_true has {n}",
            w.len()
        )));
    }
    let mut total = 0.0_f64;
    for &wi in w.iter() {
        if !wi.is_finite() {
            return Err(Error::Value(format!(
                "{name}: sample_weight must be finite"
            )));
        }
        if wi < 0.0 {
            return Err(Error::Value(format!(
                "{name}: sample_weight must be non-negative (got {wi})"
            )));
        }
        total += wi;
    }
    if total <= 0.0 {
        return Err(Error::Value(format!(
            "{name}: sample_weight must sum to a positive value"
        )));
    }
    Ok(())
}

fn weighted_mean(x: ArrayView1<'_, f64>, sample_weight: Option<ArrayView1<'_, f64>>) -> f64 {
    match sample_weight {
        None => {
            let mut acc = NeumaierSum::new();
            for &v in x.iter() {
                acc.add(v);
            }
            acc.finish() / x.len() as f64
        }
        Some(w) => {
            let mut num = NeumaierSum::new();
            let mut den = NeumaierSum::new();
            for (&xi, &wi) in x.iter().zip(w.iter()) {
                num.add(wi * xi);
                den.add(wi);
            }
            num.finish() / den.finish()
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Mean squared error `E[(y - ŷ)²]`.
pub fn mean_squared_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("mean_squared_error", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("mean_squared_error", w, y_true.len())?;
    }
    match sample_weight {
        None => {
            let mut num = NeumaierSum::new();
            for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
                let d = yt - yp;
                num.add(d * d);
            }
            Ok(num.finish() / y_true.len() as f64)
        }
        Some(w) => {
            let mut num = NeumaierSum::new();
            let mut den = NeumaierSum::new();
            for ((&yt, &yp), &wi) in y_true.iter().zip(y_pred.iter()).zip(w.iter()) {
                let d = yt - yp;
                num.add(wi * d * d);
                den.add(wi);
            }
            Ok(num.finish() / den.finish())
        }
    }
}

/// Root mean squared error, `sqrt(mean_squared_error)`.
pub fn root_mean_squared_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    Ok(mean_squared_error(y_true, y_pred, sample_weight)?.sqrt())
}

/// Mean absolute error `E[|y - ŷ|]`.
pub fn mean_absolute_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("mean_absolute_error", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("mean_absolute_error", w, y_true.len())?;
    }
    match sample_weight {
        None => {
            let mut acc = NeumaierSum::new();
            for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
                acc.add((yt - yp).abs());
            }
            Ok(acc.finish() / y_true.len() as f64)
        }
        Some(w) => {
            let mut num = NeumaierSum::new();
            let mut den = NeumaierSum::new();
            for ((&yt, &yp), &wi) in y_true.iter().zip(y_pred.iter()).zip(w.iter()) {
                num.add(wi * (yt - yp).abs());
                den.add(wi);
            }
            Ok(num.finish() / den.finish())
        }
    }
}

/// Median of the absolute residuals (unweighted; the classical robust score).
pub fn median_absolute_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
) -> Result<f64> {
    check_pair("median_absolute_error", y_true, y_pred)?;
    let mut abs: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&a, &b)| (a - b).abs())
        .collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = abs.len();
    Ok(if n % 2 == 0 {
        0.5 * (abs[n / 2 - 1] + abs[n / 2])
    } else {
        abs[n / 2]
    })
}

/// Largest absolute residual.
pub fn max_error(y_true: ArrayView1<'_, f64>, y_pred: ArrayView1<'_, f64>) -> Result<f64> {
    check_pair("max_error", y_true, y_pred)?;
    Ok(y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max))
}

/// Mean absolute percentage error, `mean(|y - ŷ| / max(|y|, ε))`.
///
/// The denominator uses `f64::EPSILON` as the floor exactly as the reference
/// does, so a zero target does not produce a division-by-zero — instead the
/// residual is divided by machine epsilon and the metric can blow up. That is
/// the intended behaviour when the target legitimately contains zero; use
/// [`symmetric_mean_absolute_percentage_error`] instead when this is not what
/// you want.
pub fn mean_absolute_percentage_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("mean_absolute_percentage_error", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("mean_absolute_percentage_error", w, y_true.len())?;
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    match sample_weight {
        None => {
            for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
                let r = (yt - yp).abs() / yt.abs().max(f64::EPSILON);
                num += r;
                den += 1.0;
            }
        }
        Some(w) => {
            for ((&yt, &yp), &wi) in y_true.iter().zip(y_pred.iter()).zip(w.iter()) {
                let r = (yt - yp).abs() / yt.abs().max(f64::EPSILON);
                num += wi * r;
                den += wi;
            }
        }
    }
    Ok(num / den)
}

/// Symmetric MAPE, `mean(2·|y - ŷ| / (|y| + |ŷ|))`, in `[0, 2]`.
///
/// Returns `Error::Value` if any pair has `|y| + |ŷ| == 0`, which is the only
/// input a symmetric MAPE is undefined on.
pub fn symmetric_mean_absolute_percentage_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("symmetric_mean_absolute_percentage_error", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("symmetric_mean_absolute_percentage_error", w, y_true.len())?;
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    let iter = y_true.iter().zip(y_pred.iter()).enumerate();
    for (i, (&yt, &yp)) in iter {
        let d = yt.abs() + yp.abs();
        if d == 0.0 {
            return Err(Error::Value(format!(
                "symmetric_mean_absolute_percentage_error: sample {i} has y_true = y_pred = 0",
            )));
        }
        let contrib = 2.0 * (yt - yp).abs() / d;
        let wi = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num += wi * contrib;
        den += wi;
    }
    Ok(num / den)
}

/// Mean squared logarithmic error, `mean((log(1+y) - log(1+ŷ))²)`.
///
/// Both `y_true` and `y_pred` must be `≥ 0`.
pub fn mean_squared_log_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("mean_squared_log_error", y_true, y_pred)?;
    for &v in y_true.iter().chain(y_pred.iter()) {
        if v < 0.0 {
            return Err(Error::Value(
                "mean_squared_log_error requires y_true, y_pred ≥ 0".into(),
            ));
        }
    }
    let logs_true: Vec<f64> = y_true.iter().map(|v| (1.0 + v).ln()).collect();
    let logs_pred: Vec<f64> = y_pred.iter().map(|v| (1.0 + v).ln()).collect();
    mean_squared_error(
        ArrayView1::from(logs_true.as_slice()),
        ArrayView1::from(logs_pred.as_slice()),
        sample_weight,
    )
}

/// Root mean squared logarithmic error, `sqrt(mean_squared_log_error)`.
pub fn root_mean_squared_log_error(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    Ok(mean_squared_log_error(y_true, y_pred, sample_weight)?.sqrt())
}

/// Mean pinball loss for a quantile prediction at level `alpha ∈ (0, 1)`.
///
/// For a target `y` and forecast `ŷ` at level `α`,
/// `L_α(y, ŷ) = α·(y - ŷ)⁺ + (1 - α)·(ŷ - y)⁺`.
/// This is the loss quantile regression minimises.
pub fn mean_pinball_loss(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    alpha: f64,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("mean_pinball_loss", y_true, y_pred)?;
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(Error::Value(format!(
            "mean_pinball_loss: alpha must be in (0, 1), got {alpha}"
        )));
    }
    if let Some(w) = sample_weight {
        check_weights("mean_pinball_loss", w, y_true.len())?;
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (i, (&yt, &yp)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let diff = yt - yp;
        let l = if diff >= 0.0 {
            alpha * diff
        } else {
            (alpha - 1.0) * diff
        };
        let wi = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num += wi * l;
        den += wi;
    }
    Ok(num / den)
}

/// Coefficient of determination R² = 1 - SS_res / SS_tot.
///
/// Returns `Error::Value` when SS_tot = 0 (a constant target); a constant
/// target has no variance to explain, so R² is genuinely undefined there.
pub fn r2_score(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("r2_score", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("r2_score", w, y_true.len())?;
    }
    let mean = weighted_mean(y_true, sample_weight);
    let mut ss_res = NeumaierSum::new();
    let mut ss_tot = NeumaierSum::new();
    match sample_weight {
        None => {
            for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
                let d = yt - yp;
                ss_res.add(d * d);
                let dm = yt - mean;
                ss_tot.add(dm * dm);
            }
        }
        Some(w) => {
            for ((&yt, &yp), &wi) in y_true.iter().zip(y_pred.iter()).zip(w.iter()) {
                let d = yt - yp;
                ss_res.add(wi * d * d);
                let dm = yt - mean;
                ss_tot.add(wi * dm * dm);
            }
        }
    }
    let ss_tot_v = ss_tot.finish();
    if ss_tot_v == 0.0 {
        return Err(Error::Value(
            "r2_score is undefined for a constant target (SS_tot = 0)".into(),
        ));
    }
    Ok(1.0 - ss_res.finish() / ss_tot_v)
}

/// Explained-variance score, `1 - Var(y - ŷ) / Var(y)`.
///
/// This coincides with R² whenever the residuals have zero mean (which is
/// automatic for a fit that includes an intercept). It stays sane when a
/// systematic bias moves the residual mean off zero.
pub fn explained_variance_score(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("explained_variance_score", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("explained_variance_score", w, y_true.len())?;
    }
    let mean_y = weighted_mean(y_true, sample_weight);
    let diff: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let mean_diff = weighted_mean(ArrayView1::from(diff.as_slice()), sample_weight);
    let mut num = NeumaierSum::new();
    let mut den = NeumaierSum::new();
    for i in 0..y_true.len() {
        let wi = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        let dd = diff[i] - mean_diff;
        num.add(wi * dd * dd);
        let dy = y_true[i] - mean_y;
        den.add(wi * dy * dy);
    }
    let (num, den) = (num.finish(), den.finish());
    if den == 0.0 {
        return Err(Error::Value(
            "explained_variance_score is undefined for a constant target".into(),
        ));
    }
    Ok(1.0 - num / den)
}

/// D² score with an absolute-error deviance,
/// `1 - MAE(y, ŷ) / MAE(y, median(y))`.
///
/// This is the natural analogue of R² for models fit by quantile regression at
/// the median.
pub fn d2_absolute_error_score(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("d2_absolute_error_score", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("d2_absolute_error_score", w, y_true.len())?;
    }
    let numerator = mean_absolute_error(y_true, y_pred, sample_weight)?;
    let mut sorted: Vec<f64> = y_true.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 0 {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    } else {
        sorted[n / 2]
    };
    let median_vec = vec![median; n];
    let denom = mean_absolute_error(
        y_true,
        ArrayView1::from(median_vec.as_slice()),
        sample_weight,
    )?;
    if denom == 0.0 {
        return Err(Error::Value(
            "d2_absolute_error_score is undefined when the target has zero absolute deviation from its median".into(),
        ));
    }
    Ok(1.0 - numerator / denom)
}

// ---------------------------------------------------------------------------
// Tweedie / Poisson / Gamma deviance
// ---------------------------------------------------------------------------

/// Mean Tweedie deviance at power `p`.
///
/// The Tweedie family unifies several exponential-dispersion deviances by a
/// single power parameter:
///
/// * `p == 0` — Gaussian, `mean((y - ŷ)²)`. Only `p = 0` allows `y_pred = 0`.
/// * `0 < p < 1` — invalid (no Tweedie distribution).
/// * `p == 1` — Poisson deviance, `2·mean(y·log(y/ŷ) - (y - ŷ))`,
///   requires `y ≥ 0` and `ŷ > 0`. `y·log(y/ŷ)` is defined by continuity as 0
///   when `y = 0`.
/// * `1 < p < 2` — compound Poisson-Gamma, `y ≥ 0`, `ŷ > 0`.
/// * `p == 2` — Gamma deviance, `2·mean(log(ŷ/y) + y/ŷ - 1)`, requires
///   `y > 0` and `ŷ > 0`.
/// * `p > 2` — positive stable, `y > 0`, `ŷ > 0`.
///
/// This is the natural log-likelihood-based error for models fit by IRLS in
/// [`solow_glm`]: a Poisson GLM is scored at `p = 1`, a Gamma GLM at `p = 2`.
pub fn mean_tweedie_deviance(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    power: f64,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("mean_tweedie_deviance", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("mean_tweedie_deviance", w, y_true.len())?;
    }
    if (0.0..1.0).contains(&power) && power != 0.0 {
        return Err(Error::Value(format!(
            "mean_tweedie_deviance: power {power} in (0, 1) does not correspond to a distribution"
        )));
    }

    // Per-power domain checks.
    let (needs_pos_y, needs_pos_pred) = if power == 0.0 {
        (false, false)
    } else if power < 1.0 {
        // negative power: y ≥ 0 required, ŷ > 0.
        (false, true)
    } else if power == 1.0 {
        (false, true)
    } else if power < 2.0 {
        (false, true)
    } else {
        (true, true)
    };

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, (&y, &m)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        if y < 0.0 {
            return Err(Error::Value(format!(
                "mean_tweedie_deviance: y_true[{i}] = {y} must be non-negative"
            )));
        }
        if needs_pos_y && y == 0.0 {
            return Err(Error::Value(format!(
                "mean_tweedie_deviance: y_true[{i}] must be strictly positive for power {power}"
            )));
        }
        if needs_pos_pred && m <= 0.0 {
            return Err(Error::Value(format!(
                "mean_tweedie_deviance: y_pred[{i}] = {m} must be strictly positive for power {power}"
            )));
        }
        let d = if power == 0.0 {
            let e = y - m;
            e * e
        } else if power == 1.0 {
            // 2·(y·log(y/ŷ) - (y - ŷ)), with the y·log y term = 0 at y = 0.
            let a = if y > 0.0 { y * (y / m).ln() } else { 0.0 };
            2.0 * (a - (y - m))
        } else if power == 2.0 {
            2.0 * ((m / y).ln() + y / m - 1.0)
        } else {
            let a = if y > 0.0 {
                y.powf(2.0 - power) / ((1.0 - power) * (2.0 - power))
            } else {
                0.0
            };
            let b = y * m.powf(1.0 - power) / (1.0 - power);
            let c = m.powf(2.0 - power) / (2.0 - power);
            2.0 * (a - b + c)
        };
        let w = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num += w * d;
        den += w;
    }
    Ok(num / den)
}

/// Mean Poisson deviance (Tweedie deviance at `power = 1`).
pub fn mean_poisson_deviance(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    mean_tweedie_deviance(y_true, y_pred, 1.0, sample_weight)
}

/// Mean Gamma deviance (Tweedie deviance at `power = 2`).
pub fn mean_gamma_deviance(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    mean_tweedie_deviance(y_true, y_pred, 2.0, sample_weight)
}

/// D² score based on the Tweedie deviance:
/// `1 - D(y, ŷ) / D(y, ȳ)` (or `D(y, median(y))` for a heavy-tailed target).
///
/// Reduces to `r2_score` at `power = 0`. This is the recommended goodness-of-
/// fit score for a fitted Poisson or Gamma GLM.
pub fn d2_tweedie_score(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    power: f64,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("d2_tweedie_score", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("d2_tweedie_score", w, y_true.len())?;
    }
    let num = mean_tweedie_deviance(y_true, y_pred, power, sample_weight)?;
    let mean = weighted_mean(y_true, sample_weight);
    // The Poisson/Gamma reference is the constant y_pred = ȳ.
    let mean_vec = vec![mean; y_true.len()];
    let baseline = mean_tweedie_deviance(
        y_true,
        ArrayView1::from(mean_vec.as_slice()),
        power,
        sample_weight,
    )?;
    if baseline == 0.0 {
        return Err(Error::Value(
            "d2_tweedie_score is undefined when the baseline (constant-mean) deviance is zero"
                .into(),
        ));
    }
    Ok(1.0 - num / baseline)
}

// ---------------------------------------------------------------------------
// Robust regression losses
// ---------------------------------------------------------------------------

/// Huber loss `mean(ρ_δ(y - ŷ))` where
/// `ρ_δ(r) = 0.5 · r²` for `|r| ≤ δ`, else `δ · (|r| - 0.5 · δ)`.
///
/// The quadratic-linear hybrid used in robust regression: quadratic near
/// zero (so it behaves like MSE for typical residuals) and linear in the
/// tails (so a handful of large residuals cannot dominate the gradient).
/// `delta` must be strictly positive.
pub fn huber_loss(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    delta: f64,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("huber_loss", y_true, y_pred)?;
    if !(delta > 0.0 && delta.is_finite()) {
        return Err(Error::Value(format!(
            "huber_loss: delta must be finite and > 0 (got {delta})"
        )));
    }
    if let Some(w) = sample_weight {
        check_weights("huber_loss", w, y_true.len())?;
    }
    let mut num = NeumaierSum::new();
    let mut den = NeumaierSum::new();
    for i in 0..y_true.len() {
        let r = (y_true[i] - y_pred[i]).abs();
        let rho = if r <= delta {
            0.5 * r * r
        } else {
            delta * (r - 0.5 * delta)
        };
        let wi = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num.add(wi * rho);
        den.add(wi);
    }
    Ok(num.finish() / den.finish())
}

/// Log-cosh loss `mean(log(cosh(y - ŷ)))`.
///
/// A smooth, twice-differentiable analogue of the absolute error that
/// asymptotes to `|r| - log 2` for large residuals and reduces to
/// `0.5 · r²` for small ones — often the default loss for regression
/// networks because its gradient never blows up.
///
/// Uses the numerically stable form `log(cosh(r)) = |r| + log(1 + e^{-2|r|}) - log 2`,
/// which avoids overflow for large residuals.
pub fn log_cosh_loss(
    y_true: ArrayView1<'_, f64>,
    y_pred: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    check_pair("log_cosh_loss", y_true, y_pred)?;
    if let Some(w) = sample_weight {
        check_weights("log_cosh_loss", w, y_true.len())?;
    }
    let ln2 = std::f64::consts::LN_2;
    let mut num = NeumaierSum::new();
    let mut den = NeumaierSum::new();
    for i in 0..y_true.len() {
        let r = (y_true[i] - y_pred[i]).abs();
        // log(cosh(r)) = r + log((1 + e^{-2r}) / 2) = r + ln(1 + e^{-2r}) - ln 2.
        let rho = r + (1.0 + (-2.0 * r).exp()).ln() - ln2;
        let wi = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num.add(wi * rho);
        den.add(wi);
    }
    Ok(num.finish() / den.finish())
}
