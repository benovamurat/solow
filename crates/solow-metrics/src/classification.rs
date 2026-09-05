//! Classification metrics — confusion matrix, precision/recall/Fβ, log-loss,
//! Brier score, ROC-AUC and average precision.
//!
//! The averaging semantics ([`Average`]) follow the canonical the reference
//! definitions:
//!
//! * `Binary` — use only the positive class (label `1`). Only valid for
//!   two-class inputs.
//! * `Macro` — the unweighted mean of the per-class scores.
//! * `Weighted` — the mean of the per-class scores weighted by their
//!   support in `y_true`.
//! * `Micro` — the metric applied to the pooled true / predicted totals
//!   across every class. For precision, recall, and F1 this reduces to the
//!   overall accuracy on multiclass data.
//!
//! Labels are `usize` class indices in `[0, num_classes)`. Callers that work
//! with string labels should map them to indices once.

use ndarray::{Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Averaging strategy for multiclass precision / recall / Fβ.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Average {
    /// Report only the score for the positive class (label `1`). Requires a
    /// two-class input.
    Binary,
    /// Unweighted mean of the per-class scores.
    Macro,
    /// Per-class scores weighted by the true support of each class.
    Weighted,
    /// The pooled score across classes (`precision_micro = recall_micro =
    /// accuracy` in the multiclass, single-label case).
    Micro,
}

/// Weighting for [`cohen_kappa_score`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KappaWeights {
    /// Standard (unweighted) κ.
    None,
    /// Linear weighting: penalty grows as `|i - j|`.
    Linear,
    /// Quadratic weighting: penalty grows as `(i - j)²`.
    Quadratic,
}

/// Precision / recall / Fβ / support, either per-class or aggregated.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionRecallFScore {
    /// Per-class precision if [`Average::Micro`] / [`Average::Macro`] /
    /// [`Average::Weighted`] returns one number per class, otherwise a single
    /// aggregated value.
    pub precision: Vec<f64>,
    /// See [`Self::precision`].
    pub recall: Vec<f64>,
    /// See [`Self::precision`].
    pub fbeta: Vec<f64>,
    /// True-class support (count of `y_true == k` per class).
    pub support: Vec<usize>,
}

/// ROC curve — false-positive rate, true-positive rate, and the score
/// thresholds that produced them.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RocCurve {
    /// Ascending false-positive rate.
    pub fpr: Vec<f64>,
    /// Ascending true-positive rate.
    pub tpr: Vec<f64>,
    /// The thresholds each point corresponds to (in descending order — the
    /// first is `+∞` in the reference sense).
    pub thresholds: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

fn check_labels(name: &str, y_true: &[usize], y_pred: &[usize]) -> Result<()> {
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
    Ok(())
}

fn infer_num_classes(y_true: &[usize], y_pred: &[usize], hint: Option<usize>) -> usize {
    let mut m = 0usize;
    for &v in y_true.iter().chain(y_pred.iter()) {
        if v + 1 > m {
            m = v + 1;
        }
    }
    match hint {
        Some(h) => h.max(m),
        None => m,
    }
}

fn check_weights(name: &str, w: Option<&[f64]>, n: usize) -> Result<()> {
    let Some(w) = w else { return Ok(()) };
    if w.len() != n {
        return Err(Error::Shape(format!(
            "{name}: sample_weight has {} entries but y_true has {n}",
            w.len()
        )));
    }
    let mut total = 0.0_f64;
    for &wi in w.iter() {
        if !wi.is_finite() || wi < 0.0 {
            return Err(Error::Value(format!(
                "{name}: sample_weight must be finite and non-negative"
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

// ---------------------------------------------------------------------------
// Confusion matrix
// ---------------------------------------------------------------------------

/// The `k×k` confusion matrix `C[i, j] = #{y_true == i ∧ y_pred == j}`.
///
/// `num_classes` is inferred from the inputs when `None`; pass an explicit
/// value to include classes that never appear in the sample.
pub fn confusion_matrix(
    y_true: &[usize],
    y_pred: &[usize],
    num_classes: Option<usize>,
) -> Result<Array2<usize>> {
    check_labels("confusion_matrix", y_true, y_pred)?;
    let k = infer_num_classes(y_true, y_pred, num_classes).max(1);
    if let Some(h) = num_classes {
        for &v in y_true.iter().chain(y_pred.iter()) {
            if v >= h {
                return Err(Error::Value(format!(
                    "confusion_matrix: label {v} is out of range for num_classes = {h}"
                )));
            }
        }
    }
    let mut c = Array2::<usize>::zeros((k, k));
    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        c[[yt, yp]] += 1;
    }
    Ok(c)
}

// ---------------------------------------------------------------------------
// Accuracy family
// ---------------------------------------------------------------------------

/// Weighted or unweighted accuracy (fraction of correct predictions, or the
/// raw count when `normalize = false`).
pub fn accuracy_score(
    y_true: &[usize],
    y_pred: &[usize],
    sample_weight: Option<&[f64]>,
    normalize: bool,
) -> Result<f64> {
    check_labels("accuracy_score", y_true, y_pred)?;
    check_weights("accuracy_score", sample_weight, y_true.len())?;
    let (mut hit, mut total) = (0.0_f64, 0.0_f64);
    for (i, (&yt, &yp)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let w = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        if yt == yp {
            hit += w;
        }
        total += w;
    }
    Ok(if normalize { hit / total } else { hit })
}

/// Zero-one loss, `1 - accuracy_score(...)`.
pub fn zero_one_loss(
    y_true: &[usize],
    y_pred: &[usize],
    sample_weight: Option<&[f64]>,
    normalize: bool,
) -> Result<f64> {
    let acc = accuracy_score(y_true, y_pred, sample_weight, normalize)?;
    Ok(if normalize {
        1.0 - acc
    } else {
        let total: f64 = match sample_weight {
            Some(w) => w.iter().sum(),
            None => y_true.len() as f64,
        };
        total - acc
    })
}

/// Balanced accuracy — the mean of per-class recall.
///
/// Works on both binary and multiclass problems and reduces to the arithmetic
/// mean of sensitivity and specificity in the binary case.
pub fn balanced_accuracy_score(y_true: &[usize], y_pred: &[usize]) -> Result<f64> {
    let cm = confusion_matrix(y_true, y_pred, None)?;
    let k = cm.nrows();
    let mut sum = 0.0_f64;
    let mut counted = 0usize;
    for i in 0..k {
        let row: usize = cm.row(i).iter().sum();
        if row == 0 {
            continue;
        }
        sum += cm[[i, i]] as f64 / row as f64;
        counted += 1;
    }
    if counted == 0 {
        return Err(Error::Value(
            "balanced_accuracy_score: no observed classes".into(),
        ));
    }
    Ok(sum / counted as f64)
}

// ---------------------------------------------------------------------------
// Precision / recall / Fβ
// ---------------------------------------------------------------------------

/// Precision, recall, Fβ and per-class support in one call.
///
/// `beta` weights recall relative to precision: `beta = 1.0` gives F1, `> 1.0`
/// weights recall more, `< 1.0` weights precision more.
pub fn precision_recall_fscore(
    y_true: &[usize],
    y_pred: &[usize],
    average: Average,
    beta: f64,
    num_classes: Option<usize>,
) -> Result<PrecisionRecallFScore> {
    if !(beta > 0.0 && beta.is_finite()) {
        return Err(Error::Value(format!(
            "precision_recall_fscore: beta must be positive and finite (got {beta})"
        )));
    }
    let cm = confusion_matrix(y_true, y_pred, num_classes)?;
    let k = cm.nrows();
    let beta2 = beta * beta;

    // Per-class TP, predicted-positive column sum, true-positive row sum.
    let mut tp = vec![0.0_f64; k];
    let mut col_sum = vec![0.0_f64; k];
    let mut row_sum = vec![0.0_f64; k];
    for i in 0..k {
        for j in 0..k {
            let v = cm[[i, j]] as f64;
            if i == j {
                tp[i] += v;
            }
            row_sum[i] += v;
            col_sum[j] += v;
        }
    }

    let precision_per: Vec<f64> = (0..k)
        .map(|i| {
            if col_sum[i] > 0.0 {
                tp[i] / col_sum[i]
            } else {
                0.0
            }
        })
        .collect();
    let recall_per: Vec<f64> = (0..k)
        .map(|i| {
            if row_sum[i] > 0.0 {
                tp[i] / row_sum[i]
            } else {
                0.0
            }
        })
        .collect();
    let fbeta_per: Vec<f64> = (0..k)
        .map(|i| {
            let (p, r) = (precision_per[i], recall_per[i]);
            let denom = beta2 * p + r;
            if denom > 0.0 {
                (1.0 + beta2) * p * r / denom
            } else {
                0.0
            }
        })
        .collect();
    let support: Vec<usize> = row_sum.iter().map(|s| *s as usize).collect();

    match average {
        Average::Binary => {
            if k != 2 {
                return Err(Error::Value(format!(
                    "Average::Binary requires exactly two classes (got {k})"
                )));
            }
            Ok(PrecisionRecallFScore {
                precision: vec![precision_per[1]],
                recall: vec![recall_per[1]],
                fbeta: vec![fbeta_per[1]],
                support: vec![support[1]],
            })
        }
        Average::Macro => Ok(PrecisionRecallFScore {
            precision: vec![mean(&precision_per)],
            recall: vec![mean(&recall_per)],
            fbeta: vec![mean(&fbeta_per)],
            support: vec![support.iter().sum()],
        }),
        Average::Weighted => {
            let total: f64 = support.iter().map(|s| *s as f64).sum();
            if total == 0.0 {
                return Err(Error::Value(
                    "Average::Weighted: total support is zero".into(),
                ));
            }
            let p = weighted(&precision_per, &support, total);
            let r = weighted(&recall_per, &support, total);
            let f = weighted(&fbeta_per, &support, total);
            Ok(PrecisionRecallFScore {
                precision: vec![p],
                recall: vec![r],
                fbeta: vec![f],
                support: vec![total as usize],
            })
        }
        Average::Micro => {
            let (tp_sum, col_total, row_total) = (
                tp.iter().sum::<f64>(),
                col_sum.iter().sum::<f64>(),
                row_sum.iter().sum::<f64>(),
            );
            let p = if col_total > 0.0 {
                tp_sum / col_total
            } else {
                0.0
            };
            let r = if row_total > 0.0 {
                tp_sum / row_total
            } else {
                0.0
            };
            let f_denom = beta2 * p + r;
            let f = if f_denom > 0.0 {
                (1.0 + beta2) * p * r / f_denom
            } else {
                0.0
            };
            Ok(PrecisionRecallFScore {
                precision: vec![p],
                recall: vec![r],
                fbeta: vec![f],
                support: vec![row_total as usize],
            })
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn weighted(vs: &[f64], w: &[usize], total: f64) -> f64 {
    vs.iter()
        .zip(w.iter())
        .map(|(v, s)| v * (*s as f64))
        .sum::<f64>()
        / total
}

/// Fβ score (single scalar).
pub fn fbeta_score(
    y_true: &[usize],
    y_pred: &[usize],
    beta: f64,
    average: Average,
    num_classes: Option<usize>,
) -> Result<f64> {
    Ok(precision_recall_fscore(y_true, y_pred, average, beta, num_classes)?.fbeta[0])
}

// ---------------------------------------------------------------------------
// Matthews / Cohen's κ
// ---------------------------------------------------------------------------

/// Matthews correlation coefficient (multiclass generalisation).
///
/// Returns `0.0` when a denominator underflows (i.e. one row or one column of
/// the confusion matrix is empty), matching the reference convention.
pub fn matthews_corrcoef(y_true: &[usize], y_pred: &[usize]) -> Result<f64> {
    let cm = confusion_matrix(y_true, y_pred, None)?;
    let k = cm.nrows();
    let n: f64 = cm.iter().map(|v| *v as f64).sum();
    let t: Vec<f64> = (0..k)
        .map(|i| cm.row(i).iter().map(|v| *v as f64).sum())
        .collect();
    let p: Vec<f64> = (0..k)
        .map(|j| cm.column(j).iter().map(|v| *v as f64).sum())
        .collect();
    let c: f64 = (0..k).map(|i| cm[[i, i]] as f64).sum();
    let s = n;
    let sum_pk_tk: f64 = t.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
    let sum_pk2: f64 = p.iter().map(|x| x * x).sum();
    let sum_tk2: f64 = t.iter().map(|x| x * x).sum();
    let num = c * s - sum_pk_tk;
    let denom = ((s * s - sum_pk2) * (s * s - sum_tk2)).sqrt();
    if denom == 0.0 {
        Ok(0.0)
    } else {
        Ok(num / denom)
    }
}

/// Cohen's κ inter-rater agreement, optionally with linear or quadratic
/// weighting for ordered categories.
pub fn cohen_kappa_score(
    y1: &[usize],
    y2: &[usize],
    weights: KappaWeights,
    num_classes: Option<usize>,
) -> Result<f64> {
    let cm = confusion_matrix(y1, y2, num_classes)?;
    let k = cm.nrows();
    let n: f64 = cm.iter().map(|v| *v as f64).sum();
    let row: Vec<f64> = (0..k)
        .map(|i| cm.row(i).iter().map(|v| *v as f64).sum())
        .collect();
    let col: Vec<f64> = (0..k)
        .map(|j| cm.column(j).iter().map(|v| *v as f64).sum())
        .collect();

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            let w = match weights {
                KappaWeights::None => {
                    if i == j {
                        0.0
                    } else {
                        1.0
                    }
                }
                KappaWeights::Linear => (i as f64 - j as f64).abs(),
                KappaWeights::Quadratic => {
                    let d = i as f64 - j as f64;
                    d * d
                }
            };
            let observed = cm[[i, j]] as f64 / n;
            let expected = row[i] * col[j] / (n * n);
            num += w * observed;
            den += w * expected;
        }
    }
    if den == 0.0 {
        return Err(Error::Value(
            "cohen_kappa_score: expected agreement is zero (single-class marginal)".into(),
        ));
    }
    Ok(1.0 - num / den)
}

// ---------------------------------------------------------------------------
// Probabilistic losses
// ---------------------------------------------------------------------------

/// Binary log-loss `-mean(y·log p + (1 - y)·log(1 - p))` where `y_true` is
/// `bool` and `y_prob` is `P(y = 1)`. Probabilities are clipped to
/// `[eps, 1 - eps]` for numerical stability (`eps = 1e-15` matches
/// the reference default).
pub fn binary_log_loss(
    y_true: &[bool],
    y_prob: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
    eps: f64,
) -> Result<f64> {
    if y_true.len() != y_prob.len() {
        return Err(Error::Shape(format!(
            "binary_log_loss: y_true has {} entries but y_prob has {}",
            y_true.len(),
            y_prob.len()
        )));
    }
    if y_true.is_empty() {
        return Err(Error::Value(
            "binary_log_loss: at least one sample is required".into(),
        ));
    }
    if !(eps.is_finite() && eps > 0.0 && eps < 0.5) {
        return Err(Error::Value(format!(
            "binary_log_loss: eps must be in (0, 0.5) (got {eps})"
        )));
    }
    if let Some(w) = sample_weight {
        if w.len() != y_true.len() {
            return Err(Error::Shape(format!(
                "binary_log_loss: sample_weight has {} entries but y_true has {}",
                w.len(),
                y_true.len()
            )));
        }
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (i, (&yt, &p)) in y_true.iter().zip(y_prob.iter()).enumerate() {
        if !p.is_finite() {
            return Err(Error::Value(
                "binary_log_loss: y_prob must be finite".into(),
            ));
        }
        let p = p.clamp(eps, 1.0 - eps);
        let l = if yt { -p.ln() } else { -(1.0 - p).ln() };
        let w = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num += w * l;
        den += w;
    }
    if den <= 0.0 {
        return Err(Error::Value(
            "binary_log_loss: sample_weight sums to zero".into(),
        ));
    }
    Ok(num / den)
}

/// Multiclass log-loss on a probability matrix (`n × k`).
///
/// Each row must sum to `≈ 1.0` (tolerance `1e-6`); every entry is clipped to
/// `[eps, 1 - eps]` and the log-loss is `-mean(log p[i, y_true[i]])`.
pub fn log_loss(
    y_true: &[usize],
    y_prob: ArrayView2<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
    eps: f64,
) -> Result<f64> {
    let n = y_true.len();
    if y_prob.nrows() != n {
        return Err(Error::Shape(format!(
            "log_loss: y_prob has {} rows but y_true has {n}",
            y_prob.nrows()
        )));
    }
    let k = y_prob.ncols();
    if k == 0 {
        return Err(Error::Shape("log_loss: y_prob has no columns".into()));
    }
    if !(eps.is_finite() && eps > 0.0 && eps < 0.5) {
        return Err(Error::Value(format!(
            "log_loss: eps must be in (0, 0.5) (got {eps})"
        )));
    }
    if let Some(w) = sample_weight {
        if w.len() != n {
            return Err(Error::Shape(format!(
                "log_loss: sample_weight has {} entries but y_true has {n}",
                w.len()
            )));
        }
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (i, row) in y_prob.rows().into_iter().enumerate() {
        let yi = y_true[i];
        if yi >= k {
            return Err(Error::Value(format!(
                "log_loss: label {yi} at row {i} is out of range for k = {k}"
            )));
        }
        let mut row_sum = 0.0_f64;
        for &p in row.iter() {
            if !p.is_finite() || p < 0.0 {
                return Err(Error::Value(format!(
                    "log_loss: y_prob at row {i} must be finite and non-negative"
                )));
            }
            row_sum += p;
        }
        if (row_sum - 1.0).abs() > 1e-6 {
            return Err(Error::Value(format!(
                "log_loss: y_prob row {i} sums to {row_sum}, expected 1"
            )));
        }
        let p_yi = row[yi].clamp(eps, 1.0 - eps);
        let w = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num += w * (-p_yi.ln());
        den += w;
    }
    if den <= 0.0 {
        return Err(Error::Value("log_loss: sample_weight sums to zero".into()));
    }
    Ok(num / den)
}

/// Brier score `mean((y - p)²)` for binary probabilistic classifiers.
pub fn brier_score_loss(
    y_true: &[bool],
    y_prob: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    if y_true.len() != y_prob.len() {
        return Err(Error::Shape(format!(
            "brier_score_loss: y_true has {} entries but y_prob has {}",
            y_true.len(),
            y_prob.len()
        )));
    }
    if let Some(w) = sample_weight {
        if w.len() != y_true.len() {
            return Err(Error::Shape(format!(
                "brier_score_loss: sample_weight has {} entries but y_true has {}",
                w.len(),
                y_true.len()
            )));
        }
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (i, (&yt, &p)) in y_true.iter().zip(y_prob.iter()).enumerate() {
        if !p.is_finite() {
            return Err(Error::Value(
                "brier_score_loss: y_prob must be finite".into(),
            ));
        }
        let y = if yt { 1.0 } else { 0.0 };
        let d = y - p;
        let w = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num += w * d * d;
        den += w;
    }
    if den <= 0.0 {
        return Err(Error::Value(
            "brier_score_loss: sample_weight sums to zero".into(),
        ));
    }
    Ok(num / den)
}

/// Binary hinge loss `mean(max(0, 1 - y·f))` with `y ∈ {-1, +1}` derived
/// from the boolean labels and `f` the (real-valued) decision function.
pub fn hinge_loss(
    y_true: &[bool],
    decision: ArrayView1<'_, f64>,
    sample_weight: Option<ArrayView1<'_, f64>>,
) -> Result<f64> {
    if y_true.len() != decision.len() {
        return Err(Error::Shape(format!(
            "hinge_loss: y_true has {} entries but decision has {}",
            y_true.len(),
            decision.len()
        )));
    }
    if let Some(w) = sample_weight {
        if w.len() != y_true.len() {
            return Err(Error::Shape(format!(
                "hinge_loss: sample_weight has {} entries but y_true has {}",
                w.len(),
                y_true.len()
            )));
        }
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (i, (&yt, &f)) in y_true.iter().zip(decision.iter()).enumerate() {
        if !f.is_finite() {
            return Err(Error::Value("hinge_loss: decision must be finite".into()));
        }
        let y = if yt { 1.0 } else { -1.0 };
        let l = (1.0 - y * f).max(0.0);
        let w = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        num += w * l;
        den += w;
    }
    if den <= 0.0 {
        return Err(Error::Value(
            "hinge_loss: sample_weight sums to zero".into(),
        ));
    }
    Ok(num / den)
}

// ---------------------------------------------------------------------------
// ROC & precision-recall curves
// ---------------------------------------------------------------------------

/// Full binary ROC curve.
///
/// The curve is monotone by construction (`fpr` and `tpr` are ascending) and
/// its endpoints are `(0, 0)` and `(1, 1)`. `y_score` may be a probability, a
/// margin, or any real-valued rank; ties are handled correctly by grouping
/// them into a single threshold.
pub fn roc_curve(y_true: &[bool], y_score: ArrayView1<'_, f64>) -> Result<RocCurve> {
    if y_true.len() != y_score.len() {
        return Err(Error::Shape(format!(
            "roc_curve: y_true has {} entries but y_score has {}",
            y_true.len(),
            y_score.len()
        )));
    }
    let pos: usize = y_true.iter().filter(|&&b| b).count();
    let neg: usize = y_true.len() - pos;
    if pos == 0 || neg == 0 {
        return Err(Error::Value(
            "roc_curve: both classes must be present in y_true".into(),
        ));
    }

    // Sort scores descending, tie-break stably.
    let mut idx: Vec<usize> = (0..y_true.len()).collect();
    idx.sort_by(|&a, &b| y_score[b].partial_cmp(&y_score[a]).unwrap());

    let (mut tp, mut fp) = (0.0_f64, 0.0_f64);
    let mut fpr = Vec::with_capacity(idx.len() + 1);
    let mut tpr = Vec::with_capacity(idx.len() + 1);
    let mut thr = Vec::with_capacity(idx.len() + 1);
    // Prepend the (0, 0) point at threshold +∞ so the curve starts at the
    // origin, matching the reference.
    fpr.push(0.0);
    tpr.push(0.0);
    thr.push(f64::INFINITY);

    let mut i = 0;
    while i < idx.len() {
        // Collect a run of tied scores and update TP / FP once for the whole run.
        let s = y_score[idx[i]];
        let mut j = i;
        while j < idx.len() && y_score[idx[j]] == s {
            if y_true[idx[j]] {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            j += 1;
        }
        fpr.push(fp / neg as f64);
        tpr.push(tp / pos as f64);
        thr.push(s);
        i = j;
    }
    Ok(RocCurve {
        fpr,
        tpr,
        thresholds: thr,
    })
}

/// Area under the ROC curve (binary).
///
/// Computed from [`roc_curve`] with the trapezoidal rule, but the
/// implementation uses the closed-form rank-based expression to avoid a full
/// sort in the common case.
pub fn roc_auc_score(y_true: &[bool], y_score: ArrayView1<'_, f64>) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(Error::Shape(format!(
            "roc_auc_score: y_true has {} entries but y_score has {}",
            y_true.len(),
            y_score.len()
        )));
    }
    let pos = y_true.iter().filter(|&&b| b).count();
    let neg = y_true.len() - pos;
    if pos == 0 || neg == 0 {
        return Err(Error::Value(
            "roc_auc_score: both classes must be present in y_true".into(),
        ));
    }
    // Rank-based Mann-Whitney U estimator.
    let mut pairs: Vec<(f64, bool)> = y_score
        .iter()
        .zip(y_true.iter())
        .map(|(&s, &b)| (s, b))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // Compute mid-ranks to handle ties.
    let n = pairs.len();
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let mid = 0.5 * ((i + 1) as f64 + j as f64);
        for r in ranks.iter_mut().take(j).skip(i) {
            *r = mid;
        }
        i = j;
    }
    let sum_ranks_pos: f64 = ranks
        .iter()
        .zip(pairs.iter())
        .filter_map(|(r, (_, b))| if *b { Some(*r) } else { None })
        .sum();
    let u = sum_ranks_pos - (pos as f64) * (pos as f64 + 1.0) / 2.0;
    Ok(u / (pos as f64 * neg as f64))
}

/// Precision-recall curve.
///
/// Sorted with decreasing threshold; the last returned point is
/// `(precision = 1, recall = 0)` and the first `(precision = TP/(TP+FP),
/// recall = TP/pos)` at the smallest positive threshold, matching
/// the reference convention.
pub fn precision_recall_curve(
    y_true: &[bool],
    y_score: ArrayView1<'_, f64>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if y_true.len() != y_score.len() {
        return Err(Error::Shape(format!(
            "precision_recall_curve: y_true has {} entries but y_score has {}",
            y_true.len(),
            y_score.len()
        )));
    }
    let pos = y_true.iter().filter(|&&b| b).count();
    if pos == 0 {
        return Err(Error::Value(
            "precision_recall_curve: y_true has no positive samples".into(),
        ));
    }

    let mut idx: Vec<usize> = (0..y_true.len()).collect();
    idx.sort_by(|&a, &b| y_score[b].partial_cmp(&y_score[a]).unwrap());
    let (mut tp, mut fp) = (0.0_f64, 0.0_f64);
    let mut prec = Vec::new();
    let mut rec = Vec::new();
    let mut thr = Vec::new();
    let mut i = 0;
    while i < idx.len() {
        let s = y_score[idx[i]];
        let mut j = i;
        while j < idx.len() && y_score[idx[j]] == s {
            if y_true[idx[j]] {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            j += 1;
        }
        let denom = tp + fp;
        prec.push(if denom > 0.0 { tp / denom } else { 1.0 });
        rec.push(tp / pos as f64);
        thr.push(s);
        i = j;
    }
    // Append the (1, 0) endpoint the reference adds.
    prec.push(1.0);
    rec.push(0.0);
    Ok((prec, rec, thr))
}

/// Average precision (a.k.a. AUPRC), computed as the step-function integral
/// `Σ (R_n - R_{n-1}) · P_n` — this matches the reference definition
/// (never the trapezoid).
pub fn average_precision_score(y_true: &[bool], y_score: ArrayView1<'_, f64>) -> Result<f64> {
    let (prec, rec, _) = precision_recall_curve(y_true, y_score)?;
    let mut ap = 0.0_f64;
    // prec / rec are ordered by descending threshold, so recall is increasing;
    // the last (prec = 1, rec = 0) endpoint is skipped in the step integral.
    let mut prev_recall = 0.0_f64;
    for i in 0..prec.len() - 1 {
        let dr = rec[i] - prev_recall;
        ap += dr * prec[i];
        prev_recall = rec[i];
    }
    Ok(ap)
}

/// Top-`k` accuracy for scored multiclass predictors.
///
/// `y_score` is `n × num_classes`; a sample counts as correct if the true
/// label appears among the `k` classes with the largest score.
pub fn top_k_accuracy_score(
    y_true: &[usize],
    y_score: ArrayView2<'_, f64>,
    k: usize,
    normalize: bool,
) -> Result<f64> {
    let n = y_true.len();
    if y_score.nrows() != n {
        return Err(Error::Shape(format!(
            "top_k_accuracy_score: y_score has {} rows but y_true has {n}",
            y_score.nrows()
        )));
    }
    if k == 0 || k > y_score.ncols() {
        return Err(Error::Value(format!(
            "top_k_accuracy_score: k = {k} must be in [1, num_classes = {}]",
            y_score.ncols()
        )));
    }
    let mut hit = 0usize;
    for (i, row) in y_score.rows().into_iter().enumerate() {
        let yi = y_true[i];
        if yi >= y_score.ncols() {
            return Err(Error::Value(format!(
                "top_k_accuracy_score: label {yi} at row {i} is out of range"
            )));
        }
        let true_score = row[yi];
        // Count how many classes have a strictly larger score than the true one.
        let better = row.iter().filter(|&&s| s > true_score).count();
        if better < k {
            hit += 1;
        }
    }
    Ok(if normalize {
        hit as f64 / n as f64
    } else {
        hit as f64
    })
}

// ---------------------------------------------------------------------------
// Multiclass ROC-AUC
// ---------------------------------------------------------------------------

/// Multiclass aggregation strategy for [`roc_auc_ovr_score`] and
/// [`roc_auc_ovo_score`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MulticlassAuc {
    /// Unweighted mean of the per-class (OvR) or per-pair (OvO) AUCs.
    Macro,
    /// Per-class AUCs weighted by the class prevalence in `y_true` (OvR only).
    Weighted,
}

/// One-vs-rest multiclass ROC-AUC.
///
/// For each class `k`, compute the binary AUC treating `y_true == k` as the
/// positive class and using column `k` of `y_score`. The reported number is
/// the macro or prevalence-weighted mean of the `k` per-class AUCs — the
/// the reference `average='macro' | 'weighted'` behaviour.
pub fn roc_auc_ovr_score(
    y_true: &[usize],
    y_score: ArrayView2<'_, f64>,
    average: MulticlassAuc,
) -> Result<f64> {
    let n = y_true.len();
    if y_score.nrows() != n {
        return Err(Error::Shape(format!(
            "roc_auc_ovr_score: y_score has {} rows but y_true has {n}",
            y_score.nrows()
        )));
    }
    let k = y_score.ncols();
    if k < 2 {
        return Err(Error::Value(
            "roc_auc_ovr_score: y_score must have at least two columns".into(),
        ));
    }
    let mut aucs = Vec::with_capacity(k);
    let mut supports = Vec::with_capacity(k);
    for c in 0..k {
        let y_bin: Vec<bool> = y_true.iter().map(|&y| y == c).collect();
        let pos = y_bin.iter().filter(|&&b| b).count();
        if pos == 0 || pos == n {
            // Class either never or always occurs; skip like the reference.
            continue;
        }
        let col = y_score.column(c);
        let auc = roc_auc_score(&y_bin, col)?;
        aucs.push(auc);
        supports.push(pos);
    }
    if aucs.is_empty() {
        return Err(Error::Value(
            "roc_auc_ovr_score: no class had both positive and negative samples".into(),
        ));
    }
    match average {
        MulticlassAuc::Macro => Ok(aucs.iter().sum::<f64>() / aucs.len() as f64),
        MulticlassAuc::Weighted => {
            let total: f64 = supports.iter().map(|&s| s as f64).sum();
            let weighted: f64 = aucs
                .iter()
                .zip(supports.iter())
                .map(|(a, s)| a * (*s as f64))
                .sum();
            Ok(weighted / total)
        }
    }
}

/// One-vs-one multiclass ROC-AUC in the Hand-Till (2001) formulation.
///
/// For every ordered pair `(a, b)` of distinct classes, compute the AUC that
/// ranks `a` above `b` on samples where `y_true ∈ {a, b}` using column `a` of
/// `y_score`, and average the `k·(k - 1)` pairwise AUCs. The `Macro`
/// aggregator gives the classical Hand-Till "M measure"; `Weighted` weights
/// each pair by its joint prevalence (this reproduces the reference
/// `average='weighted', multi_class='ovo'`).
pub fn roc_auc_ovo_score(
    y_true: &[usize],
    y_score: ArrayView2<'_, f64>,
    average: MulticlassAuc,
) -> Result<f64> {
    let n = y_true.len();
    if y_score.nrows() != n {
        return Err(Error::Shape(format!(
            "roc_auc_ovo_score: y_score has {} rows but y_true has {n}",
            y_score.nrows()
        )));
    }
    let k = y_score.ncols();
    if k < 2 {
        return Err(Error::Value(
            "roc_auc_ovo_score: y_score must have at least two columns".into(),
        ));
    }
    // Enumerate distinct classes actually present in y_true.
    let mut classes: Vec<usize> = y_true.iter().copied().collect();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return Err(Error::Value(
            "roc_auc_ovo_score: at least two classes must be present in y_true".into(),
        ));
    }
    let mut pair_aucs = Vec::new();
    let mut pair_weights = Vec::new();
    for i in 0..classes.len() {
        for j in 0..classes.len() {
            if i == j {
                continue;
            }
            let (a, b) = (classes[i], classes[j]);
            // Keep only rows in class a or b.
            let mut y_bin = Vec::new();
            let mut s_a = Vec::new();
            for (idx, &y) in y_true.iter().enumerate() {
                if y == a {
                    y_bin.push(true);
                    s_a.push(y_score[[idx, a]]);
                } else if y == b {
                    y_bin.push(false);
                    s_a.push(y_score[[idx, a]]);
                }
            }
            let n_a = y_bin.iter().filter(|&&x| x).count();
            let n_b = y_bin.len() - n_a;
            if n_a == 0 || n_b == 0 {
                continue;
            }
            let auc = roc_auc_score(&y_bin, ArrayView1::from(&s_a[..]))?;
            pair_aucs.push(auc);
            pair_weights.push((n_a + n_b) as f64);
        }
    }
    if pair_aucs.is_empty() {
        return Err(Error::Value(
            "roc_auc_ovo_score: no class pair had samples in both classes".into(),
        ));
    }
    match average {
        MulticlassAuc::Macro => Ok(pair_aucs.iter().sum::<f64>() / pair_aucs.len() as f64),
        MulticlassAuc::Weighted => {
            let total: f64 = pair_weights.iter().sum();
            let weighted: f64 = pair_aucs
                .iter()
                .zip(pair_weights.iter())
                .map(|(a, w)| a * w)
                .sum();
            Ok(weighted / total)
        }
    }
}

// ---------------------------------------------------------------------------
// Multiclass Brier, Ranked Probability Score, top-label ECE
// ---------------------------------------------------------------------------

fn check_prob_matrix(
    name: &str,
    y_true: &[usize],
    y_score: ArrayView2<'_, f64>,
    require_prob: bool,
) -> Result<()> {
    if y_score.nrows() != y_true.len() {
        return Err(Error::Shape(format!(
            "{name}: y_score has {} rows but y_true has {}",
            y_score.nrows(),
            y_true.len()
        )));
    }
    if y_true.is_empty() {
        return Err(Error::Value(format!(
            "{name}: at least one sample is required"
        )));
    }
    let k = y_score.ncols();
    if k < 2 {
        return Err(Error::Value(format!(
            "{name}: y_score must have at least two columns"
        )));
    }
    for &y in y_true {
        if y >= k {
            return Err(Error::Value(format!(
                "{name}: label {y} is out of range for {k} classes"
            )));
        }
    }
    if require_prob {
        for i in 0..y_score.nrows() {
            let mut s = 0.0_f64;
            for j in 0..k {
                let v = y_score[[i, j]];
                if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                    return Err(Error::Value(format!(
                        "{name}: y_score[{i}, {j}] = {v} must be a probability in [0, 1]"
                    )));
                }
                s += v;
            }
            if (s - 1.0).abs() > 1e-6 {
                return Err(Error::Value(format!(
                    "{name}: row {i} of y_score sums to {s}, not 1"
                )));
            }
        }
    }
    Ok(())
}

/// Multiclass Brier score, `(1/N) Σᵢ Σₖ (I(yᵢ = k) - pᵢₖ)²`.
///
/// The natural generalisation of [`brier_score_loss`] to `k > 2` classes.
/// Bounded in `[0, 2]`; a uniform predictor over `k` classes scores
/// `1 - 1/k`.
pub fn multiclass_brier_score(y_true: &[usize], y_score: ArrayView2<'_, f64>) -> Result<f64> {
    check_prob_matrix("multiclass_brier_score", y_true, y_score, true)?;
    let k = y_score.ncols();
    let n = y_true.len();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for c in 0..k {
            let y = if y_true[i] == c { 1.0 } else { 0.0 };
            let d = y_score[[i, c]] - y;
            sum += d * d;
        }
    }
    Ok(sum / n as f64)
}

/// Ranked Probability Score for an ordered multiclass classifier
/// (Epstein 1969; Murphy 1971).
///
/// Takes the cumulative squared distance between predicted and observed
/// cumulative distribution functions, averaged over samples and rescaled
/// by `1 / (K - 1)`:
///
/// `RPS = (1 / (N · (K - 1))) · Σᵢ Σ_{k=1..K-1} (Σ_{j≤k} pᵢⱼ - I(yᵢ ≤ k))²`.
///
/// A perfect ordered classifier scores 0. Requires class labels to be
/// interpretable as an ordinal scale — use [`multiclass_brier_score`]
/// for nominal ones.
pub fn ranked_probability_score(y_true: &[usize], y_score: ArrayView2<'_, f64>) -> Result<f64> {
    check_prob_matrix("ranked_probability_score", y_true, y_score, true)?;
    let n = y_true.len();
    let k = y_score.ncols();
    let mut total = 0.0_f64;
    for i in 0..n {
        let mut cum_p = 0.0_f64;
        let mut cum_y = 0.0_f64;
        let mut sample = 0.0_f64;
        for c in 0..(k - 1) {
            cum_p += y_score[[i, c]];
            cum_y += if y_true[i] == c { 1.0 } else { 0.0 };
            let d = cum_p - cum_y;
            sample += d * d;
        }
        total += sample;
    }
    Ok(total / (n as f64 * (k as f64 - 1.0)))
}

/// Top-label Expected Calibration Error for a multiclass classifier
/// (Guo-Pleiss-Sun-Weinberger 2017).
///
/// Bins samples by their predicted-class confidence `max_k y_scoreᵢₖ`
/// into `n_bins` uniform bins on `[0, 1]` and returns
/// `Σₖ (nₖ / N) · |mean_confₖ - mean_accₖ|` — the standard "top-1 ECE"
/// number reported alongside multiclass classifier accuracy.
pub fn top_label_calibration_error(
    y_true: &[usize],
    y_score: ArrayView2<'_, f64>,
    n_bins: usize,
) -> Result<f64> {
    check_prob_matrix("top_label_calibration_error", y_true, y_score, false)?;
    if n_bins < 2 {
        return Err(Error::Value(format!(
            "top_label_calibration_error: n_bins must be ≥ 2 (got {n_bins})"
        )));
    }
    let n = y_true.len();
    let k = y_score.ncols();
    let mut sum_conf = vec![0.0_f64; n_bins];
    let mut sum_acc = vec![0.0_f64; n_bins];
    let mut counts = vec![0usize; n_bins];
    for i in 0..n {
        let mut best_c = 0usize;
        let mut best_p = y_score[[i, 0]];
        for c in 1..k {
            let v = y_score[[i, c]];
            if v > best_p {
                best_p = v;
                best_c = c;
            }
        }
        let mut b = (best_p * n_bins as f64).floor() as usize;
        if b >= n_bins {
            b = n_bins - 1;
        }
        sum_conf[b] += best_p;
        sum_acc[b] += if best_c == y_true[i] { 1.0 } else { 0.0 };
        counts[b] += 1;
    }
    let mut ece = 0.0_f64;
    for b in 0..n_bins {
        if counts[b] == 0 {
            continue;
        }
        let mean_c = sum_conf[b] / counts[b] as f64;
        let mean_a = sum_acc[b] / counts[b] as f64;
        ece += (counts[b] as f64 / n as f64) * (mean_c - mean_a).abs();
    }
    Ok(ece)
}

// ---------------------------------------------------------------------------
// Focal loss (binary + multiclass)
// ---------------------------------------------------------------------------

/// Binary focal loss (Lin et al. 2017).
///
/// `FL(pₜ) = -α · (1 - pₜ)^γ · log(pₜ)` where
/// `pₜ = p if y = 1 else 1 - p` and `α` re-weights the positive class
/// (pass `alpha = 0.5` for the unweighted focal loss). The `γ = 0` case
/// reduces to the standard binary cross-entropy up to the α weighting.
///
/// Uses a numerically-safe log-1-minus formulation so probabilities can
/// be arbitrarily close to 0 or 1 without producing NaNs.
pub fn binary_focal_loss(
    y_true: &[bool],
    y_prob: ArrayView1<'_, f64>,
    gamma: f64,
    alpha: f64,
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    if y_true.len() != y_prob.len() {
        return Err(Error::Shape(format!(
            "binary_focal_loss: y_true has {} entries but y_prob has {}",
            y_true.len(),
            y_prob.len()
        )));
    }
    if y_true.is_empty() {
        return Err(Error::Value(
            "binary_focal_loss: at least one sample is required".into(),
        ));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(Error::Value(format!(
            "binary_focal_loss: alpha must be in [0, 1] (got {alpha})"
        )));
    }
    if gamma < 0.0 {
        return Err(Error::Value(format!(
            "binary_focal_loss: gamma must be ≥ 0 (got {gamma})"
        )));
    }
    check_weights("binary_focal_loss", sample_weight, y_true.len())?;
    let mut total = 0.0_f64;
    let mut wsum = 0.0_f64;
    let eps = 1e-15_f64;
    for i in 0..y_true.len() {
        let p = y_prob[i].clamp(eps, 1.0 - eps);
        let (pt, a) = if y_true[i] {
            (p, alpha)
        } else {
            (1.0 - p, 1.0 - alpha)
        };
        let one_minus_pt = 1.0 - pt;
        let modulating = if gamma == 0.0 {
            1.0
        } else {
            one_minus_pt.powf(gamma)
        };
        let loss_i = -a * modulating * pt.ln();
        let wi = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        total += wi * loss_i;
        wsum += wi;
    }
    Ok(total / wsum)
}

/// Multiclass focal loss.
///
/// Per-sample loss `-α · (1 - pᵧ)^γ · log(pᵧ)` where `pᵧ` is the predicted
/// probability for the true class and `α` is a global class re-weighting
/// (pass `alpha = 1.0` for unweighted focal loss). Probabilities are
/// clipped to `[1e-15, 1 - 1e-15]` for the log.
pub fn multiclass_focal_loss(
    y_true: &[usize],
    y_score: ArrayView2<'_, f64>,
    gamma: f64,
    alpha: f64,
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    check_prob_matrix("multiclass_focal_loss", y_true, y_score, false)?;
    if alpha <= 0.0 {
        return Err(Error::Value(format!(
            "multiclass_focal_loss: alpha must be > 0 (got {alpha})"
        )));
    }
    if gamma < 0.0 {
        return Err(Error::Value(format!(
            "multiclass_focal_loss: gamma must be ≥ 0 (got {gamma})"
        )));
    }
    check_weights("multiclass_focal_loss", sample_weight, y_true.len())?;
    let eps = 1e-15_f64;
    let mut total = 0.0_f64;
    let mut wsum = 0.0_f64;
    for i in 0..y_true.len() {
        let py = y_score[[i, y_true[i]]].clamp(eps, 1.0 - eps);
        let one_minus_py = 1.0 - py;
        let modulating = if gamma == 0.0 {
            1.0
        } else {
            one_minus_py.powf(gamma)
        };
        let loss_i = -alpha * modulating * py.ln();
        let wi = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        total += wi * loss_i;
        wsum += wi;
    }
    Ok(total / wsum)
}
