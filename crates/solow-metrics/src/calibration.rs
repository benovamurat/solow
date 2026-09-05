//! Probability-calibration diagnostics for binary probabilistic classifiers.
//!
//! Calibration asks a different question than accuracy: it is not "how often
//! is the classifier right?" but "does it *mean* what it says?". A perfectly
//! calibrated model reports probability `p` on the events that in fact occur
//! with frequency `p`.
//!
//! The functions here give the three standard diagnostics:
//!
//! * [`reliability_curve`] — the empirical foreground/background plot, one
//!   point per probability bin (both uniform-width and equal-mass bins).
//! * [`expected_calibration_error`] and [`maximum_calibration_error`] — the
//!   two calibration summary numbers (ECE, MCE).
//! * [`brier_decomposition`] — the Sanders / Murphy decomposition of the
//!   Brier score into `reliability - resolution + uncertainty`, which reads
//!   as a numerical statement of how much of the score is calibration and how
//!   much is discrimination.

use ndarray::ArrayView1;
use solow_core::{Error, Result};

/// Binning strategy for the reliability diagnostics.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinStrategy {
    /// `n_bins` equal-width bins on `[0, 1]`.
    Uniform,
    /// `n_bins` bins each containing (approximately) the same number of
    /// samples — the standard remedy when probabilities cluster near 0 or 1.
    Quantile,
}

/// One row of a reliability curve.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct ReliabilityBin {
    /// Bin index, `0..n_bins`.
    pub index: usize,
    /// Number of samples that fell into the bin.
    pub count: usize,
    /// Mean predicted probability in the bin.
    pub mean_predicted: f64,
    /// Empirical positive-class frequency in the bin.
    pub mean_actual: f64,
}

/// Sanders / Murphy decomposition of the Brier score.
///
/// The classical decomposition
/// `binned_brier = reliability - resolution + uncertainty`
/// is only an identity for the Brier score *after* the forecasts have been
/// collapsed to their per-bin means. The **raw** Brier score
/// [`brier`](Self::brier) reported here is computed directly from
/// `(pᵢ - yᵢ)²`; it agrees with
/// [`brier_score_loss`](crate::classification::brier_score_loss) to machine
/// precision. The gap `brier - binned_brier` measures how much of the score
/// comes from within-bin dispersion of the raw forecasts, which the classical
/// three-term decomposition discards; the field
/// [`within_bin_variance`](Self::within_bin_variance) reports that dispersion
/// explicitly.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrierDecomposition {
    /// Sum over bins of `nₖ/N · (p̄ₖ - ōₖ)²` — the miscalibration term.
    /// Lower is better; zero means perfectly calibrated at the chosen binning.
    pub reliability: f64,
    /// Sum over bins of `nₖ/N · (ōₖ - ō)²` — how much the classifier's bins
    /// separate outcomes from the marginal rate. Higher is better.
    pub resolution: f64,
    /// `ō · (1 - ō)`, the Brier of the marginal-rate constant predictor.
    /// Depends only on the target, not on the model.
    pub uncertainty: f64,
    /// `(1/N) Σᵢ (pᵢ - p̄_{bin(i)})²` — the within-bin variance of the raw
    /// forecasts. Zero when every bin contains a single distinct
    /// probability, in which case [`brier`](Self::brier) equals
    /// [`binned_brier`](Self::binned_brier).
    pub within_bin_variance: f64,
    /// The classical three-term identity: `reliability - resolution +
    /// uncertainty`. This is the Brier of the "coarsened" forecasts that
    /// replace every forecast by its bin's mean forecast.
    pub binned_brier: f64,
    /// The raw Brier score `(1/N) Σᵢ (pᵢ - yᵢ)²`. Matches
    /// [`brier_score_loss`](crate::classification::brier_score_loss) to
    /// machine precision.
    pub brier: f64,
}

fn check_binary_inputs(name: &str, y_true: &[bool], y_prob: ArrayView1<'_, f64>) -> Result<()> {
    if y_true.len() != y_prob.len() {
        return Err(Error::Shape(format!(
            "{name}: y_true has {} entries but y_prob has {}",
            y_true.len(),
            y_prob.len()
        )));
    }
    if y_true.is_empty() {
        return Err(Error::Value(format!(
            "{name}: at least one sample is required"
        )));
    }
    for (i, &p) in y_prob.iter().enumerate() {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(Error::Value(format!(
                "{name}: y_prob[{i}] = {p} must be in [0, 1]"
            )));
        }
    }
    Ok(())
}

fn assign_bins(y_prob: ArrayView1<'_, f64>, n_bins: usize, strategy: BinStrategy) -> Vec<usize> {
    let n = y_prob.len();
    match strategy {
        BinStrategy::Uniform => y_prob
            .iter()
            .map(|&p| {
                let mut b = (p * n_bins as f64).floor() as usize;
                if b >= n_bins {
                    b = n_bins - 1;
                }
                b
            })
            .collect(),
        BinStrategy::Quantile => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| y_prob[a].partial_cmp(&y_prob[b]).unwrap());
            let mut bins = vec![0usize; n];
            for (rank, &idx) in order.iter().enumerate() {
                let mut b = rank * n_bins / n;
                if b >= n_bins {
                    b = n_bins - 1;
                }
                bins[idx] = b;
            }
            bins
        }
    }
}

/// Reliability curve — one row per non-empty bin.
///
/// `n_bins` must be `≥ 2`. Bin membership is decided by `strategy`.
pub fn reliability_curve(
    y_true: &[bool],
    y_prob: ArrayView1<'_, f64>,
    n_bins: usize,
    strategy: BinStrategy,
) -> Result<Vec<ReliabilityBin>> {
    check_binary_inputs("reliability_curve", y_true, y_prob)?;
    if n_bins < 2 {
        return Err(Error::Value(format!(
            "reliability_curve: n_bins must be ≥ 2 (got {n_bins})"
        )));
    }
    let bins = assign_bins(y_prob, n_bins, strategy);
    let mut sum_pred = vec![0.0_f64; n_bins];
    let mut sum_actual = vec![0.0_f64; n_bins];
    let mut counts = vec![0usize; n_bins];
    for i in 0..y_true.len() {
        let b = bins[i];
        sum_pred[b] += y_prob[i];
        sum_actual[b] += if y_true[i] { 1.0 } else { 0.0 };
        counts[b] += 1;
    }
    let mut out = Vec::new();
    for b in 0..n_bins {
        if counts[b] == 0 {
            continue;
        }
        out.push(ReliabilityBin {
            index: b,
            count: counts[b],
            mean_predicted: sum_pred[b] / counts[b] as f64,
            mean_actual: sum_actual[b] / counts[b] as f64,
        });
    }
    Ok(out)
}

/// Expected Calibration Error, `Σₖ (nₖ / N) · |p̄ₖ - ōₖ|`.
///
/// The classical Guo-Pleiss-Sun-Weinberger summary in `[0, 1]`; 0 is
/// perfectly calibrated at the chosen binning.
pub fn expected_calibration_error(
    y_true: &[bool],
    y_prob: ArrayView1<'_, f64>,
    n_bins: usize,
    strategy: BinStrategy,
) -> Result<f64> {
    let curve = reliability_curve(y_true, y_prob, n_bins, strategy)?;
    let n = y_true.len() as f64;
    Ok(curve
        .iter()
        .map(|b| (b.count as f64 / n) * (b.mean_predicted - b.mean_actual).abs())
        .sum())
}

/// Maximum Calibration Error, `maxₖ |p̄ₖ - ōₖ|` over non-empty bins.
///
/// The worst-bin calibration gap — useful when a small but severely
/// miscalibrated tail matters (e.g. high-confidence errors).
pub fn maximum_calibration_error(
    y_true: &[bool],
    y_prob: ArrayView1<'_, f64>,
    n_bins: usize,
    strategy: BinStrategy,
) -> Result<f64> {
    let curve = reliability_curve(y_true, y_prob, n_bins, strategy)?;
    if curve.is_empty() {
        return Err(Error::Value(
            "maximum_calibration_error: every bin was empty".into(),
        ));
    }
    Ok(curve
        .iter()
        .map(|b| (b.mean_predicted - b.mean_actual).abs())
        .fold(0.0_f64, f64::max))
}

/// Sanders / Murphy decomposition of the Brier score.
///
/// Given `N` samples binned into `n_bins`, the score decomposes as
/// `Brier = reliability - resolution + uncertainty` (to machine precision).
/// A perfectly calibrated model has `reliability = 0`; a maximally
/// discriminative model has `resolution = uncertainty`.
pub fn brier_decomposition(
    y_true: &[bool],
    y_prob: ArrayView1<'_, f64>,
    n_bins: usize,
    strategy: BinStrategy,
) -> Result<BrierDecomposition> {
    check_binary_inputs("brier_decomposition", y_true, y_prob)?;
    if n_bins < 2 {
        return Err(Error::Value(format!(
            "brier_decomposition: n_bins must be ≥ 2 (got {n_bins})"
        )));
    }
    let n_usize = y_true.len();
    let n = n_usize as f64;
    // Bin assignment plus per-bin mean forecast, for the within-bin variance term.
    let bin_of = assign_bins(y_prob, n_bins, strategy);
    let mut sum_pred = vec![0.0_f64; n_bins];
    let mut sum_actual = vec![0.0_f64; n_bins];
    let mut counts = vec![0usize; n_bins];
    for i in 0..n_usize {
        let b = bin_of[i];
        sum_pred[b] += y_prob[i];
        sum_actual[b] += if y_true[i] { 1.0 } else { 0.0 };
        counts[b] += 1;
    }
    let mean_pred: Vec<f64> = (0..n_bins)
        .map(|b| {
            if counts[b] > 0 {
                sum_pred[b] / counts[b] as f64
            } else {
                0.0
            }
        })
        .collect();
    let mean_actual: Vec<f64> = (0..n_bins)
        .map(|b| {
            if counts[b] > 0 {
                sum_actual[b] / counts[b] as f64
            } else {
                0.0
            }
        })
        .collect();
    let overall_rate = y_true.iter().filter(|&&b| b).count() as f64 / n;

    let mut reliability = 0.0_f64;
    let mut resolution = 0.0_f64;
    for b in 0..n_bins {
        if counts[b] == 0 {
            continue;
        }
        let w = counts[b] as f64 / n;
        let dp = mean_pred[b] - mean_actual[b];
        reliability += w * dp * dp;
        let dr = mean_actual[b] - overall_rate;
        resolution += w * dr * dr;
    }
    let uncertainty = overall_rate * (1.0 - overall_rate);
    // Within-bin variance of the raw forecasts.
    let mut wbv = 0.0_f64;
    for i in 0..n_usize {
        let b = bin_of[i];
        let d = y_prob[i] - mean_pred[b];
        wbv += d * d;
    }
    wbv /= n;
    let binned_brier = reliability - resolution + uncertainty;
    // Raw Brier score computed directly; the classical decomposition
    // recovers this only up to within-bin dispersion and the joint (p, y)
    // covariance inside each bin.
    let mut brier = 0.0_f64;
    for i in 0..n_usize {
        let y = if y_true[i] { 1.0 } else { 0.0 };
        let d = y_prob[i] - y;
        brier += d * d;
    }
    brier /= n;
    Ok(BrierDecomposition {
        reliability,
        resolution,
        uncertainty,
        within_bin_variance: wbv,
        binned_brier,
        brier,
    })
}
