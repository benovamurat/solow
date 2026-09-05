//! Distribution-free prediction intervals — conformal prediction.
//!
//! Conformal prediction wraps an already-fit point-prediction model in
//! a procedure that outputs an interval with a **guaranteed** finite-
//! sample coverage rate `≥ 1 - α`, regardless of the model class or
//! how well it happens to fit the data. The only assumption is that
//! the calibration data and the test data are exchangeable (i.i.d. is
//! sufficient).
//!
//! Two flavours ship here:
//!
//! * [`SplitConformal`] — the classical Vovk-Gammerman-Shafer split
//!   conformal predictor. Uses a held-out calibration set to compute
//!   the empirical quantile of the residuals, then adds ± that quantile
//!   to every new prediction. Cheapest; most common in production.
//! * [`JackknifePlus`] — the Barber-Candes-Ramdas-Tibshirani (2021)
//!   jackknife+ predictor. Requires leave-one-out residuals as input
//!   (or the equivalent from a bootstrap) and gives a slightly wider
//!   but still finite-sample-valid interval without a held-out set.
//!
//! Both take **residual magnitudes** rather than a fitted model —
//! the caller stays in control of the underlying regressor and simply
//! hands the interval builder the numbers it needs.

use solow_core::{Error, Result};

/// A prediction interval.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionInterval {
    /// Lower bound of the interval.
    pub low: f64,
    /// Upper bound of the interval.
    pub high: f64,
}

impl PredictionInterval {
    /// Width of the interval `high - low`.
    pub fn width(&self) -> f64 {
        self.high - self.low
    }

    /// Does the interval contain `y`?
    pub fn contains(&self, y: f64) -> bool {
        self.low <= y && y <= self.high
    }
}

// ---------------------------------------------------------------------------
// Split conformal
// ---------------------------------------------------------------------------

/// Split conformal (Vovk-Gammerman-Shafer) prediction interval.
///
/// The interval is `predictionᵢ ± q̂`, where `q̂` is the `⌈(n + 1)(1 - α)⌉`-th
/// smallest of the calibration-set absolute residuals — the classical
/// finite-sample quantile that guarantees marginal coverage `≥ 1 - α`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitConformal {
    /// The conformal quantile used to build every interval.
    pub quantile: f64,
    /// Miscoverage level `α` this quantile was computed at.
    pub alpha: f64,
    /// Number of calibration samples.
    pub n_cal: usize,
}

impl SplitConformal {
    /// Fit a split conformal predictor from the absolute residuals of a
    /// held-out calibration set.
    pub fn fit(residuals: &[f64], alpha: f64) -> Result<Self> {
        if residuals.is_empty() {
            return Err(Error::Value(
                "SplitConformal::fit: residuals must be non-empty".into(),
            ));
        }
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(Error::Value(format!(
                "SplitConformal::fit: alpha must be in (0, 1), got {alpha}"
            )));
        }
        let mut sorted: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        for &r in &sorted {
            if !r.is_finite() {
                return Err(Error::Value(
                    "SplitConformal::fit: residuals must be finite".into(),
                ));
            }
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        // Finite-sample rank: ⌈(n + 1)(1 - α)⌉ — 1-indexed.
        let rank = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let idx = rank.saturating_sub(1).min(n - 1);
        Ok(SplitConformal {
            quantile: sorted[idx],
            alpha,
            n_cal: n,
        })
    }

    /// Turn a point prediction into a conformal prediction interval.
    pub fn interval(&self, prediction: f64) -> PredictionInterval {
        PredictionInterval {
            low: prediction - self.quantile,
            high: prediction + self.quantile,
        }
    }
}

// ---------------------------------------------------------------------------
// Jackknife+
// ---------------------------------------------------------------------------

/// Jackknife+ (Barber-Candes-Ramdas-Tibshirani 2021) prediction interval.
///
/// For each new point, the interval is
/// `[Q_{α/2}(µ̂₋ᵢ(x) - Rᵢ), Q_{1-α/2}(µ̂₋ᵢ(x) + Rᵢ)]`
/// where `Rᵢ` are the leave-one-out absolute residuals and `µ̂₋ᵢ(x)`
/// are the leave-one-out point predictions on `x`. Both are supplied
/// by the caller — the fit and prediction steps live outside this
/// crate. The interval is finite-sample valid at coverage `1 - 2α`
/// under exchangeability.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct JackknifePlus {
    /// Leave-one-out absolute residuals `Rᵢ = |yᵢ - µ̂₋ᵢ(xᵢ)|`.
    pub loo_residuals: Vec<f64>,
    /// Miscoverage level `α`.
    pub alpha: f64,
}

impl JackknifePlus {
    /// Build a jackknife+ predictor from the leave-one-out residuals.
    pub fn new(loo_residuals: Vec<f64>, alpha: f64) -> Result<Self> {
        if loo_residuals.is_empty() {
            return Err(Error::Value(
                "JackknifePlus::new: loo_residuals must be non-empty".into(),
            ));
        }
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(Error::Value(format!(
                "JackknifePlus::new: alpha must be in (0, 1), got {alpha}"
            )));
        }
        for &r in &loo_residuals {
            if !r.is_finite() || r < 0.0 {
                return Err(Error::Value(
                    "JackknifePlus::new: loo_residuals must be finite and ≥ 0".into(),
                ));
            }
        }
        Ok(JackknifePlus {
            loo_residuals,
            alpha,
        })
    }

    /// Build the jackknife+ interval at a new point.
    ///
    /// `loo_predictions` is the vector `µ̂₋ᵢ(x)` — the prediction the
    /// model fit on all but the `i`-th training point would give on
    /// this new `x`. Must have the same length as `loo_residuals`.
    pub fn interval(&self, loo_predictions: &[f64]) -> Result<PredictionInterval> {
        if loo_predictions.len() != self.loo_residuals.len() {
            return Err(Error::Shape(format!(
                "JackknifePlus::interval: expected {} loo_predictions but got {}",
                self.loo_residuals.len(),
                loo_predictions.len()
            )));
        }
        let n = loo_predictions.len();
        let mut lows: Vec<f64> = (0..n)
            .map(|i| loo_predictions[i] - self.loo_residuals[i])
            .collect();
        let mut highs: Vec<f64> = (0..n)
            .map(|i| loo_predictions[i] + self.loo_residuals[i])
            .collect();
        lows.sort_by(|a, b| a.partial_cmp(b).unwrap());
        highs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rank_lo = (self.alpha * (n as f64 + 1.0)).floor() as usize;
        let rank_hi = ((1.0 - self.alpha) * (n as f64 + 1.0)).ceil() as usize;
        let lo_idx = rank_lo.min(n - 1);
        let hi_idx = rank_hi.saturating_sub(1).min(n - 1);
        Ok(PredictionInterval {
            low: lows[lo_idx],
            high: highs[hi_idx],
        })
    }
}
