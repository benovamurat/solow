//! Post-hoc probability calibrators.
//!
//! Three families of calibrator ship here — all three fit on a held-out
//! calibration set and expose a common `transform(scores) -> Vec<f64>`
//! interface so they can be swapped in and out of a scoring pipeline:
//!
//! * [`PlattScaling`] — logistic (Platt 1999) calibration for binary
//!   probabilistic classifiers: fit `σ(a · s + b)` on the calibration
//!   scores by Newton-Raphson.
//! * [`IsotonicRegression`] — the pool-adjacent-violators (PAV) monotone
//!   fit. Nonparametric; recovers arbitrary monotone score → probability
//!   maps but needs a few hundred calibration points to be stable.
//! * [`TemperatureScaling`] — single-parameter Guo-Pleiss-Sun-Weinberger
//!   (2017) softmax temperature for multiclass logits. Doesn't shift the
//!   argmax (accuracy stays the same) but sharpens or softens confidence.
//!
//! All three train through a bounded, convex objective, so no random
//! seed is required and repeated fits produce byte-identical models.

use ndarray::{ArrayView1, ArrayView2};
use solow_core::{Error, Result};

// ---------------------------------------------------------------------------
// Platt scaling
// ---------------------------------------------------------------------------

/// Logistic (Platt 1999) calibrator for binary probabilistic classifiers.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlattScaling {
    /// Slope of the fitted logistic.
    pub a: f64,
    /// Intercept of the fitted logistic.
    pub b: f64,
}

impl PlattScaling {
    /// Fit Platt scaling by Newton-Raphson on the log-likelihood.
    ///
    /// Uses the numerically-safe `t_i = (N_+ + 1) / (N_+ + 2)` and
    /// `t_i = 1 / (N_- + 2)` targets recommended by Platt (1999,
    /// Appendix D) to avoid overfitting near the sample boundaries.
    /// Converges in a handful of Newton steps on well-scaled inputs.
    pub fn fit(scores: ArrayView1<'_, f64>, y_true: &[bool]) -> Result<Self> {
        if scores.len() != y_true.len() {
            return Err(Error::Shape(format!(
                "PlattScaling::fit: scores has {} entries but y_true has {}",
                scores.len(),
                y_true.len()
            )));
        }
        if y_true.is_empty() {
            return Err(Error::Value(
                "PlattScaling::fit: at least one sample is required".into(),
            ));
        }
        let n_pos = y_true.iter().filter(|&&b| b).count();
        let n_neg = y_true.len() - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return Err(Error::Value(
                "PlattScaling::fit: both classes must be present in y_true".into(),
            ));
        }
        let hi_target = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let lo_target = 1.0 / (n_neg as f64 + 2.0);
        let t: Vec<f64> = y_true
            .iter()
            .map(|&b| if b { hi_target } else { lo_target })
            .collect();

        // Newton on -log-likelihood of f(x) = 1 / (1 + exp(a * x + b)).
        let mut a = 0.0_f64;
        let mut b = ((n_neg as f64 + 1.0) / (n_pos as f64 + 1.0)).ln();
        let n = scores.len();
        for _ in 0..100 {
            // Gradient and Hessian.
            let mut g1 = 0.0_f64;
            let mut g2 = 0.0_f64;
            let mut h11 = 1e-12_f64;
            let mut h12 = 0.0_f64;
            let mut h22 = 1e-12_f64;
            for i in 0..n {
                let fx = a * scores[i] + b;
                // p = 1 / (1 + exp(fx))
                let p = if fx >= 0.0 {
                    (-fx).exp() / (1.0 + (-fx).exp())
                } else {
                    1.0 / (1.0 + fx.exp())
                };
                let d1 = t[i] - p;
                g1 += scores[i] * d1;
                g2 += d1;
                let d2 = p * (1.0 - p);
                h11 += scores[i] * scores[i] * d2;
                h12 += scores[i] * d2;
                h22 += d2;
            }
            let det = h11 * h22 - h12 * h12;
            if det.abs() < 1e-300 {
                break;
            }
            let da = (h22 * g1 - h12 * g2) / det;
            let db = (-h12 * g1 + h11 * g2) / det;
            a += da;
            b += db;
            if da.abs() + db.abs() < 1e-10 {
                break;
            }
        }
        Ok(PlattScaling { a, b })
    }

    /// Apply the fitted logistic to new raw scores.
    pub fn transform(&self, scores: ArrayView1<'_, f64>) -> Vec<f64> {
        scores
            .iter()
            .map(|&s| {
                let fx = self.a * s + self.b;
                if fx >= 0.0 {
                    (-fx).exp() / (1.0 + (-fx).exp())
                } else {
                    1.0 / (1.0 + fx.exp())
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Isotonic regression (pool-adjacent-violators)
// ---------------------------------------------------------------------------

/// Monotone step-function calibrator fit by pool-adjacent-violators.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IsotonicRegression {
    /// Ascending unique score thresholds.
    pub thresholds: Vec<f64>,
    /// Predicted probability at each threshold (monotone non-decreasing).
    pub values: Vec<f64>,
}

impl IsotonicRegression {
    /// Fit a monotone non-decreasing step function by PAV.
    pub fn fit(scores: ArrayView1<'_, f64>, y_true: &[bool]) -> Result<Self> {
        if scores.len() != y_true.len() {
            return Err(Error::Shape(format!(
                "IsotonicRegression::fit: scores has {} entries but y_true has {}",
                scores.len(),
                y_true.len()
            )));
        }
        if y_true.is_empty() {
            return Err(Error::Value(
                "IsotonicRegression::fit: at least one sample is required".into(),
            ));
        }
        // Sort by score.
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());

        // Pool-adjacent-violators (weighted).
        let mut xs: Vec<f64> = order.iter().map(|&i| scores[i]).collect();
        let mut ys: Vec<f64> = order
            .iter()
            .map(|&i| if y_true[i] { 1.0 } else { 0.0 })
            .collect();
        let mut w: Vec<f64> = vec![1.0; xs.len()];

        let mut i = 0usize;
        while i + 1 < ys.len() {
            if ys[i] > ys[i + 1] {
                // Merge blocks i and i + 1.
                let wi = w[i];
                let wj = w[i + 1];
                let new_w = wi + wj;
                let new_y = (wi * ys[i] + wj * ys[i + 1]) / new_w;
                ys[i] = new_y;
                w[i] = new_w;
                // The block now covers the range [xs[i]..xs[i+1]]; keep the
                // left edge as the block's threshold.
                ys.remove(i + 1);
                w.remove(i + 1);
                xs.remove(i + 1);
                // Rewind to catch cascading violations.
                if i > 0 {
                    i -= 1;
                }
            } else {
                i += 1;
            }
        }
        Ok(IsotonicRegression {
            thresholds: xs,
            values: ys,
        })
    }

    /// Piecewise-linear interpolation to keep the calibrator smooth.
    pub fn transform(&self, scores: ArrayView1<'_, f64>) -> Vec<f64> {
        scores
            .iter()
            .map(|&s| {
                if self.thresholds.is_empty() {
                    return 0.5;
                }
                if s <= self.thresholds[0] {
                    return self.values[0];
                }
                if s >= *self.thresholds.last().unwrap() {
                    return *self.values.last().unwrap();
                }
                // Binary search for the interval.
                let mut lo = 0usize;
                let mut hi = self.thresholds.len() - 1;
                while lo + 1 < hi {
                    let mid = (lo + hi) / 2;
                    if self.thresholds[mid] <= s {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let (x0, x1) = (self.thresholds[lo], self.thresholds[hi]);
                let (y0, y1) = (self.values[lo], self.values[hi]);
                if (x1 - x0).abs() < 1e-300 {
                    return y1;
                }
                let t = (s - x0) / (x1 - x0);
                (1.0 - t) * y0 + t * y1
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Temperature scaling
// ---------------------------------------------------------------------------

/// Multiclass temperature-scaling calibrator (Guo-Pleiss-Sun-Weinberger 2017).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperatureScaling {
    /// The fitted temperature. Values greater than 1 soften confidence;
    /// values less than 1 sharpen it. `T = 1` is the identity map.
    pub temperature: f64,
}

impl TemperatureScaling {
    /// Fit temperature by golden-section search on the multiclass log-loss.
    ///
    /// The search interval is `[0.05, 10.0]`, which covers every regime
    /// encountered in practice; the resulting `T` is exact to the interval
    /// width divided by the golden ratio raised to the tolerance loop
    /// count (100 iterations, well below `1e-6`).
    pub fn fit(logits: ArrayView2<'_, f64>, y_true: &[usize]) -> Result<Self> {
        if logits.nrows() != y_true.len() {
            return Err(Error::Shape(format!(
                "TemperatureScaling::fit: logits has {} rows but y_true has {}",
                logits.nrows(),
                y_true.len()
            )));
        }
        let k = logits.ncols();
        if k < 2 {
            return Err(Error::Value(
                "TemperatureScaling::fit: logits must have ≥ 2 columns".into(),
            ));
        }
        for &y in y_true {
            if y >= k {
                return Err(Error::Value(format!(
                    "TemperatureScaling::fit: label {y} is out of range for {k} classes"
                )));
            }
        }

        let nll = |t: f64| -> f64 {
            let mut nll = 0.0_f64;
            for i in 0..logits.nrows() {
                let scaled: Vec<f64> = (0..k).map(|c| logits[[i, c]] / t).collect();
                let m = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = scaled.iter().map(|v| (v - m).exp()).sum();
                let log_z = m + sum_exp.ln();
                nll += log_z - scaled[y_true[i]];
            }
            nll
        };

        // Golden-section minimisation on [lo, hi].
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut lo = 0.05_f64;
        let mut hi = 10.0_f64;
        let mut c = hi - (hi - lo) / phi;
        let mut d = lo + (hi - lo) / phi;
        let mut fc = nll(c);
        let mut fd = nll(d);
        for _ in 0..100 {
            if fc < fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - (hi - lo) / phi;
                fc = nll(c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + (hi - lo) / phi;
                fd = nll(d);
            }
            if (hi - lo).abs() < 1e-8 {
                break;
            }
        }
        Ok(TemperatureScaling {
            temperature: 0.5 * (lo + hi),
        })
    }

    /// Apply the calibrator: softmax over `logits / T`.
    pub fn transform(&self, logits: ArrayView2<'_, f64>) -> ndarray::Array2<f64> {
        let (n, k) = (logits.nrows(), logits.ncols());
        let mut out = ndarray::Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let scaled: Vec<f64> = (0..k).map(|c| logits[[i, c]] / self.temperature).collect();
            let m = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = scaled.iter().map(|v| (v - m).exp()).sum();
            for c in 0..k {
                out[[i, c]] = ((scaled[c] - m).exp()) / sum_exp;
            }
        }
        out
    }
}
