//! # solow-calibration
//!
//! Post-hoc probability-calibration wrappers — the reference
//! `CalibratedClassifierCV`.
//!
//! * [`CalibratedClassifierCV`] — accepts a base classifier whose
//!   `decision_function` output is calibrated by either Platt's sigmoid
//!   (`Method::Sigmoid`) or an isotonic regression (`Method::Isotonic`).
//!   Uses a caller-supplied fold split for the training scheme.
//!
//! # References
//!
//! * Platt, J. (1999). *Probabilistic Outputs for Support Vector
//!   Machines and Comparisons to Regularized Likelihood Methods.*
//! * Zadrozny, B., & Elkan, C. (2002). *Transforming Classifier Scores
//!   into Accurate Multiclass Probability Estimates.* KDD 2002.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use ndarray::{Array1, ArrayView2};
use solow_core::{Error, Result};

/// The calibration method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Platt scaling — logistic regression on scores.
    Sigmoid,
    /// Isotonic regression on scores.
    Isotonic,
}

/// Base score-producing binary classifier.
pub trait ScoreClassifier {
    /// Fit on `(x, y)`.
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[u8]) -> Result<()>;

    /// Return the decision-function value (larger = more likely class 1).
    fn decision_function(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
}

/// Fitted calibrated classifier.
pub struct CalibratedClassifierCV<C: ScoreClassifier> {
    /// Trained base classifier (fitted on the full training set).
    pub base: C,
    /// Calibration method used.
    pub method: Method,
    /// Sigmoid fit `(a, b)` — Platt's `p = 1 / (1 + exp(a·f(x) + b))`.
    pub sigmoid_params: Option<(f64, f64)>,
    /// Isotonic thresholds and values (parallel arrays), sorted by threshold.
    pub isotonic_thresholds: Option<(Vec<f64>, Vec<f64>)>,
}

impl<C: ScoreClassifier> CalibratedClassifierCV<C> {
    /// Fit — trains `base` on the full data then calibrates its scores.
    pub fn fit(
        mut base: C,
        x: ArrayView2<'_, f64>,
        y: &[u8],
        method: Method,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(Error::Shape("CalibratedClassifierCV: y/x length mismatch".into()));
        }
        base.fit(x, y)?;
        let scores = base.decision_function(x)?;
        let (sig, iso) = match method {
            Method::Sigmoid => (Some(platt_fit(scores.as_slice().unwrap(), y)), None),
            Method::Isotonic => (
                None,
                Some(isotonic_fit(scores.as_slice().unwrap(), y)),
            ),
        };
        Ok(Self {
            base,
            method,
            sigmoid_params: sig,
            isotonic_thresholds: iso,
        })
    }

    /// Calibrated probability of class 1.
    pub fn predict_proba1(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let scores = self.base.decision_function(x)?;
        let mut out = Array1::<f64>::zeros(scores.len());
        for i in 0..scores.len() {
            out[i] = match self.method {
                Method::Sigmoid => {
                    let (a, b) = self.sigmoid_params.unwrap();
                    1.0 / (1.0 + (a * scores[i] + b).exp())
                }
                Method::Isotonic => {
                    let (t, v) = self.isotonic_thresholds.as_ref().unwrap();
                    isotonic_lookup(t, v, scores[i])
                }
            };
        }
        Ok(out)
    }
}

fn platt_fit(scores: &[f64], y: &[u8]) -> (f64, f64) {
    // Newton's method on the log-loss with priors as in Lin-Lin-Weng (2007).
    let n_pos = y.iter().filter(|&&yi| yi == 1).count() as f64;
    let n_neg = (y.len() as f64) - n_pos;
    let hi_target = (n_pos + 1.0) / (n_pos + 2.0);
    let lo_target = 1.0 / (n_neg + 2.0);
    let targets: Vec<f64> = y
        .iter()
        .map(|&yi| if yi == 1 { hi_target } else { lo_target })
        .collect();
    let mut a = 0.0_f64;
    let mut b = ((n_neg + 1.0) / (n_pos + 1.0)).ln();
    for _ in 0..50 {
        let mut h11 = 1e-12_f64;
        let mut h22 = 1e-12_f64;
        let mut h12 = 0.0_f64;
        let mut g1 = 0.0_f64;
        let mut g2 = 0.0_f64;
        for i in 0..scores.len() {
            let f = a * scores[i] + b;
            let (p, q) = if f >= 0.0 {
                let e = (-f).exp();
                (e / (1.0 + e), 1.0 / (1.0 + e))
            } else {
                let e = f.exp();
                (1.0 / (1.0 + e), e / (1.0 + e))
            };
            let d1 = targets[i] - p;
            let d2 = p * q;
            g1 += scores[i] * d1;
            g2 += d1;
            h11 += scores[i] * scores[i] * d2;
            h22 += d2;
            h12 += scores[i] * d2;
        }
        let det = h11 * h22 - h12 * h12;
        if det.abs() < 1e-20 {
            break;
        }
        let da = -(h22 * g1 - h12 * g2) / det;
        let db = -(-h12 * g1 + h11 * g2) / det;
        a += da;
        b += db;
        if da.abs() < 1e-8 && db.abs() < 1e-8 {
            break;
        }
    }
    (a, b)
}

fn isotonic_fit(scores: &[f64], y: &[u8]) -> (Vec<f64>, Vec<f64>) {
    // Pool-adjacent-violators (PAV) on scores sorted ascending.
    let mut pairs: Vec<(f64, f64)> =
        scores.iter().zip(y.iter()).map(|(&s, &yi)| (s, yi as f64)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = pairs.len();
    let mut values: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let mut weights = vec![1.0_f64; n];
    loop {
        let mut merged = false;
        let mut i = 0_usize;
        while i + 1 < values.len() {
            if values[i] > values[i + 1] {
                let new_w = weights[i] + weights[i + 1];
                let new_v = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / new_w;
                values[i] = new_v;
                weights[i] = new_w;
                values.remove(i + 1);
                weights.remove(i + 1);
                merged = true;
            } else {
                i += 1;
            }
        }
        if !merged {
            break;
        }
    }
    // Turn the merged blocks back into a step function.
    let mut thresholds: Vec<f64> = Vec::new();
    let mut out_values: Vec<f64> = Vec::new();
    let mut idx = 0_usize;
    let mut block = 0_usize;
    while block < values.len() {
        let block_size = weights[block] as usize;
        let last_in_block = idx + block_size - 1;
        thresholds.push(pairs[last_in_block].0);
        out_values.push(values[block]);
        idx += block_size;
        block += 1;
    }
    (thresholds, out_values)
}

fn isotonic_lookup(thresholds: &[f64], values: &[f64], s: f64) -> f64 {
    if thresholds.is_empty() {
        return 0.5;
    }
    if s <= thresholds[0] {
        return values[0];
    }
    if s >= *thresholds.last().unwrap() {
        return *values.last().unwrap();
    }
    // Binary search.
    let mut lo = 0_usize;
    let mut hi = thresholds.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if thresholds[mid] <= s {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Piecewise linear interpolation between values[lo] and values[hi].
    let t0 = thresholds[lo];
    let t1 = thresholds[hi];
    let alpha = (s - t0) / (t1 - t0).max(1e-12);
    values[lo] * (1.0 - alpha) + values[hi] * alpha
}

/// Commonly-used imports.
pub mod prelude {
    pub use super::{CalibratedClassifierCV, Method, ScoreClassifier};
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct Toy {
        threshold: f64,
    }

    impl ScoreClassifier for Toy {
        fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[u8]) -> Result<()> {
            let mut mp = 0.0_f64;
            let mut mn = 0.0_f64;
            let mut np = 0_usize;
            let mut nn = 0_usize;
            for i in 0..y.len() {
                if y[i] == 1 {
                    mp += x[[i, 0]];
                    np += 1;
                } else {
                    mn += x[[i, 0]];
                    nn += 1;
                }
            }
            self.threshold = 0.5 * (mp / np.max(1) as f64 + mn / nn.max(1) as f64);
            Ok(())
        }

        fn decision_function(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
            let n = x.nrows();
            let mut out = Array1::<f64>::zeros(n);
            for i in 0..n {
                out[i] = x[[i, 0]] - self.threshold;
            }
            Ok(out)
        }
    }

    #[test]
    fn calibrated_sigmoid_gives_monotone_probabilities() {
        let x = array![[0.0_f64], [0.5], [1.0], [1.5], [2.0], [2.5]];
        let y = vec![0_u8, 0, 0, 1, 1, 1];
        let c = CalibratedClassifierCV::fit(Toy { threshold: 0.0 }, x.view(), &y, Method::Sigmoid).unwrap();
        let probs = c.predict_proba1(x.view()).unwrap();
        for i in 1..6 {
            assert!(probs[i] >= probs[i - 1] - 1e-8, "monotonicity broken at row {i}");
        }
    }

    #[test]
    fn calibrated_isotonic_gives_monotone_probabilities() {
        let x = array![[0.0_f64], [0.5], [1.0], [1.5], [2.0], [2.5]];
        let y = vec![0_u8, 0, 0, 1, 1, 1];
        let c = CalibratedClassifierCV::fit(Toy { threshold: 0.0 }, x.view(), &y, Method::Isotonic).unwrap();
        let probs = c.predict_proba1(x.view()).unwrap();
        for i in 1..6 {
            assert!(probs[i] >= probs[i - 1] - 1e-8, "monotonicity broken at row {i}");
        }
    }
}
