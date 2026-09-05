//! OutputCodeClassifier — Dietterich-Bakiri (1995) error-correcting
//! output codes.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::traits::{BinaryClassifier, MultiClassifier};

/// Fitted output-code classifier.
pub struct OutputCodeClassifier<C: BinaryClassifier, F: FnMut() -> C> {
    /// Trained binary estimators (one per code column).
    pub estimators: Vec<C>,
    /// Code matrix `(n_classes × n_codes)` in `{−1, +1}`.
    pub code: Array2<f64>,
    /// Sorted labels.
    pub classes: Vec<i64>,
    /// Factory kept for re-fits.
    pub factory: F,
    /// Seed used to draw the code.
    pub seed: u64,
}

impl<C: BinaryClassifier, F: FnMut() -> C> OutputCodeClassifier<C, F> {
    /// Fit with `code_size = 1.5 · n_classes`.
    pub fn fit(
        factory: F,
        x: ArrayView2<'_, f64>,
        y: &[i64],
        seed: u64,
    ) -> Result<Self> {
        Self::fit_with(factory, x, y, 1.5, seed)
    }

    /// Full-configuration fit — `code_size` is a multiplier on `n_classes`.
    pub fn fit_with(
        mut factory: F,
        x: ArrayView2<'_, f64>,
        y: &[i64],
        code_size: f64,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("OutputCodeClassifier: y/x length mismatch".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        if classes.len() < 2 {
            return Err(Error::Value("OutputCodeClassifier: need ≥ 2 classes".into()));
        }
        let n_classes = classes.len();
        let n_codes = ((n_classes as f64) * code_size).ceil().max(1.0) as usize;
        // Deterministic Rademacher code from the seed.
        let mut state = seed.wrapping_add(0xC0DE_C0DE_C0DE_C0DE);
        let mut code = Array2::<f64>::zeros((n_classes, n_codes));
        for i in 0..n_classes {
            for j in 0..n_codes {
                let r = lcg_next(&mut state);
                code[[i, j]] = if r & 1 == 0 { -1.0 } else { 1.0 };
            }
        }
        // Fit one binary classifier per code column.
        let mut estimators: Vec<C> = Vec::with_capacity(n_codes);
        for c in 0..n_codes {
            let yy: Vec<u8> = y.iter().map(|&yi| {
                let ci = classes.iter().position(|&v| v == yi).unwrap();
                if code[[ci, c]] > 0.0 { 1 } else { 0 }
            }).collect();
            let mut est = factory();
            est.fit(x, &yy)?;
            estimators.push(est);
        }
        Ok(Self {
            estimators,
            code,
            classes,
            factory,
            seed,
        })
    }
}

impl<C: BinaryClassifier, F: FnMut() -> C> MultiClassifier for OutputCodeClassifier<C, F> {
    fn fit(&mut self, _x: ArrayView2<'_, f64>, _y: &[i64]) -> Result<()> {
        Err(Error::Value(
            "OutputCodeClassifier: use the associated `fit_with` constructor to re-seed".into(),
        ))
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let k = self.classes.len();
        let n_codes = self.code.ncols();
        // Get p(y=1) for each code column → convert to signed prediction 2p − 1.
        let mut signed = Array2::<f64>::zeros((n, n_codes));
        for c in 0..n_codes {
            let p = self.estimators[c].predict_proba1(x)?;
            for i in 0..n {
                signed[[i, c]] = 2.0 * p[i] - 1.0;
            }
        }
        // Class score = -Hamming distance to code row → highest is closest.
        // We turn scores into softmax-style probabilities across classes.
        let mut scores = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for cls in 0..k {
                let mut s = 0.0_f64;
                for c in 0..n_codes {
                    s += signed[[i, c]] * self.code[[cls, c]];
                }
                scores[[i, cls]] = s;
            }
        }
        // Softmax with numerical shift.
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let mut mx = f64::NEG_INFINITY;
            for c in 0..k {
                if scores[[i, c]] > mx {
                    mx = scores[[i, c]];
                }
            }
            let mut sum = 0.0_f64;
            for c in 0..k {
                let e = (scores[[i, c]] - mx).exp();
                out[[i, c]] = e;
                sum += e;
            }
            let sum = sum.max(1e-30);
            for c in 0..k {
                out[[i, c]] /= sum;
            }
        }
        Ok(out)
    }

    fn classes(&self) -> Vec<i64> {
        self.classes.clone()
    }
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

// Prevent unused-import warning.
#[allow(dead_code)]
fn _touch(a: Array1<f64>) -> Array1<f64> {
    a
}
