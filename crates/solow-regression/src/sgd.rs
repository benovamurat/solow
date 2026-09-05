//! SGDRegressor, SGDClassifier, Perceptron, PassiveAggressiveRegressor,
//! PassiveAggressiveClassifier — first-order online learners.

use ndarray::{Array1, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Loss function for SGD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SgdLoss {
    /// Least-squares (regression).
    SquaredError,
    /// Huber (regression).
    Huber { epsilon: f64 },
    /// ε-insensitive (regression).
    EpsilonInsensitive { epsilon: f64 },
    /// Hinge (binary classification, `y ∈ {-1, +1}`).
    Hinge,
    /// Modified Huber (binary classification).
    ModifiedHuber,
    /// Logistic (binary classification).
    Log,
    /// Perceptron loss (binary classification).
    Perceptron,
}

/// Regularisation penalty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SgdPenalty {
    /// No penalty.
    None,
    /// L1 (Lasso-flavour).
    L1,
    /// L2 (Ridge-flavour).
    L2,
    /// Elastic net (`l1_ratio·L1 + (1 − l1_ratio)·L2`).
    ElasticNet { l1_ratio: f64 },
}

/// Fitted SGDRegressor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SgdRegressor {
    /// Coefficients `(d + 1)` (last = intercept).
    pub coef: Array1<f64>,
    /// Total steps taken.
    pub n_iter: usize,
    /// Whether an intercept is fit.
    pub fit_intercept: bool,
}

impl SgdRegressor {
    /// Fit with the reference defaults.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        Self::fit_with(
            x, y, SgdLoss::SquaredError, SgdPenalty::L2, 0.0001, true, 1000, 0.01, 42,
        )
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        loss: SgdLoss,
        penalty: SgdPenalty,
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        eta0: f64,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("SgdRegressor: y/x row mismatch".into()));
        }
        let p = if fit_intercept { d + 1 } else { d };
        let mut w = Array1::<f64>::zeros(p);
        let mut t = 1_u64;
        let mut state = seed.wrapping_add(0xF00D_C0DE);
        for _epoch in 0..max_iter {
            // Deterministic shuffle.
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = uniform_index(&mut state, (i + 1) as u64);
                order.swap(i, j);
            }
            for &i in &order {
                let mut yhat = if fit_intercept { w[d] } else { 0.0 };
                for j in 0..d {
                    yhat += x[[i, j]] * w[j];
                }
                let resid = y[i] - yhat;
                let grad_loss = match loss {
                    SgdLoss::SquaredError => -resid,
                    SgdLoss::Huber { epsilon } => {
                        if resid.abs() < epsilon {
                            -resid
                        } else {
                            -epsilon * resid.signum()
                        }
                    }
                    SgdLoss::EpsilonInsensitive { epsilon } => {
                        if resid.abs() < epsilon {
                            0.0
                        } else {
                            -resid.signum()
                        }
                    }
                    _ => 0.0,
                };
                let eta = eta0 / (1.0 + alpha * eta0 * t as f64);
                for j in 0..d {
                    let mut update = grad_loss * x[[i, j]];
                    update += match penalty {
                        SgdPenalty::None => 0.0,
                        SgdPenalty::L1 => alpha * w[j].signum(),
                        SgdPenalty::L2 => alpha * w[j],
                        SgdPenalty::ElasticNet { l1_ratio } => {
                            alpha * (l1_ratio * w[j].signum() + (1.0 - l1_ratio) * w[j])
                        }
                    };
                    w[j] -= eta * update;
                }
                if fit_intercept {
                    w[d] -= eta * grad_loss;
                }
                t += 1;
            }
        }
        Ok(Self {
            coef: w,
            n_iter: max_iter,
            fit_intercept,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if self.fit_intercept && self.coef.len() != d + 1 {
            return Err(Error::Shape("SgdRegressor::predict: shape mismatch".into()));
        }
        if !self.fit_intercept && self.coef.len() != d {
            return Err(Error::Shape("SgdRegressor::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = if self.fit_intercept { self.coef[d] } else { 0.0 };
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

/// Fitted SGDClassifier (binary, `y ∈ {-1, +1}`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SgdClassifier {
    /// Coefficients `(d + 1)` (last = intercept).
    pub coef: Array1<f64>,
    /// Total steps taken.
    pub n_iter: usize,
    /// Loss used.
    pub loss: SgdLoss,
}

impl SgdClassifier {
    /// Fit binary with `y ∈ {-1, +1}`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        Self::fit_with(
            x, y, SgdLoss::Hinge, SgdPenalty::L2, 0.0001, true, 1000, 0.01, 42,
        )
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        loss: SgdLoss,
        penalty: SgdPenalty,
        alpha: f64,
        fit_intercept: bool,
        max_iter: usize,
        eta0: f64,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("SgdClassifier: y/x row mismatch".into()));
        }
        let p = if fit_intercept { d + 1 } else { d };
        let mut w = Array1::<f64>::zeros(p);
        let mut t = 1_u64;
        let mut state = seed.wrapping_add(0xF00D_D00D);
        for _epoch in 0..max_iter {
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = uniform_index(&mut state, (i + 1) as u64);
                order.swap(i, j);
            }
            for &i in &order {
                let mut score = if fit_intercept { w[d] } else { 0.0 };
                for j in 0..d {
                    score += x[[i, j]] * w[j];
                }
                let z = y[i] * score;
                let grad = match loss {
                    SgdLoss::Hinge => {
                        if z < 1.0 { -y[i] } else { 0.0 }
                    }
                    SgdLoss::Perceptron => {
                        if score * y[i] <= 0.0 { -y[i] } else { 0.0 }
                    }
                    SgdLoss::ModifiedHuber => {
                        if z >= 1.0 {
                            0.0
                        } else if z >= -1.0 {
                            -2.0 * y[i] * (1.0 - z)
                        } else {
                            -4.0 * y[i]
                        }
                    }
                    SgdLoss::Log => -y[i] / (1.0 + (y[i] * score).exp()),
                    _ => 0.0,
                };
                let eta = eta0 / (1.0 + alpha * eta0 * t as f64);
                for j in 0..d {
                    let mut update = grad * x[[i, j]];
                    update += match penalty {
                        SgdPenalty::None => 0.0,
                        SgdPenalty::L1 => alpha * w[j].signum(),
                        SgdPenalty::L2 => alpha * w[j],
                        SgdPenalty::ElasticNet { l1_ratio } => {
                            alpha * (l1_ratio * w[j].signum() + (1.0 - l1_ratio) * w[j])
                        }
                    };
                    w[j] -= eta * update;
                }
                if fit_intercept {
                    w[d] -= eta * grad;
                }
                t += 1;
            }
        }
        Ok(Self {
            coef: w,
            n_iter: max_iter,
            loss,
        })
    }

    /// Predict binary labels `∈ {-1, +1}`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.coef[d];
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = if s >= 0.0 { 1.0 } else { -1.0 };
        }
        Ok(out)
    }
}

/// Perceptron — a special case of SGDClassifier with the perceptron loss.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Perceptron {
    /// Underlying SGD classifier.
    pub inner: SgdClassifier,
}

impl Perceptron {
    /// Fit with the reference defaults.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        let inner = SgdClassifier::fit_with(
            x, y, SgdLoss::Perceptron, SgdPenalty::None, 0.0, true, 1000, 1.0, 42,
        )?;
        Ok(Self { inner })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }
}

/// PassiveAggressiveClassifier — margin-based online classifier
/// (Crammer et al. 2006).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PassiveAggressiveClassifier {
    /// Coefficients `(d + 1)`.
    pub coef: Array1<f64>,
    /// Capacity `C`.
    pub c: f64,
    /// Total steps taken.
    pub n_iter: usize,
}

impl PassiveAggressiveClassifier {
    /// Fit binary `y ∈ {-1, +1}` with defaults `C = 1.0`, `max_iter = 1000`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        Self::fit_with(x, y, 1.0, 1000, 42)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        c: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("PA-Classifier: y/x row mismatch".into()));
        }
        let mut w = Array1::<f64>::zeros(d + 1);
        let mut state = seed.wrapping_add(0xF00D_C0DE);
        for _epoch in 0..max_iter {
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = uniform_index(&mut state, (i + 1) as u64);
                order.swap(i, j);
            }
            for &i in &order {
                let mut score = w[d];
                for j in 0..d {
                    score += x[[i, j]] * w[j];
                }
                let loss = (1.0 - y[i] * score).max(0.0);
                if loss > 0.0 {
                    let mut xn = 1.0_f64;
                    for j in 0..d {
                        xn += x[[i, j]] * x[[i, j]];
                    }
                    let tau = (loss / xn).min(c);
                    for j in 0..d {
                        w[j] += tau * y[i] * x[[i, j]];
                    }
                    w[d] += tau * y[i];
                }
            }
        }
        Ok(Self {
            coef: w,
            c,
            n_iter: max_iter,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.coef[d];
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = if s >= 0.0 { 1.0 } else { -1.0 };
        }
        Ok(out)
    }
}

/// PassiveAggressiveRegressor — ε-insensitive PA variant.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PassiveAggressiveRegressor {
    /// Coefficients `(d + 1)`.
    pub coef: Array1<f64>,
    /// Capacity `C`.
    pub c: f64,
    /// ε-insensitive band.
    pub epsilon: f64,
    /// Total steps taken.
    pub n_iter: usize,
}

impl PassiveAggressiveRegressor {
    /// Fit with the reference defaults `C = 1.0`, `epsilon = 0.1`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        Self::fit_with(x, y, 1.0, 0.1, 1000, 42)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        c: f64,
        epsilon: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("PA-Regressor: y/x row mismatch".into()));
        }
        let mut w = Array1::<f64>::zeros(d + 1);
        let mut state = seed.wrapping_add(0xF00D_C0DE);
        for _epoch in 0..max_iter {
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = uniform_index(&mut state, (i + 1) as u64);
                order.swap(i, j);
            }
            for &i in &order {
                let mut yhat = w[d];
                for j in 0..d {
                    yhat += x[[i, j]] * w[j];
                }
                let diff = y[i] - yhat;
                let loss = (diff.abs() - epsilon).max(0.0);
                if loss > 0.0 {
                    let mut xn = 1.0_f64;
                    for j in 0..d {
                        xn += x[[i, j]] * x[[i, j]];
                    }
                    let tau = (loss / xn).min(c) * diff.signum();
                    for j in 0..d {
                        w[j] += tau * x[[i, j]];
                    }
                    w[d] += tau;
                }
            }
        }
        Ok(Self {
            coef: w,
            c,
            epsilon,
            n_iter: max_iter,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.coef[d];
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let max = u64::MAX - (u64::MAX % n);
    if *state < max {
        (*state % n) as usize
    } else {
        (state.wrapping_mul(3) % n) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sgd_regressor_learns_a_linear_signal() {
        let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let m = SgdRegressor::fit_with(
            x.view(), y.view(),
            SgdLoss::SquaredError, SgdPenalty::L2, 1e-6, true, 500, 0.01, 42,
        ).unwrap();
        let p = m.predict(x.view()).unwrap();
        let mse = (0..8).map(|i| (p[i] - y[i]).powi(2)).sum::<f64>() / 8.0;
        assert!(mse < 0.5, "mse = {mse}");
    }

    #[test]
    fn sgd_classifier_separates_two_easy_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2], [5.3, 5.3]
        ];
        let y = array![-1.0_f64, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
        let m = SgdClassifier::fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..4 {
            assert_eq!(p[i], -1.0);
        }
        for i in 4..8 {
            assert_eq!(p[i], 1.0);
        }
    }

    #[test]
    fn perceptron_separates_two_easy_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = array![-1.0_f64, -1.0, -1.0, 1.0, 1.0, 1.0];
        let m = Perceptron::fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..3 {
            assert_eq!(p[i], -1.0);
        }
        for i in 3..6 {
            assert_eq!(p[i], 1.0);
        }
    }

    #[test]
    fn passive_aggressive_classifier_separates_two_easy_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = array![-1.0_f64, -1.0, -1.0, 1.0, 1.0, 1.0];
        let m = PassiveAggressiveClassifier::fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..3 {
            assert_eq!(p[i], -1.0);
        }
        for i in 3..6 {
            assert_eq!(p[i], 1.0);
        }
    }
}
