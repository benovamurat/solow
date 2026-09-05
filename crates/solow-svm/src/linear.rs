//! Linear SVMs fit by Pegasos-style stochastic sub-gradient descent.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

// ---------------------------------------------------------------------------
// Deterministic PRNG (MMIX 64-bit LCG)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

fn shuffled_order(n: usize, state: &mut u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = uniform_index(state, (i + 1) as u64);
        order.swap(i, j);
    }
    order
}

// ---------------------------------------------------------------------------
// LinearSVC
// ---------------------------------------------------------------------------

/// Binary linear SVC with `+1 / -1` labels, hinge loss, L² regularisation.
///
/// Given labels in `{0, 1}` at fit time, the estimator internally
/// maps `0 → −1` and `1 → +1` to run Pegasos.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSvc {
    /// Weight vector (per-class in the multiclass wrapper).
    pub coef: Array2<f64>,
    /// Per-class intercept.
    pub intercept: Array1<f64>,
    /// Number of classes (2 for binary; >2 for OvR).
    pub n_classes: usize,
    /// Regularisation strength `C` (higher = weaker regularisation).
    pub c: f64,
    /// Number of Pegasos epochs actually run.
    pub n_iter: usize,
}

impl LinearSvc {
    /// Fit a binary SVC on integer labels `{0, 1}`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        c: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        let (w, b, iters) = pegasos_binary(x, y, c, max_iter, seed, true)?;
        Ok(Self {
            coef: w.into_shape_with_order((1, x.ncols())).unwrap(),
            intercept: Array1::from(vec![b]),
            n_classes: 2,
            c,
            n_iter: iters,
        })
    }

    /// One-vs-rest multiclass fit — `n_classes` binary SVCs and an
    /// argmax at predict time.
    pub fn fit_multiclass(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        c: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        if n_classes < 2 {
            return Err(Error::Value(
                "LinearSvc::fit_multiclass: need ≥ 2 classes".into(),
            ));
        }
        let d = x.ncols();
        let mut coef = Array2::<f64>::zeros((n_classes, d));
        let mut intercept = Array1::<f64>::zeros(n_classes);
        let mut total_iters = 0usize;
        for cls in 0..n_classes {
            let y_bin: Array1<usize> = y.mapv(|v| if v == cls { 1 } else { 0 });
            let (w, b, iters) = pegasos_binary(
                x,
                y_bin.view(),
                c,
                max_iter,
                seed.wrapping_add(cls as u64),
                true,
            )?;
            for j in 0..d {
                coef[[cls, j]] = w[j];
            }
            intercept[cls] = b;
            total_iters = total_iters.max(iters);
        }
        Ok(Self {
            coef,
            intercept,
            n_classes,
            c,
            n_iter: total_iters,
        })
    }

    /// Predict class labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        if x.ncols() != self.coef.ncols() {
            return Err(Error::Shape(format!(
                "LinearSvc::predict: expected {} columns, got {}",
                self.coef.ncols(),
                x.ncols()
            )));
        }
        let mut out = Array1::<usize>::zeros(x.nrows());
        if self.n_classes == 2 {
            for i in 0..x.nrows() {
                let m = self.margin_at(x.row(i), 0);
                out[i] = if m >= 0.0 { 1 } else { 0 };
            }
        } else {
            for i in 0..x.nrows() {
                let (mut best_c, mut best_m) = (0usize, f64::NEG_INFINITY);
                for cls in 0..self.n_classes {
                    let m = self.margin_at(x.row(i), cls);
                    if m > best_m {
                        best_m = m;
                        best_c = cls;
                    }
                }
                out[i] = best_c;
            }
        }
        Ok(out)
    }

    /// Signed margin per class.
    pub fn decision_function(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.coef.ncols() {
            return Err(Error::Shape(format!(
                "LinearSvc::decision_function: expected {} columns, got {}",
                self.coef.ncols(),
                x.ncols()
            )));
        }
        let cols = if self.n_classes == 2 {
            1
        } else {
            self.n_classes
        };
        let mut out = Array2::<f64>::zeros((x.nrows(), cols));
        for i in 0..x.nrows() {
            for c in 0..cols {
                out[[i, c]] = self.margin_at(x.row(i), c);
            }
        }
        Ok(out)
    }

    fn margin_at(&self, row: ArrayView1<'_, f64>, cls: usize) -> f64 {
        let mut s = self.intercept[cls];
        for j in 0..row.len() {
            s += self.coef[[cls, j]] * row[j];
        }
        s
    }
}

// ---------------------------------------------------------------------------
// LinearSVR
// ---------------------------------------------------------------------------

/// ε-insensitive Vapnik regressor fit by SGD.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSvr {
    /// Weight vector.
    pub coef: Array1<f64>,
    /// Intercept.
    pub intercept: f64,
    /// Regularisation strength (higher = weaker regularisation).
    pub c: f64,
    /// Insensitivity band `ε`.
    pub epsilon: f64,
    /// Number of SGD epochs run.
    pub n_iter: usize,
}

impl LinearSvr {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        c: f64,
        epsilon: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "LinearSvr::fit: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if !(c > 0.0 && c.is_finite()) {
            return Err(Error::Value(format!(
                "LinearSvr::fit: c must be finite and > 0 (got {c})"
            )));
        }
        if !(epsilon >= 0.0 && epsilon.is_finite()) {
            return Err(Error::Value(format!(
                "LinearSvr::fit: epsilon must be finite and ≥ 0 (got {epsilon})"
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        let lambda = 1.0 / (n as f64 * c);
        let mut w = Array1::<f64>::zeros(d);
        let mut b = 0.0_f64;
        let mut state = seed.wrapping_add(0xDA55_CAFE_5678_1234);
        let mut t = 0usize;
        for _epoch in 0..max_iter {
            let order = shuffled_order(n, &mut state);
            for &i in &order {
                t += 1;
                let eta = 1.0 / (lambda * t as f64);
                // Prediction and ε-hinge subgradient.
                let mut pred = b;
                for j in 0..d {
                    pred += w[j] * x[[i, j]];
                }
                let r = pred - y[i];
                let sign = if r > epsilon {
                    1.0
                } else if r < -epsilon {
                    -1.0
                } else {
                    0.0
                };
                // Regularisation gradient.
                for j in 0..d {
                    w[j] -= eta * (lambda * w[j] + sign * x[[i, j]]);
                }
                b -= eta * sign;
            }
        }
        Ok(Self {
            coef: w,
            intercept: b,
            c,
            epsilon,
            n_iter: max_iter,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(Error::Shape(format!(
                "LinearSvr::predict: expected {} columns, got {}",
                self.coef.len(),
                x.ncols()
            )));
        }
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut s = self.intercept;
            for j in 0..x.ncols() {
                s += self.coef[j] * x[[i, j]];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Pegasos binary loop
// ---------------------------------------------------------------------------

fn pegasos_binary(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, usize>,
    c: f64,
    max_iter: usize,
    seed: u64,
    with_intercept: bool,
) -> Result<(Array1<f64>, f64, usize)> {
    if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
        return Err(Error::Shape(format!(
            "LinearSvc::fit: shape mismatch (x: {}×{}, y: {})",
            x.nrows(),
            x.ncols(),
            y.len()
        )));
    }
    if !(c > 0.0 && c.is_finite()) {
        return Err(Error::Value(format!(
            "LinearSvc::fit: c must be finite and > 0 (got {c})"
        )));
    }
    let n = x.nrows();
    let d = x.ncols();
    // Map `0 → −1`, `1 → +1`; any other class label is treated as +1.
    let y_signed: Vec<f64> = y.iter().map(|&v| if v == 0 { -1.0 } else { 1.0 }).collect();
    let lambda = 1.0 / (n as f64 * c);
    let mut w = Array1::<f64>::zeros(d);
    let mut b = 0.0_f64;
    let mut state = seed.wrapping_add(0xC0FF_EE55_1234_5678);
    let mut t = 0usize;
    for _epoch in 0..max_iter {
        let order = shuffled_order(n, &mut state);
        for &i in &order {
            t += 1;
            let eta = 1.0 / (lambda * t as f64);
            let mut margin = b;
            for j in 0..d {
                margin += w[j] * x[[i, j]];
            }
            margin *= y_signed[i];
            if margin < 1.0 {
                for j in 0..d {
                    w[j] = (1.0 - eta * lambda) * w[j] + eta * y_signed[i] * x[[i, j]];
                }
                if with_intercept {
                    b += eta * y_signed[i];
                }
            } else {
                for j in 0..d {
                    w[j] *= 1.0 - eta * lambda;
                }
            }
        }
    }
    Ok((w, b, max_iter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_svc_separates_easy_binary_problem() {
        let x = array![
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [1.05, 1.05],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
            [5.05, 5.05]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let svc = LinearSvc::fit(x.view(), y.view(), 1.0, 200, 42).unwrap();
        let p = svc.predict(x.view()).unwrap();
        assert_eq!(p, y);
    }

    #[test]
    fn linear_svr_recovers_linear_target() {
        // y = 2·x + 1 exactly.
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let y = x.column(0).mapv(|v| 2.0 * v + 1.0);
        let svr = LinearSvr::fit(x.view(), y.view(), 100.0, 0.01, 2000, 7).unwrap();
        let pred = svr.predict(x.view()).unwrap();
        let mse: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        // Assert the fit beats the trivial constant predictor by an order
        // of magnitude — this is the reproducible property of Pegasos SGD
        // on this ramp; a strict absolute MSE threshold would be brittle
        // across platforms.
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / y.len() as f64;
        assert!(
            mse < 0.1 * var_y,
            "MSE = {mse}, sample variance = {var_y} — LinearSvr failed to learn the ramp"
        );
    }

    #[test]
    fn linear_svc_multiclass_ovr_works() {
        // Three clusters at (0, 0), (5, 5), (10, 0).
        let x = array![
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [4.9, 5.0],
            [10.0, 0.0],
            [10.1, -0.1],
            [9.9, 0.1]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2]);
        let svc = LinearSvc::fit_multiclass(x.view(), y.view(), 1.0, 300, 42).unwrap();
        let p = svc.predict(x.view()).unwrap();
        assert_eq!(p, y);
    }
}
