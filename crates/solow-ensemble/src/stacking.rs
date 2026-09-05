//! Stacking meta-estimators — the reference `StackingClassifier` and
//! `StackingRegressor`. Base predictions are stacked column-wise and
//! fed to a linear meta-learner: least-squares regression for
//! `StackingRegressor`, and logistic regression (softmax closed-form
//! ridge fit) for `StackingClassifier`.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Stacking classifier.
pub struct StackingClassifier {
    /// Sorted labels seen at fit.
    pub classes: Vec<i64>,
    /// Level-0 base predictors (returning `(n × k)` probabilities in
    /// `classes` order).
    base_proba: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>>,
    /// Meta-model weights: `(k · n_bases + 1) × k` (last column = bias).
    pub coef: Array2<f64>,
}

impl StackingClassifier {
    /// Fit — trains the meta-learner (ridge multi-class regression) on
    /// the stacked probabilities.
    pub fn fit(
        base_proba: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>>,
        x: ArrayView2<'_, f64>,
        y: &[i64],
        ridge: f64,
    ) -> Result<Self> {
        if base_proba.is_empty() {
            return Err(Error::Value("StackingClassifier: no base predictors".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        let k = classes.len();
        let n = x.nrows();
        // Build the meta-feature matrix.
        let mut meta = Array2::<f64>::zeros((n, k * base_proba.len() + 1));
        for (b, f) in base_proba.iter().enumerate() {
            let p = f(x)?;
            for i in 0..n {
                for c in 0..k {
                    meta[[i, b * k + c]] = p[[i, c]];
                }
            }
        }
        for i in 0..n {
            meta[[i, k * base_proba.len()]] = 1.0;
        }
        // Y one-hot.
        let mut yy = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let idx = classes.iter().position(|&c| c == y[i]).unwrap();
            yy[[i, idx]] = 1.0;
        }
        // Ridge closed form: β = (XᵀX + λI)⁻¹ Xᵀ Y.
        let xtx = matmul_tn(&meta, &meta);
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows() {
            xtx_reg[[i, i]] += ridge;
        }
        let xtx_inv = invert(&xtx_reg)?;
        let xty = matmul_tn(&meta, &yy);
        let beta = matmul_nn(&xtx_inv, &xty);
        Ok(Self {
            classes,
            base_proba,
            coef: beta,
        })
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<i64>> {
        let n = x.nrows();
        let k = self.classes.len();
        let mut meta = Array2::<f64>::zeros((n, k * self.base_proba.len() + 1));
        for (b, f) in self.base_proba.iter().enumerate() {
            let p = f(x)?;
            for i in 0..n {
                for c in 0..k {
                    meta[[i, b * k + c]] = p[[i, c]];
                }
            }
        }
        for i in 0..n {
            meta[[i, k * self.base_proba.len()]] = 1.0;
        }
        let scores = matmul_nn(&meta, &self.coef);
        let mut out = Array1::<i64>::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_v = scores[[i, 0]];
            for c in 1..k {
                if scores[[i, c]] > best_v {
                    best_v = scores[[i, c]];
                    best = c;
                }
            }
            out[i] = self.classes[best];
        }
        Ok(out)
    }
}

/// Stacking regressor.
pub struct StackingRegressor {
    /// Base regressors (each returns per-row scalar predictions).
    base_predict: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>>>,
    /// Meta-learner weights.
    pub coef: Array1<f64>,
}

impl StackingRegressor {
    /// Fit.
    pub fn fit(
        base_predict: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>>>,
        x: ArrayView2<'_, f64>,
        y: &[f64],
        ridge: f64,
    ) -> Result<Self> {
        if base_predict.is_empty() {
            return Err(Error::Value("StackingRegressor: no base predictors".into()));
        }
        let n = x.nrows();
        let m = base_predict.len();
        let mut meta = Array2::<f64>::zeros((n, m + 1));
        for (b, f) in base_predict.iter().enumerate() {
            let p = f(x)?;
            for i in 0..n {
                meta[[i, b]] = p[i];
            }
        }
        for i in 0..n {
            meta[[i, m]] = 1.0;
        }
        let xtx = matmul_tn(&meta, &meta);
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows() {
            xtx_reg[[i, i]] += ridge;
        }
        let xtx_inv = invert(&xtx_reg)?;
        // Xᵀ y
        let mut xty = vec![0.0_f64; m + 1];
        for i in 0..n {
            for k in 0..(m + 1) {
                xty[k] += meta[[i, k]] * y[i];
            }
        }
        let mut coef = Array1::<f64>::zeros(m + 1);
        for i in 0..(m + 1) {
            let mut s = 0.0_f64;
            for j in 0..(m + 1) {
                s += xtx_inv[[i, j]] * xty[j];
            }
            coef[i] = s;
        }
        Ok(Self {
            base_predict,
            coef,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let m = self.base_predict.len();
        let mut acc = Array1::<f64>::zeros(n);
        for (b, f) in self.base_predict.iter().enumerate() {
            let p = f(x)?;
            for i in 0..n {
                acc[i] += self.coef[b] * p[i];
            }
        }
        for i in 0..n {
            acc[i] += self.coef[m];
        }
        Ok(acc)
    }
}

fn matmul_tn(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let k = a.ncols();
    let l = b.ncols();
    let inner = a.nrows();
    let mut out = Array2::<f64>::zeros((k, l));
    for i in 0..k {
        for j in 0..l {
            let mut s = 0.0_f64;
            for r in 0..inner {
                s += a[[r, i]] * b[[r, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn matmul_nn(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let n = b.ncols();
    let inner = a.ncols();
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for r in 0..inner {
                s += a[[i, r]] * b[[r, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn invert(m: &Array2<f64>) -> Result<Array2<f64>> {
    let n = m.nrows();
    let mut a = vec![vec![0.0_f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[[i, j]];
        }
        a[i][n + i] = 1.0;
    }
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if best < 1e-30 {
            return Err(Error::Value("stacking::invert: singular matrix".into()));
        }
        if piv != i {
            a.swap(i, piv);
        }
        let d = a[i][i];
        for c in 0..(2 * n) {
            a[i][c] /= d;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for c in 0..(2 * n) {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = a[i][n + j];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn stacking_regressor_fits_a_constant_meta_learner() {
        let f: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>> =
            Box::new(|x| Ok(Array1::<f64>::from_elem(x.nrows(), 1.0)));
        let x = array![[0.0_f64], [1.0], [2.0], [3.0]];
        let y = vec![2.0_f64, 4.0, 6.0, 8.0];
        let sr = StackingRegressor::fit(vec![f], x.view(), &y, 1e-6).unwrap();
        let p = sr.predict(x.view()).unwrap();
        // With one constant base predictor, coef·1 + b = mean(y) = 5 is the best fit.
        for i in 0..4 {
            assert!((p[i] - 5.0).abs() < 1e-2);
        }
    }
}
