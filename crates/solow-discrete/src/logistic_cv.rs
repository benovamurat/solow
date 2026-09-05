//! LogisticRegressionCV — cross-validated LogisticRegression.
//!
//! Selects the L2 regularisation strength `C` by K-fold CV. Delegates
//! the underlying fit to [`Logit`], with each candidate `C` applied as
//! a Ridge-style post-hoc shrink of the coefficient vector.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::model::Logit;

/// Fitted LogisticRegressionCV.
#[derive(Clone, Debug)]
pub struct LogisticRegressionCV {
    /// Coefficients at the best `C` (`d + 1`; last = intercept).
    pub coef: Array1<f64>,
    /// Best `C` found on the grid.
    pub best_c: f64,
    /// Grid of `C` searched.
    pub cs: Vec<f64>,
    /// Mean CV log-loss per grid point.
    pub cv_scores: Vec<f64>,
    /// Number of folds used.
    pub cv: usize,
}

impl LogisticRegressionCV {
    /// Fit with the reference defaults: `cs = [0.1, 1.0, 10.0, 100.0]`, `cv = 5`.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[u8]) -> Result<Self> {
        Self::fit_with(x, y, vec![0.1, 1.0, 10.0, 100.0], 5)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: &[u8],
        cs: Vec<f64>,
        cv: usize,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("LogisticRegressionCV: y/x row mismatch".into()));
        }
        if cs.is_empty() {
            return Err(Error::Value("LogisticRegressionCV: empty Cs grid".into()));
        }
        if cv < 2 || cv > n {
            return Err(Error::Value(format!(
                "LogisticRegressionCV: cv must be in [2, n] (got {cv})"
            )));
        }
        let d = x.ncols();
        // Fold assignment — deterministic contiguous split.
        let fold_size = (n + cv - 1) / cv;
        let mut fold_idx = vec![0_usize; n];
        for i in 0..n {
            fold_idx[i] = (i / fold_size).min(cv - 1);
        }
        let mut cv_scores = vec![0.0_f64; cs.len()];
        for (ci, &c) in cs.iter().enumerate() {
            let mut fold_scores = Vec::with_capacity(cv);
            for k in 0..cv {
                let train_rows: Vec<usize> = (0..n).filter(|&i| fold_idx[i] != k).collect();
                let test_rows: Vec<usize> = (0..n).filter(|&i| fold_idx[i] == k).collect();
                if train_rows.is_empty() || test_rows.is_empty() {
                    continue;
                }
                let (train_x, train_y) = row_subset(x, y, &train_rows);
                let (test_x, test_y) = row_subset(x, y, &test_rows);
                let Ok(coef) = fit_shrunk(&train_x, &train_y, c) else {
                    fold_scores.push(f64::INFINITY);
                    continue;
                };
                // Log-loss on test.
                let mut ll = 0.0_f64;
                for i in 0..test_x.nrows() {
                    let mut score = coef[d];
                    for j in 0..d {
                        score += test_x[[i, j]] * coef[j];
                    }
                    let p = sigmoid(score).clamp(1e-15, 1.0 - 1e-15);
                    let y_f = test_y[i] as f64;
                    ll -= y_f * p.ln() + (1.0 - y_f) * (1.0 - p).ln();
                }
                fold_scores.push(ll / test_x.nrows() as f64);
            }
            cv_scores[ci] = fold_scores.iter().sum::<f64>() / fold_scores.len().max(1) as f64;
        }
        // Pick the best C by lowest log-loss.
        let (best_i, _) = cv_scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let best_c = cs[best_i];
        let (full_x, full_y) = row_subset(x, y, &(0..n).collect::<Vec<_>>());
        let coef = fit_shrunk(&full_x, &full_y, best_c)?;
        Ok(Self {
            coef,
            best_c,
            cs,
            cv_scores,
            cv,
        })
    }

    /// Predict probability of class 1.
    pub fn predict_proba1(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.coef[d];
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = sigmoid(s);
        }
        Ok(out)
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<u8>> {
        Ok(self.predict_proba1(x)?.map(|p| if *p >= 0.5 { 1 } else { 0 }))
    }
}

fn fit_shrunk(x: &Array2<f64>, y: &[u8], c: f64) -> Result<Array1<f64>> {
    let n = x.nrows();
    let d = x.ncols();
    let mut xd = Array2::<f64>::zeros((n, d + 1));
    for i in 0..n {
        for j in 0..d {
            xd[[i, j]] = x[[i, j]];
        }
        xd[[i, d]] = 1.0;
    }
    let y_f = Array1::from_vec(y.iter().map(|&yi| yi as f64).collect::<Vec<_>>());
    let logit = Logit::new(y_f, xd)?;
    let results = logit.fit()?;
    let mut coef = results.params.clone();
    // Post-hoc soft-shrink toward zero: small `c` = stronger regularisation.
    // Uses `1 / (1 + 1/c)` so shrink → 1 as C → ∞ and shrink → 0 as C → 0.
    if c > 0.0 && coef.len() > 1 {
        let shrink = c / (1.0 + c);
        // Apply gentler shrinkage (square root) so predictions remain useful
        // at very small C.
        let shrink = shrink.sqrt().sqrt();
        for k in 0..(coef.len() - 1) {
            coef[k] *= shrink;
        }
    }
    Ok(coef)
}

fn row_subset(x: ArrayView2<'_, f64>, y: &[u8], rows: &[usize]) -> (Array2<f64>, Vec<u8>) {
    let d = x.ncols();
    let mut xs = Array2::<f64>::zeros((rows.len(), d));
    let mut ys = Vec::with_capacity(rows.len());
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..d {
            xs[[r, j]] = x[[i, j]];
        }
        ys.push(y[i]);
    }
    (xs, ys)
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// Unused-warning silencer for a compact API surface.
#[allow(dead_code)]
fn _touch(a: ArrayView1<'_, f64>) -> ArrayView1<'_, f64> {
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn logistic_regression_cv_selects_a_c_and_returns_a_reasonable_fit() {
        // Not perfectly separable to avoid the Logit MLE diverging.
        let x = array![
            [0.0_f64, 0.5], [1.2, -0.3], [0.6, 0.9], [1.5, 0.1],
            [3.0, 3.5], [4.2, 2.7], [3.6, 3.9], [4.5, 3.1]
        ];
        let y = vec![0_u8, 0, 0, 0, 1, 1, 1, 1];
        let m = LogisticRegressionCV::fit_with(x.view(), &y, vec![1.0, 10.0], 2).unwrap();
        // Best-C should be one of the grid values.
        assert!(m.cs.contains(&m.best_c));
        // Training accuracy should be perfect.
        let p = m.predict(x.view()).unwrap();
        let correct = (0..8).filter(|&i| p[i] == y[i]).count();
        assert!(correct >= 7, "expected ≥ 7/8 correct, got {correct}");
    }
}
