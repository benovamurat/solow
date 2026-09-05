//! MultiOutputRegressor / MultiOutputClassifier — one base estimator
//! trained independently per output column.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::traits::{MultiClassifier, Regressor};

/// Multi-output regressor.
pub struct MultiOutputRegressor<R: Regressor, F: FnMut() -> R> {
    /// One regressor per output column.
    pub estimators: Vec<R>,
    /// Factory kept for re-fits.
    pub factory: F,
    /// Number of outputs captured at fit.
    pub n_outputs: usize,
}

impl<R: Regressor, F: FnMut() -> R> MultiOutputRegressor<R, F> {
    /// Fit.
    pub fn fit(
        mut factory: F,
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Result<Self> {
        if x.nrows() != y.nrows() {
            return Err(Error::Shape("MultiOutputRegressor: row counts differ".into()));
        }
        let q = y.ncols();
        let n = y.nrows();
        let mut estimators: Vec<R> = Vec::with_capacity(q);
        for c in 0..q {
            let yc: Vec<f64> = (0..n).map(|i| y[[i, c]]).collect();
            let mut est = factory();
            est.fit(x, &yc)?;
            estimators.push(est);
        }
        Ok(Self {
            estimators,
            factory,
            n_outputs: q,
        })
    }

    /// Predict `f̂(X)` — a matrix `(n × q)`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let q = self.n_outputs;
        let mut out = Array2::<f64>::zeros((n, q));
        for c in 0..q {
            let p = self.estimators[c].predict(x)?;
            for i in 0..n {
                out[[i, c]] = p[i];
            }
        }
        Ok(out)
    }
}

/// Multi-output classifier — one classifier per column of `Y`.
pub struct MultiOutputClassifier<C: MultiClassifier, F: FnMut() -> C> {
    /// One classifier per output column.
    pub estimators: Vec<C>,
    /// Factory.
    pub factory: F,
    /// Number of outputs.
    pub n_outputs: usize,
}

impl<C: MultiClassifier, F: FnMut() -> C> MultiOutputClassifier<C, F> {
    /// Fit.
    pub fn fit(
        mut factory: F,
        x: ArrayView2<'_, f64>,
        y: &[Vec<i64>],
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("MultiOutputClassifier: y/x length mismatch".into()));
        }
        let q = y[0].len();
        for row in y {
            if row.len() != q {
                return Err(Error::Shape(
                    "MultiOutputClassifier: y rows have inconsistent widths".into(),
                ));
            }
        }
        let mut estimators: Vec<C> = Vec::with_capacity(q);
        for c in 0..q {
            let yc: Vec<i64> = y.iter().map(|r| r[c]).collect();
            let mut est = factory();
            est.fit(x, &yc)?;
            estimators.push(est);
        }
        Ok(Self {
            estimators,
            factory,
            n_outputs: q,
        })
    }

    /// Predict labels — an `(n × q)` grid of class ids.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Vec<Array1<i64>>> {
        let mut out: Vec<Array1<i64>> = Vec::with_capacity(self.n_outputs);
        for c in 0..self.n_outputs {
            out.push(self.estimators[c].predict(x)?);
        }
        Ok(out)
    }
}
