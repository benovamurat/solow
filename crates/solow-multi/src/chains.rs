//! Classifier and regressor chains (Read et al. 2011). Each output is
//! predicted using the true features plus every earlier prediction in
//! the chain order, capturing label dependence.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::traits::{MultiClassifier, Regressor};

/// Classifier chain.
pub struct ClassifierChain<C: MultiClassifier, F: FnMut() -> C> {
    /// One classifier per output, in chain order.
    pub estimators: Vec<C>,
    /// Chain order — indices into the original `y` columns.
    pub order: Vec<usize>,
    /// Factory kept for re-fits.
    pub factory: F,
}

impl<C: MultiClassifier, F: FnMut() -> C> ClassifierChain<C, F> {
    /// Fit in the natural column order.
    pub fn fit(
        mut factory: F,
        x: ArrayView2<'_, f64>,
        y: &[Vec<i64>],
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("ClassifierChain: y/x length mismatch".into()));
        }
        let q = y[0].len();
        let order: Vec<usize> = (0..q).collect();
        let mut estimators: Vec<C> = Vec::with_capacity(q);
        let p = x.ncols();
        for step in 0..q {
            let n_extra = step;
            let mut xx = Array2::<f64>::zeros((n, p + n_extra));
            for i in 0..n {
                for j in 0..p {
                    xx[[i, j]] = x[[i, j]];
                }
                for k in 0..n_extra {
                    xx[[i, p + k]] = y[i][order[k]] as f64;
                }
            }
            let yc: Vec<i64> = (0..n).map(|i| y[i][order[step]]).collect();
            let mut est = factory();
            est.fit(xx.view(), &yc)?;
            estimators.push(est);
        }
        Ok(Self {
            estimators,
            order,
            factory,
        })
    }

    /// Predict — outputs are stacked in original column order.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Vec<Array1<i64>>> {
        let n = x.nrows();
        let p = x.ncols();
        let q = self.order.len();
        let mut chain_out: Vec<Array1<i64>> = vec![Array1::<i64>::zeros(n); q];
        for step in 0..q {
            let n_extra = step;
            let mut xx = Array2::<f64>::zeros((n, p + n_extra));
            for i in 0..n {
                for j in 0..p {
                    xx[[i, j]] = x[[i, j]];
                }
                for k in 0..n_extra {
                    xx[[i, p + k]] = chain_out[self.order[k]][i] as f64;
                }
            }
            let pred = self.estimators[step].predict(xx.view())?;
            chain_out[self.order[step]] = pred;
        }
        Ok(chain_out)
    }
}

/// Regressor chain.
pub struct RegressorChain<R: Regressor, F: FnMut() -> R> {
    /// One regressor per output.
    pub estimators: Vec<R>,
    /// Chain order.
    pub order: Vec<usize>,
    /// Factory.
    pub factory: F,
}

impl<R: Regressor, F: FnMut() -> R> RegressorChain<R, F> {
    /// Fit.
    pub fn fit(
        mut factory: F,
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.nrows() != n {
            return Err(Error::Shape("RegressorChain: row counts differ".into()));
        }
        let q = y.ncols();
        let p = x.ncols();
        let order: Vec<usize> = (0..q).collect();
        let mut estimators: Vec<R> = Vec::with_capacity(q);
        for step in 0..q {
            let n_extra = step;
            let mut xx = Array2::<f64>::zeros((n, p + n_extra));
            for i in 0..n {
                for j in 0..p {
                    xx[[i, j]] = x[[i, j]];
                }
                for k in 0..n_extra {
                    xx[[i, p + k]] = y[[i, order[k]]];
                }
            }
            let yc: Vec<f64> = (0..n).map(|i| y[[i, order[step]]]).collect();
            let mut est = factory();
            est.fit(xx.view(), &yc)?;
            estimators.push(est);
        }
        Ok(Self {
            estimators,
            order,
            factory,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let p = x.ncols();
        let q = self.order.len();
        let mut out = Array2::<f64>::zeros((n, q));
        for step in 0..q {
            let n_extra = step;
            let mut xx = Array2::<f64>::zeros((n, p + n_extra));
            for i in 0..n {
                for j in 0..p {
                    xx[[i, j]] = x[[i, j]];
                }
                for k in 0..n_extra {
                    xx[[i, p + k]] = out[[i, self.order[k]]];
                }
            }
            let pred = self.estimators[step].predict(xx.view())?;
            for i in 0..n {
                out[[i, self.order[step]]] = pred[i];
            }
        }
        Ok(out)
    }
}
