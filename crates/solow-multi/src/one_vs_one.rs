//! OneVsOneClassifier meta-estimator.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::traits::{BinaryClassifier, MultiClassifier};

/// One-vs-one wrapper. Trains `k(k − 1)/2` binary classifiers, one per
/// pair, and votes at predict time.
pub struct OneVsOneClassifier<C: BinaryClassifier, F: FnMut() -> C> {
    /// Trained pairwise classifiers, indexed by `pair_index`.
    pub estimators: Vec<C>,
    /// Sorted unique labels seen at fit.
    pub classes: Vec<i64>,
    /// `pair_index[k]` is `(i, j)` — indices into `classes`.
    pub pair_index: Vec<(usize, usize)>,
    /// Factory kept for re-fits.
    pub factory: F,
}

impl<C: BinaryClassifier, F: FnMut() -> C> OneVsOneClassifier<C, F> {
    /// Fit.
    pub fn fit(mut factory: F, x: ArrayView2<'_, f64>, y: &[i64]) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("OneVsOneClassifier: y/x length mismatch".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        if classes.len() < 2 {
            return Err(Error::Value("OneVsOneClassifier: need at least 2 classes".into()));
        }
        let mut estimators: Vec<C> = Vec::new();
        let mut pair_index: Vec<(usize, usize)> = Vec::new();
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                let ci = classes[i];
                let cj = classes[j];
                let mut rows: Vec<usize> = Vec::new();
                let mut yy: Vec<u8> = Vec::new();
                for r in 0..n {
                    if y[r] == ci {
                        rows.push(r);
                        yy.push(0);
                    } else if y[r] == cj {
                        rows.push(r);
                        yy.push(1);
                    }
                }
                let sub = row_subset(x, &rows);
                let mut est = factory();
                est.fit(sub.view(), &yy)?;
                estimators.push(est);
                pair_index.push((i, j));
            }
        }
        Ok(Self {
            estimators,
            classes,
            pair_index,
            factory,
        })
    }
}

impl<C: BinaryClassifier, F: FnMut() -> C> MultiClassifier for OneVsOneClassifier<C, F> {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[i64]) -> Result<()> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("OneVsOneClassifier::fit: y/x length mismatch".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        let mut estimators: Vec<C> = Vec::new();
        let mut pair_index: Vec<(usize, usize)> = Vec::new();
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                let ci = classes[i];
                let cj = classes[j];
                let mut rows: Vec<usize> = Vec::new();
                let mut yy: Vec<u8> = Vec::new();
                for r in 0..n {
                    if y[r] == ci {
                        rows.push(r);
                        yy.push(0);
                    } else if y[r] == cj {
                        rows.push(r);
                        yy.push(1);
                    }
                }
                let sub = row_subset(x, &rows);
                let mut est = (self.factory)();
                est.fit(sub.view(), &yy)?;
                estimators.push(est);
                pair_index.push((i, j));
            }
        }
        self.estimators = estimators;
        self.classes = classes;
        self.pair_index = pair_index;
        Ok(())
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let k = self.classes.len();
        // Wu-Lin-Weng pairwise coupling: solve K r = ν  where
        //   K_ii = Σ_{j ≠ i} r_ji²  and  K_ij = -r_ij r_ji  for i ≠ j
        // simplifies to voting-with-probabilities in the equal-weight case.
        // We use the vote-count fallback for robustness across base classifiers.
        let mut votes = Array2::<f64>::zeros((n, k));
        for (e_idx, &(i, j)) in self.pair_index.iter().enumerate() {
            let p1 = self.estimators[e_idx].predict_proba1(x)?;
            for r in 0..n {
                votes[[r, j]] += p1[r];
                votes[[r, i]] += 1.0 - p1[r];
            }
        }
        // Row-normalise.
        for r in 0..n {
            let s: f64 = (0..k).map(|c| votes[[r, c]]).sum::<f64>().max(1e-30);
            for c in 0..k {
                votes[[r, c]] /= s;
            }
        }
        Ok(votes)
    }

    fn classes(&self) -> Vec<i64> {
        self.classes.clone()
    }
}

fn row_subset(x: ArrayView2<'_, f64>, rows: &[usize]) -> ndarray::Array2<f64> {
    let p = x.ncols();
    let mut out = ndarray::Array2::<f64>::zeros((rows.len(), p));
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..p {
            out[[r, j]] = x[[i, j]];
        }
    }
    out
}

// Prevent unused-import warnings if Array1 is only used behind trait paths.
#[allow(dead_code)]
fn _touch(a: Array1<f64>) -> Array1<f64> {
    a
}
