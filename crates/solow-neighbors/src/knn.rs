//! K-nearest-neighbours estimators built on top of [`crate::KdTree`].
//!
//! * [`KNeighborsClassifier`] — majority-vote classification. Ties on
//!   the vote break on the smallest class index, matching the reference.
//! * [`KNeighborsRegressor`] — mean of the `k` nearest neighbour target
//!   values. Uniform or distance-weighted.
//!
//! # Weighting
//!
//! [`WeightKind::Uniform`] gives every neighbour weight `1`.
//! [`WeightKind::Distance`] gives weight `1 / d`, clamped to a large
//! constant when `d = 0` (i.e. the query coincides with a training
//! point) — this matches the reference `'distance'` behaviour, which
//! puts all of the weight on the coincident point.
//!
//! # Complexity
//!
//! Fit is `O(n · log n)` (KDTree build). Predict per query is
//! `O(k · log n)` expected in low `d`; total `O(m · k · log n)`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::kdtree::{KdTree, Neighbor};

/// How to weight neighbours in classifier / regressor voting.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WeightKind {
    /// Every neighbour weight = 1.
    Uniform,
    /// Neighbour weight = 1 / distance (clamped to a large constant at zero).
    Distance,
}

// ---------------------------------------------------------------------------
// KNeighborsClassifier
// ---------------------------------------------------------------------------

/// K-nearest-neighbours classifier.
#[derive(Clone, Debug)]
pub struct KNeighborsClassifier {
    tree: KdTree,
    labels: Array1<usize>,
    n_classes: usize,
    /// Number of neighbours to vote with.
    pub k: usize,
    /// Weighting scheme.
    pub weights: WeightKind,
}

impl KNeighborsClassifier {
    /// Fit on features `x` and integer class labels `y`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        k: usize,
        weights: WeightKind,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "KNeighborsClassifier::fit: x has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        if k == 0 || k > x.nrows() {
            return Err(Error::Value(format!(
                "KNeighborsClassifier::fit: k must be in [1, n] (got k={k}, n={})",
                x.nrows()
            )));
        }
        let tree = KdTree::build(x, 16.max(k))?;
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        Ok(Self {
            tree,
            labels: y.to_owned(),
            n_classes,
            k,
            weights,
        })
    }

    /// Predict class labels for every row of `x`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        let mut out = Array1::<usize>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let nbrs = self.tree.k_nearest(x.row(i), self.k)?;
            out[i] = vote(&nbrs, &self.labels, self.n_classes, self.weights);
        }
        Ok(out)
    }

    /// Predict per-class probabilities for every row of `x`.
    ///
    /// Uniform-weighted probabilities are `count(class) / k`;
    /// distance-weighted probabilities are the normalised inverse-
    /// distance sums per class.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            let nbrs = self.tree.k_nearest(x.row(i), self.k)?;
            let mut ws = vec![0.0_f64; self.n_classes];
            let mut total = 0.0_f64;
            for n in &nbrs {
                let w = weight(n.distance, self.weights);
                ws[self.labels[n.index]] += w;
                total += w;
            }
            if total > 0.0 {
                for c in 0..self.n_classes {
                    out[[i, c]] = ws[c] / total;
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// KNeighborsRegressor
// ---------------------------------------------------------------------------

/// K-nearest-neighbours regressor.
#[derive(Clone, Debug)]
pub struct KNeighborsRegressor {
    tree: KdTree,
    y: Array1<f64>,
    /// Number of neighbours to average over.
    pub k: usize,
    /// Weighting scheme.
    pub weights: WeightKind,
}

impl KNeighborsRegressor {
    /// Fit on features `x` and continuous target `y`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        k: usize,
        weights: WeightKind,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "KNeighborsRegressor::fit: x has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        if k == 0 || k > x.nrows() {
            return Err(Error::Value(format!(
                "KNeighborsRegressor::fit: k must be in [1, n] (got k={k}, n={})",
                x.nrows()
            )));
        }
        for &v in y.iter() {
            if !v.is_finite() {
                return Err(Error::Value(
                    "KNeighborsRegressor::fit: y must be finite".into(),
                ));
            }
        }
        let tree = KdTree::build(x, 16.max(k))?;
        Ok(Self {
            tree,
            y: y.to_owned(),
            k,
            weights,
        })
    }

    /// Predict targets for every row of `x`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let nbrs = self.tree.k_nearest(x.row(i), self.k)?;
            let (mut num, mut den) = (0.0_f64, 0.0_f64);
            for n in &nbrs {
                let w = weight(n.distance, self.weights);
                num += w * self.y[n.index];
                den += w;
            }
            out[i] = if den > 0.0 {
                num / den
            } else {
                // Degenerate: all neighbours at infinite distance — return
                // the unweighted mean.
                let s: f64 = nbrs.iter().map(|n| self.y[n.index]).sum();
                s / nbrs.len() as f64
            };
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn weight(dist: f64, kind: WeightKind) -> f64 {
    match kind {
        WeightKind::Uniform => 1.0,
        WeightKind::Distance => {
            if dist <= 1e-15 {
                1e15
            } else {
                1.0 / dist
            }
        }
    }
}

fn vote(nbrs: &[Neighbor], labels: &Array1<usize>, n_classes: usize, kind: WeightKind) -> usize {
    let mut ws = vec![0.0_f64; n_classes];
    for n in nbrs {
        ws[labels[n.index]] += weight(n.distance, kind);
    }
    // Argmax with lowest-index tie-break.
    let mut best = 0usize;
    let mut best_w = f64::NEG_INFINITY;
    for (c, &w) in ws.iter().enumerate() {
        if w > best_w {
            best_w = w;
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    fn classifier_recovers_labels_at_training_points() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
            [11.0, 11.0]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let clf = KNeighborsClassifier::fit(x.view(), y.view(), 3, WeightKind::Uniform).unwrap();
        let pred = clf.predict(x.view()).unwrap();
        assert_eq!(pred, y);
        // predict_proba rows must sum to 1.
        let p = clf.predict_proba(x.view()).unwrap();
        for i in 0..p.nrows() {
            let s: f64 = p.row(i).iter().sum();
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn regressor_interpolates_between_neighbours() {
        // y = 2x on [0, 10].
        let x =
            ndarray::Array2::from_shape_vec((11, 1), (0..11).map(|i| i as f64).collect()).unwrap();
        let y = x.column(0).mapv(|v| 2.0 * v);
        let reg = KNeighborsRegressor::fit(x.view(), y.view(), 2, WeightKind::Uniform).unwrap();
        // At x = 3.4 the two closest points are 3 and 4, mean = (2·3 + 2·4)/2 = 7.
        let q = array![[3.4]];
        let pred = reg.predict(q.view()).unwrap();
        assert_abs_diff_eq!(pred[0], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn distance_weighting_gives_full_weight_to_a_coincident_point() {
        let x = array![[0.0], [1.0], [10.0]];
        let y = Array1::from(vec![5.0, 100.0, 999.0]);
        let reg = KNeighborsRegressor::fit(x.view(), y.view(), 3, WeightKind::Distance).unwrap();
        let q = array![[0.0]];
        // Query is at row 0 → distance-weighted prediction snaps to y[0] = 5.
        let pred = reg.predict(q.view()).unwrap();
        assert!((pred[0] - 5.0).abs() < 1e-6, "pred = {}", pred[0]);
    }
}
