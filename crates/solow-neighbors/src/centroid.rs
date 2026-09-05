//! Nearest-centroid classifier (Rocchio 1971) — a memory-cheap,
//! parameter-free classifier that assigns each new sample to the
//! class whose centroid is Euclidean-nearest.
//!
//! Optional per-feature within-class shrinkage (Tibshirani-Hastie-
//! Narasimhan-Chu 2002, "shrunken centroids") — the classical text-
//! classification setting.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Fitted nearest-centroid classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct NearestCentroid {
    /// Class centroids `(n_classes, d)`.
    pub centroids: Array2<f64>,
    /// Number of classes.
    pub n_classes: usize,
    /// Shrinkage threshold used at fit time (`0` for no shrinkage).
    pub shrink_threshold: f64,
}

impl NearestCentroid {
    /// Fit with no shrinkage.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 0.0)
    }

    /// Fit with optional shrinkage threshold — every centroid
    /// coordinate is soft-thresholded toward the global mean by
    /// `shrink_threshold · pooled_std_dev`. Matches
    /// `neighbors.NearestCentroid(shrink_threshold=…)`.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        shrink_threshold: f64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "NearestCentroid::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if !(shrink_threshold >= 0.0 && shrink_threshold.is_finite()) {
            return Err(Error::Value(
                "NearestCentroid::fit_with: shrink_threshold must be finite and ≥ 0".into(),
            ));
        }
        let n = x.nrows();
        let d = x.ncols();
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        // Per-class sums and counts.
        let mut sums = Array2::<f64>::zeros((n_classes, d));
        let mut counts = vec![0usize; n_classes];
        for i in 0..n {
            let c = y[i];
            counts[c] += 1;
            for j in 0..d {
                sums[[c, j]] += x[[i, j]];
            }
        }
        // Centroids.
        let mut centroids = Array2::<f64>::zeros((n_classes, d));
        for c in 0..n_classes {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[[c, j]] = sums[[c, j]] / counts[c] as f64;
                }
            }
        }
        if shrink_threshold > 0.0 {
            // Global mean.
            let mut global = vec![0.0_f64; d];
            for j in 0..d {
                for i in 0..n {
                    global[j] += x[[i, j]];
                }
                global[j] /= n as f64;
            }
            // Pooled within-class standard error per feature.
            let mut mse = vec![0.0_f64; d];
            for i in 0..n {
                let c = y[i];
                for j in 0..d {
                    let d_ic = x[[i, j]] - centroids[[c, j]];
                    mse[j] += d_ic * d_ic;
                }
            }
            let df = (n as f64 - n_classes as f64).max(1.0);
            for v in mse.iter_mut() {
                *v = (*v / df).sqrt();
            }
            let s0 = median(&mse.clone());
            for c in 0..n_classes {
                let m_c = if counts[c] > 0 {
                    ((1.0 / counts[c] as f64) - (1.0 / n as f64))
                        .max(0.0)
                        .sqrt()
                } else {
                    0.0
                };
                for j in 0..d {
                    let scale = m_c * (mse[j] + s0);
                    if scale <= 0.0 {
                        continue;
                    }
                    let diff = centroids[[c, j]] - global[j];
                    let scaled = diff / scale;
                    let soft = soft_threshold(scaled, shrink_threshold);
                    centroids[[c, j]] = global[j] + scale * soft;
                }
            }
        }
        Ok(Self {
            centroids,
            n_classes,
            shrink_threshold,
        })
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        if x.ncols() != self.centroids.ncols() {
            return Err(Error::Shape(format!(
                "NearestCentroid::predict: expected {} columns, got {}",
                self.centroids.ncols(),
                x.ncols()
            )));
        }
        let mut out = Array1::<usize>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let (mut best_c, mut best_d) = (0usize, f64::INFINITY);
            for c in 0..self.n_classes {
                let mut dist2 = 0.0_f64;
                for j in 0..x.ncols() {
                    let dd = x[[i, j]] - self.centroids[[c, j]];
                    dist2 += dd * dd;
                }
                if dist2 < best_d {
                    best_d = dist2;
                    best_c = c;
                }
            }
            out[i] = best_c;
        }
        Ok(out)
    }
}

fn soft_threshold(z: f64, lambda: f64) -> f64 {
    if z > lambda {
        z - lambda
    } else if z < -lambda {
        z + lambda
    } else {
        0.0
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = s.len();
    if m == 0 {
        return 0.0;
    }
    if m % 2 == 0 {
        0.5 * (s[m / 2 - 1] + s[m / 2])
    } else {
        s[m / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn nearest_centroid_perfect_on_easy_data() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [5.0, 5.0],
            [5.5, 5.5],
            [4.5, 5.5],
            [5.5, 4.5]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let nc = NearestCentroid::fit(x.view(), y.view()).unwrap();
        let p = nc.predict(x.view()).unwrap();
        assert_eq!(p, y);
    }

    #[test]
    fn nearest_centroid_shrinkage_still_classifies_easy_data() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [5.0, 5.0],
            [5.5, 5.5],
            [4.5, 5.5],
            [5.5, 4.5]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let nc = NearestCentroid::fit_with(x.view(), y.view(), 0.5).unwrap();
        let p = nc.predict(x.view()).unwrap();
        assert_eq!(p, y);
    }
}
