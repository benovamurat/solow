//! KNNImputer — replace missing values by the mean of the nearest
//! neighbours (Troyanskaya et al. 2001).

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted KNNImputer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KnnImputer {
    /// Training rows kept for lookup at transform time.
    pub reference: Array2<f64>,
    /// Number of neighbours considered.
    pub n_neighbors: usize,
    /// Column count.
    pub n_features_in: usize,
}

impl KnnImputer {
    /// Fit with `n_neighbors = 5`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 5)
    }

    /// Full-configuration fit.
    pub fn fit_with(x: ArrayView2<'_, f64>, n_neighbors: usize) -> Result<Self> {
        if n_neighbors == 0 {
            return Err(Error::Value("KnnImputer: n_neighbors must be ≥ 1".into()));
        }
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value("KnnImputer: empty input".into()));
        }
        Ok(Self {
            reference: x.to_owned(),
            n_neighbors,
            n_features_in: x.ncols(),
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_in {
            return Err(Error::Shape("KnnImputer::transform: column count mismatch".into()));
        }
        let n = x.nrows();
        let d = x.ncols();
        let m = self.reference.nrows();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                out[[i, j]] = x[[i, j]];
            }
        }
        for i in 0..n {
            let mut has_missing = false;
            for j in 0..d {
                if !x[[i, j]].is_finite() {
                    has_missing = true;
                    break;
                }
            }
            if !has_missing {
                continue;
            }
            // Compute distances to reference rows over shared observed features.
            let mut dists: Vec<(usize, f64)> = Vec::with_capacity(m);
            for r in 0..m {
                let mut sq = 0.0_f64;
                let mut cnt = 0_usize;
                for j in 0..d {
                    let a = x[[i, j]];
                    let b = self.reference[[r, j]];
                    if a.is_finite() && b.is_finite() {
                        sq += (a - b).powi(2);
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    dists.push((r, (sq * d as f64 / cnt as f64).sqrt()));
                }
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for j in 0..d {
                if !x[[i, j]].is_finite() {
                    let mut acc = 0.0_f64;
                    let mut cnt = 0_usize;
                    for (r, _) in dists.iter().take(self.n_neighbors) {
                        let v = self.reference[[*r, j]];
                        if v.is_finite() {
                            acc += v;
                            cnt += 1;
                        }
                    }
                    out[[i, j]] = if cnt > 0 { acc / cnt as f64 } else { 0.0 };
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn knn_imputer_fills_missing_from_nearest_neighbours() {
        let x = array![
            [0.0_f64, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [5.0, 5.0, 5.0],
            [5.1, 5.1, 5.1],
            [0.05, f64::NAN, 0.05]
        ];
        let m = KnnImputer::fit_with(x.view(), 3).unwrap();
        let z = m.transform(x.view()).unwrap();
        // The imputed value should be close to the mean of the closest rows (small cluster).
        assert!(z[[5, 1]] < 1.0, "expected small imputed value, got {}", z[[5, 1]]);
    }
}
