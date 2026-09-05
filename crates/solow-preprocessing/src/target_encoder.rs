//! TargetEncoder — mean-target encoding with a smoothing prior.
//!
//! For each categorical column `j`, replaces every value `v` with
//!
//! ```text
//!     ĉ(v) = (nⱼ,ᵥ · μⱼ,ᵥ + m · μ_global) / (nⱼ,ᵥ + m)
//! ```
//!
//! `μⱼ,ᵥ` = mean of `y` when `xⱼ = v`, and `m` is a Bayesian
//! smoothing hyperparameter (`smooth` in the reference ≥ 1.3).

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// TargetEncoder.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct TargetEncoder {
    /// Per-column category → smoothed mean.
    pub encodings: Vec<std::collections::BTreeMap<i64, f64>>,
    /// Global fallback mean.
    pub target_mean: f64,
    /// Smoothing weight `m`.
    pub smooth: f64,
    /// Column count at fit time.
    pub n_features_in: usize,
}

impl TargetEncoder {
    /// Fit with `smooth = 10.0`.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[f64]) -> Result<Self> {
        Self::fit_with(x, y, 10.0)
    }

    /// Full-configuration fit. Column values are read as `i64` (i.e.,
    /// the caller has already integer-encoded the categorical column).
    pub fn fit_with(x: ArrayView2<'_, f64>, y: &[f64], smooth: f64) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("TargetEncoder: y/x length mismatch".into()));
        }
        if smooth < 0.0 {
            return Err(Error::Value("TargetEncoder: smooth must be ≥ 0".into()));
        }
        let d = x.ncols();
        let mut target_mean = 0.0_f64;
        for &v in y {
            target_mean += v;
        }
        target_mean /= n as f64;
        let mut encodings: Vec<std::collections::BTreeMap<i64, f64>> = Vec::with_capacity(d);
        for j in 0..d {
            let mut sums: std::collections::BTreeMap<i64, (f64, usize)> = Default::default();
            for i in 0..n {
                let cat = x[[i, j]].round() as i64;
                let e = sums.entry(cat).or_insert((0.0, 0));
                e.0 += y[i];
                e.1 += 1;
            }
            let mut map: std::collections::BTreeMap<i64, f64> = Default::default();
            for (k, (s, c)) in sums {
                let mean = s / c as f64;
                let smoothed = (c as f64 * mean + smooth * target_mean) / (c as f64 + smooth);
                map.insert(k, smoothed);
            }
            encodings.push(map);
        }
        Ok(Self {
            encodings,
            target_mean,
            smooth,
            n_features_in: d,
        })
    }

    /// Transform: replace each value with its smoothed target mean.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_in {
            return Err(Error::Shape("TargetEncoder::transform: column count mismatch".into()));
        }
        let n = x.nrows();
        let d = x.ncols();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let cat = x[[i, j]].round() as i64;
                out[[i, j]] = *self.encodings[j].get(&cat).unwrap_or(&self.target_mean);
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
    fn target_encoder_replaces_categories_with_smoothed_means() {
        // 3 categories, y = category * 10 → smoothed toward global mean = 10.
        let x = array![[0.0_f64], [0.0], [1.0], [1.0], [2.0], [2.0]];
        let y = vec![0.0_f64, 0.0, 10.0, 10.0, 20.0, 20.0];
        let te = TargetEncoder::fit_with(x.view(), &y, 1.0).unwrap();
        let z = te.transform(x.view()).unwrap();
        assert!((z[[0, 0]] - z[[1, 0]]).abs() < 1e-12);
        assert!(z[[0, 0]] < z[[2, 0]]);
        assert!(z[[2, 0]] < z[[4, 0]]);
    }
}
