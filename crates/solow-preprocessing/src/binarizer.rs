//! `Binarizer` — element-wise threshold transformer.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted Binarizer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Binarizer {
    /// Threshold value.
    pub threshold: f64,
    /// Column count seen at fit.
    pub n_features_in: usize,
}

impl Binarizer {
    /// Fit with the given threshold.
    pub fn fit(x: ArrayView2<'_, f64>, threshold: f64) -> Result<Self> {
        if x.ncols() == 0 {
            return Err(Error::Value("Binarizer: no columns".into()));
        }
        if !threshold.is_finite() {
            return Err(Error::Value("Binarizer: threshold must be finite".into()));
        }
        Ok(Self {
            threshold,
            n_features_in: x.ncols(),
        })
    }

    /// Fit with the reference default `threshold = 0.0`.
    pub fn fit_default(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit(x, 0.0)
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_in {
            return Err(Error::Shape("Binarizer::transform: column count mismatch".into()));
        }
        let mut out = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                out[[i, j]] = if x[[i, j]] > self.threshold { 1.0 } else { 0.0 };
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
    fn binarizer_thresholds_at_zero_by_default() {
        let x = array![[-1.0_f64, 0.0, 1.0], [0.5, -0.5, 2.0]];
        let b = Binarizer::fit_default(x.view()).unwrap();
        let z = b.transform(x.view()).unwrap();
        assert_eq!(z[[0, 0]], 0.0);
        assert_eq!(z[[0, 1]], 0.0);
        assert_eq!(z[[0, 2]], 1.0);
        assert_eq!(z[[1, 0]], 1.0);
    }
}
