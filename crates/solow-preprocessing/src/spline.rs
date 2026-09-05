//! SplineTransformer — B-spline basis expansion.
//!
//! Produces truncated-power basis columns of a given degree over a
//! set of caller-configured knots (uniform quantiles by default).
//! the reference ≥ 1.0 exposes this as `preprocessing.SplineTransformer`.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Knot placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KnotStrategy {
    /// Equally-spaced knots.
    Uniform,
    /// Quantile-based knots.
    Quantile,
}

/// Fitted SplineTransformer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SplineTransformer {
    /// Per-column knot vectors (sorted).
    pub knots: Vec<Vec<f64>>,
    /// Spline degree.
    pub degree: usize,
    /// Whether an intercept (constant column) is included.
    pub include_bias: bool,
    /// Column count at fit time.
    pub n_features_in: usize,
}

impl SplineTransformer {
    /// Fit with defaults `n_knots = 5`, `degree = 3`, `include_bias = true`,
    /// `strategy = Uniform`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 5, 3, KnotStrategy::Uniform, true)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_knots: usize,
        degree: usize,
        strategy: KnotStrategy,
        include_bias: bool,
    ) -> Result<Self> {
        if n_knots < 2 {
            return Err(Error::Value("SplineTransformer: n_knots must be ≥ 2".into()));
        }
        if degree == 0 {
            return Err(Error::Value("SplineTransformer: degree must be ≥ 1".into()));
        }
        let d = x.ncols();
        let mut knots: Vec<Vec<f64>> = Vec::with_capacity(d);
        for j in 0..d {
            let mut col: Vec<f64> = (0..x.nrows()).map(|i| x[[i, j]]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut k = Vec::with_capacity(n_knots);
            for i in 0..n_knots {
                let q = i as f64 / (n_knots - 1) as f64;
                let pos = match strategy {
                    KnotStrategy::Uniform => {
                        let (lo, hi) = (col[0], col[col.len() - 1]);
                        lo + q * (hi - lo)
                    }
                    KnotStrategy::Quantile => {
                        let p = (q * (col.len() - 1) as f64) as usize;
                        col[p]
                    }
                };
                k.push(pos);
            }
            knots.push(k);
        }
        Ok(Self {
            knots,
            degree,
            include_bias,
            n_features_in: d,
        })
    }

    /// Transform via the truncated-power basis `{1, x, x², …, xᵈ,
    /// (x − κ₁)ᵈ₊, …, (x − κₘ)ᵈ₊}` per column.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_in {
            return Err(Error::Shape("SplineTransformer::transform: column count mismatch".into()));
        }
        let n = x.nrows();
        let d = x.ncols();
        let per_col_basis = self.degree + self.knots[0].len();
        let stride = if self.include_bias { per_col_basis + 1 } else { per_col_basis };
        let mut out = Array2::<f64>::zeros((n, d * stride));
        for i in 0..n {
            for j in 0..d {
                let base = j * stride;
                let xij = x[[i, j]];
                let mut idx = 0_usize;
                if self.include_bias {
                    out[[i, base + idx]] = 1.0;
                    idx += 1;
                }
                let mut p = 1.0_f64;
                for _ in 0..self.degree {
                    p *= xij;
                    out[[i, base + idx]] = p;
                    idx += 1;
                }
                for &k in &self.knots[j] {
                    let d_val = (xij - k).max(0.0);
                    out[[i, base + idx]] = d_val.powi(self.degree as i32);
                    idx += 1;
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
    fn spline_transformer_widens_by_the_right_multiplier() {
        let x = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0]];
        // 5 knots + degree 3 + bias = 3 + 5 + 1 = 9 features per input col.
        let s = SplineTransformer::fit_with(x.view(), 5, 3, KnotStrategy::Uniform, true).unwrap();
        let z = s.transform(x.view()).unwrap();
        assert_eq!(z.shape(), &[6, 9]);
    }
}
