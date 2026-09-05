//! `FunctionTransformer` — apply a caller-supplied elementwise (or
//! row-wise) function on transform.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted FunctionTransformer.
pub struct FunctionTransformer {
    /// Forward function `f(x) → x'`.
    pub func: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>,
    /// Optional inverse function `f⁻¹(x') → x`.
    pub inverse_func: Option<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>>,
    /// Column count seen at fit.
    pub n_features_in: usize,
    /// Whether to check that transform preserves column count.
    pub validate: bool,
}

impl FunctionTransformer {
    /// Fit — just captures the expected column count.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        func: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>,
        inverse_func: Option<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>>,
    ) -> Result<Self> {
        if x.ncols() == 0 {
            return Err(Error::Value("FunctionTransformer: no columns".into()));
        }
        Ok(Self {
            func,
            inverse_func,
            n_features_in: x.ncols(),
            validate: true,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if self.validate && x.ncols() != self.n_features_in {
            return Err(Error::Shape(
                "FunctionTransformer::transform: column count mismatch".into(),
            ));
        }
        (self.func)(x)
    }

    /// Inverse-transform if an inverse was provided.
    pub fn inverse_transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        match &self.inverse_func {
            None => Err(Error::Value(
                "FunctionTransformer::inverse_transform: no inverse function provided".into(),
            )),
            Some(f) => f(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn function_transformer_applies_log_and_inverses_it() {
        let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let forward: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>> =
            Box::new(|arr| {
                let mut out = arr.to_owned();
                for v in out.iter_mut() {
                    *v = (*v).ln();
                }
                Ok(out)
            });
        let inverse: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>> =
            Box::new(|arr| {
                let mut out = arr.to_owned();
                for v in out.iter_mut() {
                    *v = (*v).exp();
                }
                Ok(out)
            });
        let ft = FunctionTransformer::fit(x.view(), forward, Some(inverse)).unwrap();
        let z = ft.transform(x.view()).unwrap();
        let back = ft.inverse_transform(z.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((back[[i, j]] - x[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
