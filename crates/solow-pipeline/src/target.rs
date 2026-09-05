//! `TransformedTargetRegressor` — regression on a transformed target
//! (compose.TransformedTargetRegressor).
//!
//! Wraps a regressor callback and a pair of forward / inverse target
//! transformers so the caller can, e.g., fit on `log(y)` and predict
//! back on the original scale.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// TransformedTargetRegressor. `fit_regressor` receives the transformed
/// target and returns a boxed predict closure.
pub struct TransformedTargetRegressor {
    /// The trained predict closure (returns predictions in the *transformed*
    /// target space).
    predict_fn: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>>,
    /// Inverse target-space function.
    inverse_target: Box<dyn Fn(&Array1<f64>) -> Array1<f64>>,
}

impl TransformedTargetRegressor {
    /// Fit.
    pub fn fit<F, G, H>(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        forward_target: F,
        inverse_target: G,
        fit_regressor: H,
    ) -> Result<Self>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
        G: Fn(&Array1<f64>) -> Array1<f64> + 'static,
        H: FnOnce(ArrayView2<'_, f64>, ArrayView1<'_, f64>)
            -> Result<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>>>,
    {
        if y.len() != x.nrows() {
            return Err(Error::Shape("TransformedTargetRegressor: y/x length mismatch".into()));
        }
        let y_owned = y.to_owned();
        let y_transformed = forward_target(&y_owned);
        let predict_fn = fit_regressor(x, y_transformed.view())?;
        Ok(Self {
            predict_fn,
            inverse_target: Box::new(inverse_target),
        })
    }

    /// Predict — the returned values are in the ORIGINAL target scale.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let raw = (self.predict_fn)(x)?;
        Ok((self.inverse_target)(&raw))
    }
}

// Prevent unused-import warning on Array2.
#[allow(dead_code)]
fn _touch(a: Array2<f64>) -> Array2<f64> {
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn transformed_target_regressor_fits_on_log_and_predicts_on_original_scale() {
        // y = exp(2 + 0.5·x)  →  log(y) = 2 + 0.5·x
        let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
        let y_raw = array![
            (2.5_f64).exp(),
            (3.0_f64).exp(),
            (3.5_f64).exp(),
            (4.0_f64).exp(),
            (4.5_f64).exp()
        ];
        let m = TransformedTargetRegressor::fit(
            x.view(),
            y_raw.view(),
            |y| y.iter().map(|v| v.ln()).collect::<Array1<f64>>(),
            |y| y.iter().map(|v| v.exp()).collect::<Array1<f64>>(),
            |x, y| {
                // Fit y = a + b·x via OLS then return a boxed predictor.
                let n = x.nrows() as f64;
                let mean_x = x.column(0).sum() / n;
                let mean_y = y.sum() / n;
                let mut sxy = 0.0_f64;
                let mut sxx = 0.0_f64;
                for i in 0..x.nrows() {
                    let dx = x[[i, 0]] - mean_x;
                    let dy = y[i] - mean_y;
                    sxy += dx * dy;
                    sxx += dx * dx;
                }
                let b = sxy / sxx;
                let a = mean_y - b * mean_x;
                Ok(Box::new(move |xnew: ArrayView2<'_, f64>| {
                    Ok(xnew.column(0).map(|xi| a + b * xi))
                }))
            },
        ).unwrap();
        let pred = m.predict(x.view()).unwrap();
        for i in 0..5 {
            assert!(
                (pred[i] - y_raw[i]).abs() < 0.1,
                "row {i}: pred = {}, y = {}",
                pred[i],
                y_raw[i]
            );
        }
    }
}
