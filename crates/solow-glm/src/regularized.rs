//! the reference-style thin wrappers on top of the low-level GLM API:
//! `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::family::Family;
use crate::glm::{Glm, GlmResults};
use crate::links::Link;
use crate::tweedie::TweedieGlm;

/// Fitted Poisson regressor (log link).
#[derive(Clone, Debug)]
pub struct PoissonRegressor {
    /// Coefficients `(d + 1)` — last is the intercept.
    pub coef: Array1<f64>,
    /// Full IRLS results kept for inspection.
    pub results: GlmResults,
    /// L2 penalty α used.
    pub alpha: f64,
}

impl PoissonRegressor {
    /// Fit with the reference defaults `alpha = 1.0`, `fit_intercept = true`.
    ///
    /// The `alpha` parameter is currently applied as a Ridge-style
    /// post-hoc shrink of the coefficient vector (a pragmatic proxy
    /// while the IRLS solver in solow-glm doesn't yet expose L2).
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, alpha: f64) -> Result<Self> {
        for &yi in y.iter() {
            if yi < 0.0 {
                return Err(Error::Value("PoissonRegressor: y must be ≥ 0".into()));
            }
        }
        let (xd, _has_const) = ensure_constant(x);
        let glm = Glm::with_link(y.to_owned(), xd, Family::Poisson, Link::Log)?;
        let results = glm.fit()?;
        let mut coef = results.params.clone();
        if alpha > 0.0 && coef.len() > 1 {
            let shrink = 1.0 / (1.0 + alpha);
            for k in 0..(coef.len() - 1) {
                coef[k] *= shrink;
            }
        }
        Ok(Self { coef, results, alpha })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if self.coef.len() != d + 1 {
            return Err(Error::Shape("PoissonRegressor::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut eta = self.coef[d];
            for j in 0..d {
                eta += x[[i, j]] * self.coef[j];
            }
            out[i] = eta.exp();
        }
        Ok(out)
    }
}

/// Fitted Gamma regressor (log link).
#[derive(Clone, Debug)]
pub struct GammaRegressor {
    /// Coefficients `(d + 1)` — last is the intercept.
    pub coef: Array1<f64>,
    /// Full IRLS results kept for inspection.
    pub results: GlmResults,
    /// L2 penalty α used.
    pub alpha: f64,
}

impl GammaRegressor {
    /// Fit with the reference defaults `alpha = 1.0`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, alpha: f64) -> Result<Self> {
        for &yi in y.iter() {
            if yi <= 0.0 {
                return Err(Error::Value("GammaRegressor: y must be > 0".into()));
            }
        }
        let (xd, _has_const) = ensure_constant(x);
        let glm = Glm::with_link(y.to_owned(), xd, Family::Gamma, Link::Log)?;
        let results = glm.fit()?;
        let mut coef = results.params.clone();
        if alpha > 0.0 && coef.len() > 1 {
            let shrink = 1.0 / (1.0 + alpha);
            for k in 0..(coef.len() - 1) {
                coef[k] *= shrink;
            }
        }
        Ok(Self { coef, results, alpha })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if self.coef.len() != d + 1 {
            return Err(Error::Shape("GammaRegressor::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut eta = self.coef[d];
            for j in 0..d {
                eta += x[[i, j]] * self.coef[j];
            }
            out[i] = eta.exp();
        }
        Ok(out)
    }
}

/// Fitted Tweedie regressor (log link, `p ∈ (1, 2)`).
#[derive(Clone, Debug)]
pub struct TweedieRegressor {
    /// Coefficients `(d + 1)` — last is the intercept.
    pub coef: Array1<f64>,
    /// Full IRLS results.
    pub results: GlmResults,
    /// Variance power.
    pub power: f64,
    /// L2 penalty α used.
    pub alpha: f64,
}

impl TweedieRegressor {
    /// Fit with the reference defaults `power = 1.5`, `alpha = 1.0`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, alpha: f64) -> Result<Self> {
        Self::fit_with(x, y, 1.5, alpha)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        power: f64,
        alpha: f64,
    ) -> Result<Self> {
        for &yi in y.iter() {
            if yi < 0.0 {
                return Err(Error::Value("TweedieRegressor: y must be ≥ 0".into()));
            }
        }
        let (xd, _has_const) = ensure_constant(x);
        let results = TweedieGlm::new(y.to_owned(), xd, power)?.fit()?;
        let mut coef = results.params.clone();
        if alpha > 0.0 && coef.len() > 1 {
            let shrink = 1.0 / (1.0 + alpha);
            for k in 0..(coef.len() - 1) {
                coef[k] *= shrink;
            }
        }
        Ok(Self {
            coef,
            results,
            power,
            alpha,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if self.coef.len() != d + 1 {
            return Err(Error::Shape("TweedieRegressor::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut eta = self.coef[d];
            for j in 0..d {
                eta += x[[i, j]] * self.coef[j];
            }
            out[i] = eta.exp();
        }
        Ok(out)
    }
}

fn ensure_constant(x: ArrayView2<'_, f64>) -> (Array2<f64>, bool) {
    // Add an intercept column of 1s if the last column isn't already
    // an all-ones column.
    let n = x.nrows();
    let d = x.ncols();
    let last_ones = d > 0 && (0..n).all(|i| (x[[i, d - 1]] - 1.0).abs() < 1e-12);
    if last_ones {
        return (x.to_owned(), true);
    }
    let mut xa = Array2::<f64>::zeros((n, d + 1));
    for i in 0..n {
        for j in 0..d {
            xa[[i, j]] = x[[i, j]];
        }
        xa[[i, d]] = 1.0;
    }
    (xa, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn poisson_regressor_recovers_a_log_linear_mean() {
        // μ = exp(0.5 + 0.3·x)
        let x = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let mu = [0.5_f64, 0.8, 1.1, 1.4, 1.7, 2.0].map(|e| e.exp());
        let y = Array1::from_vec(mu.iter().cloned().collect::<Vec<_>>());
        let m = PoissonRegressor::fit(x.view(), y.view(), 0.0).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..6 {
            assert!((p[i] - y[i]).abs() < 0.2, "row {i}: pred={} y={}", p[i], y[i]);
        }
    }
}
