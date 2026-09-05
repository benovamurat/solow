//! Deterministic conditional-mean (regression) imputation for one variable.
//!
//! This is the deterministic core of a single MICE update step. A linear
//! regression of the target variable on the other variables is fit using only
//! the rows where the target is observed, and the fitted coefficients are then
//! used to predict the conditional mean of the target at the rows where it is
//! missing. No random posterior draw and no predictive-mean-matching lookup is
//! performed, so the result is reproducible and matches the reference exactly.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_regression::LinearModel;

/// Result of a single deterministic conditional-mean imputation.
#[derive(Debug, Clone)]
pub struct ConditionalMeanImputation {
    /// Regression coefficients from the fit on the observed rows.
    pub params: Array1<f64>,
    /// Conditional means predicted at the observed rows.
    pub fitted_observed: Array1<f64>,
    /// Conditional means predicted at the missing rows; these are the
    /// deterministic imputations.
    pub imputed_missing: Array1<f64>,
    /// Residual scale (`sigma^2`) of the observed-row fit.
    pub scale: f64,
}

/// Fit `endog_obs ~ exog_obs` by ordinary least squares and predict the
/// conditional mean at `exog_miss`.
///
/// The caller supplies the design matrices directly (including any intercept
/// column), exactly as the reference splits the data into observed and missing
/// blocks before imputing. `exog_obs` and `exog_miss` must share the same
/// number of columns, and `endog_obs` must have one entry per observed row.
pub fn conditional_mean_impute(
    endog_obs: Array1<f64>,
    exog_obs: Array2<f64>,
    exog_miss: &Array2<f64>,
) -> Result<ConditionalMeanImputation> {
    let (n_obs, k) = exog_obs.dim();
    if endog_obs.len() != n_obs {
        return Err(Error::Shape(format!(
            "endog_obs has length {} but exog_obs has {n_obs} rows",
            endog_obs.len()
        )));
    }
    if exog_miss.ncols() != k {
        return Err(Error::Shape(format!(
            "exog_miss has {} columns but exog_obs has {k}",
            exog_miss.ncols()
        )));
    }
    if n_obs == 0 {
        return Err(Error::Value("no observed rows to fit".into()));
    }

    let model = LinearModel::ols(endog_obs, exog_obs)?;
    let res = model.fit()?;
    let imputed_missing = model.predict(&res.params, exog_miss);

    Ok(ConditionalMeanImputation {
        params: res.params.clone(),
        fitted_observed: res.fittedvalues.clone(),
        imputed_missing,
        scale: res.scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn exact_linear_relationship_is_reproduced() {
        // y = 1 + 2*x on the observed rows; the conditional mean at a missing
        // row with x = 10 must be 21.
        let endog_obs = array![3.0, 5.0, 7.0, 9.0];
        let exog_obs = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let exog_miss = array![[1.0, 10.0], [1.0, 0.0]];
        let res = conditional_mean_impute(endog_obs, exog_obs, &exog_miss).unwrap();
        assert!((res.params[0] - 1.0).abs() < 1e-10);
        assert!((res.params[1] - 2.0).abs() < 1e-10);
        assert!((res.imputed_missing[0] - 21.0).abs() < 1e-9);
        assert!((res.imputed_missing[1] - 1.0).abs() < 1e-9);
        // A perfect fit has zero residual scale.
        assert!(res.scale.abs() < 1e-18);
    }

    #[test]
    fn rejects_shape_mismatches() {
        let endog_obs = array![1.0, 2.0];
        let exog_obs = array![[1.0, 0.0], [1.0, 1.0]];
        // exog_miss with wrong number of columns.
        let bad = array![[1.0]];
        assert!(conditional_mean_impute(endog_obs.clone(), exog_obs.clone(), &bad).is_err());
        // endog length mismatch.
        let short = array![1.0];
        let good_miss = array![[1.0, 2.0]];
        assert!(conditional_mean_impute(short, exog_obs, &good_miss).is_err());
    }
}
