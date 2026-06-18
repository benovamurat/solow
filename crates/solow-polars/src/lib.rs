//! # solow-polars
//!
//! A thin, dependency-light bridge from [Polars](https://www.pola.rs/)
//! `DataFrame`s into the Solow statistical stack.
//!
//! Polars is the de-facto DataFrame library of the Rust data ecosystem, while
//! Solow provides the estimators (OLS/GLS, GLM, …). This crate closes the gap:
//! pull numeric columns out of a `DataFrame` as `ndarray` vectors / matrices and
//! fit a model in a single call, without hand-writing the column-to-array
//! plumbing every time.
//!
//! ```no_run
//! use polars::prelude::*;
//! use solow_polars::ols_from_frame;
//!
//! let df = df![
//!     "y" => [1.0_f64, 2.0, 3.0, 4.0, 5.0],
//!     "x1" => [1.0_f64, 2.0, 3.0, 4.0, 5.0],
//!     "x2" => [2.0_f64, 1.0, 4.0, 3.0, 6.0],
//! ]
//! .unwrap();
//!
//! // Fit `y ~ 1 + x1 + x2` straight from the DataFrame.
//! let res = ols_from_frame(&df, "y", &["x1", "x2"], true).unwrap();
//! println!("coefficients: {:?}", res.params);
//! println!("R^2 = {:.4}", res.rsquared);
//! ```

use ndarray::{Array1, Array2};
use polars::prelude::*;
use solow_core::error::{Error, Result};
use solow_core::tools::{add_constant, HasConstant};
use solow_glm::{Family, Glm, GlmResults};
use solow_regression::{LinearModel, LinearResults};

/// Translate a Polars error into the Solow [`Error`] vocabulary.
fn polars_err(context: &str, e: PolarsError) -> Error {
    Error::Value(format!("{context}: {e}"))
}

/// Convert a single numeric Polars [`Series`] into an `ndarray` [`Array1<f64>`].
///
/// The series is cast to `Float64` first, so integer, boolean and other numeric
/// dtypes are accepted. Fails if the column cannot be represented as `f64`
/// (e.g. a string column) or if it contains null values, since downstream
/// estimators reject non-finite input.
///
/// ```
/// use polars::prelude::*;
/// use solow_polars::series_to_array1;
///
/// let s = Series::new("x".into(), &[1i64, 2, 3]);
/// let v = series_to_array1(&s).unwrap();
/// assert_eq!(v.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn series_to_array1(series: &Series) -> Result<Array1<f64>> {
    let name = series.name();
    if series.null_count() > 0 {
        return Err(Error::Value(format!(
            "column '{name}' contains {} null value(s); drop or impute them first",
            series.null_count()
        )));
    }

    let cast = series
        .cast(&DataType::Float64)
        .map_err(|e| polars_err(&format!("column '{name}' is not numeric"), e))?;
    // A non-numeric column (e.g. strings) can *succeed* at casting to Float64 in
    // Polars by parsing each value and yielding null on failure. Catch that here
    // so we reject such columns instead of silently producing NaNs / nulls.
    if cast.null_count() > 0 {
        return Err(Error::Value(format!(
            "column '{name}' is not numeric (cast to f64 produced {} null value(s))",
            cast.null_count()
        )));
    }
    let chunked = cast
        .f64()
        .map_err(|e| polars_err(&format!("column '{name}' f64 view"), e))?;

    // `cont_slice` succeeds for a single-chunk, null-free Float64 column (the
    // common case after a cast); fall back to a copying iterator otherwise.
    let values: Vec<f64> = match chunked.cont_slice() {
        Ok(slice) => slice.to_vec(),
        Err(_) => chunked.into_no_null_iter().collect::<Vec<f64>>(),
    };
    Ok(Array1::from(values))
}

/// Convert the selected numeric columns of a [`DataFrame`] into a design matrix
/// [`Array2<f64>`], with one row per observation and one column per name in
/// `cols` (in the given order).
///
/// Every requested column is cast to `f64`; missing column names and
/// non-numeric / null-bearing columns produce a descriptive error.
///
/// ```
/// use polars::prelude::*;
/// use solow_polars::dataframe_to_array2;
///
/// let df = df!["a" => [1.0_f64, 2.0], "b" => [3.0_f64, 4.0]].unwrap();
/// let m = dataframe_to_array2(&df, &["a", "b"]).unwrap();
/// assert_eq!(m.dim(), (2, 2));
/// assert_eq!(m[[1, 0]], 2.0);
/// ```
pub fn dataframe_to_array2(df: &DataFrame, cols: &[&str]) -> Result<Array2<f64>> {
    if cols.is_empty() {
        return Err(Error::Value(
            "dataframe_to_array2: no columns requested".into(),
        ));
    }
    let nrows = df.height();
    let ncols = cols.len();
    let mut out = Array2::<f64>::zeros((nrows, ncols));

    for (j, &col_name) in cols.iter().enumerate() {
        let column = df
            .column(col_name)
            .map_err(|e| polars_err(&format!("no column '{col_name}'"), e))?;
        // `Column` -> materialized `Series`, then reuse the vector conversion so
        // the null / numeric checks stay in one place.
        let series = column.as_materialized_series();
        let v = series_to_array1(series)?;
        if v.len() != nrows {
            return Err(Error::Shape(format!(
                "column '{col_name}' has {} rows, expected {nrows}",
                v.len()
            )));
        }
        out.column_mut(j).assign(&v);
    }
    Ok(out)
}

/// Build a design matrix from `x_cols` (optionally prepending a constant column)
/// and fit an ordinary-least-squares model of `y_col` on it, returning the
/// fitted [`LinearResults`].
///
/// When `add_intercept` is `true`, a leading column of ones is prepended via
/// [`solow_core::tools::add_constant`] (skipping it if the data already carries a
/// constant column), so the first coefficient is the intercept.
///
/// ```
/// use polars::prelude::*;
/// use solow_polars::ols_from_frame;
///
/// // y = 2 + 3*x exactly.
/// let df = df![
///     "y" => [5.0_f64, 8.0, 11.0, 14.0],
///     "x" => [1.0_f64, 2.0, 3.0, 4.0],
/// ]
/// .unwrap();
/// let res = ols_from_frame(&df, "y", &["x"], true).unwrap();
/// assert!((res.params[0] - 2.0).abs() < 1e-9); // intercept
/// assert!((res.params[1] - 3.0).abs() < 1e-9); // slope
/// ```
pub fn ols_from_frame(
    df: &DataFrame,
    y_col: &str,
    x_cols: &[&str],
    add_intercept: bool,
) -> Result<LinearResults> {
    let (y, design) = build_endog_exog(df, y_col, x_cols, add_intercept)?;
    LinearModel::ols(y, design)?.fit()
}

/// Like [`ols_from_frame`], but fits a generalized linear model with the given
/// [`Family`] (e.g. [`Family::Poisson`] for counts, [`Family::Binomial`] for a
/// 0/1 response). Returns the fitted [`GlmResults`].
///
/// ```no_run
/// use polars::prelude::*;
/// use solow_glm::Family;
/// use solow_polars::glm_from_frame;
///
/// let df = df![
///     "count" => [1.0_f64, 3.0, 2.0, 5.0, 4.0],
///     "x"     => [0.0_f64, 1.0, 1.0, 2.0, 2.0],
/// ]
/// .unwrap();
/// let res = glm_from_frame(&df, "count", &["x"], true, Family::Poisson).unwrap();
/// println!("{:?}", res.params);
/// ```
pub fn glm_from_frame(
    df: &DataFrame,
    y_col: &str,
    x_cols: &[&str],
    add_intercept: bool,
    family: Family,
) -> Result<GlmResults> {
    let (y, design) = build_endog_exog(df, y_col, x_cols, add_intercept)?;
    Glm::new(y, design, family)?.fit()
}

/// Shared helper: pull the response vector and (optionally intercept-augmented)
/// design matrix out of a `DataFrame`.
fn build_endog_exog(
    df: &DataFrame,
    y_col: &str,
    x_cols: &[&str],
    add_intercept: bool,
) -> Result<(Array1<f64>, Array2<f64>)> {
    if x_cols.is_empty() {
        return Err(Error::Value(
            "need at least one predictor column in `x_cols`".into(),
        ));
    }
    let y_series = df
        .column(y_col)
        .map_err(|e| polars_err(&format!("no response column '{y_col}'"), e))?
        .as_materialized_series();
    let y = series_to_array1(y_series)?;

    let x = dataframe_to_array2(df, x_cols)?;
    let design = if add_intercept {
        add_constant(&x, true, HasConstant::Skip)?
    } else {
        x
    };
    Ok((y, design))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn series_to_array1_casts_integers() {
        let s = Series::new("x".into(), &[10i32, 20, 30]);
        let v = series_to_array1(&s).unwrap();
        assert_eq!(v.to_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn series_to_array1_rejects_strings() {
        let s = Series::new("name".into(), &["a", "b", "c"]);
        assert!(series_to_array1(&s).is_err());
    }

    #[test]
    fn series_to_array1_rejects_nulls() {
        let s = Series::new("x".into(), &[Some(1.0_f64), None, Some(3.0)]);
        let err = series_to_array1(&s).unwrap_err();
        assert!(matches!(err, Error::Value(_)));
    }

    #[test]
    fn dataframe_to_array2_selects_and_orders_columns() {
        let df = df![
            "a" => [1.0_f64, 2.0, 3.0],
            "b" => [4.0_f64, 5.0, 6.0],
            "c" => [7.0_f64, 8.0, 9.0],
        ]
        .unwrap();
        // Request in a non-source order to confirm ordering is honored.
        let m = dataframe_to_array2(&df, &["c", "a"]).unwrap();
        assert_eq!(m.dim(), (3, 2));
        assert_eq!(m.column(0).to_vec(), vec![7.0, 8.0, 9.0]);
        assert_eq!(m.column(1).to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dataframe_to_array2_errors_on_missing_column() {
        let df = df!["a" => [1.0_f64, 2.0]].unwrap();
        assert!(dataframe_to_array2(&df, &["a", "missing"]).is_err());
    }

    #[test]
    fn ols_recovers_known_coefficients() {
        // Construct data with an exact linear relationship:
        //   y = 1.5 + 2.0*x1 - 0.5*x2
        let x1 = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = [2.0_f64, 1.0, 4.0, 3.0, 6.0, 5.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.5 + 2.0 * a - 0.5 * b)
            .collect();
        let df = df![
            "y" => y,
            "x1" => x1.to_vec(),
            "x2" => x2.to_vec(),
        ]
        .unwrap();

        let res = ols_from_frame(&df, "y", &["x1", "x2"], true).unwrap();

        // params[0] = intercept, then x1, x2 in order.
        assert_eq!(res.params.len(), 3);
        assert!(
            approx(res.params[0], 1.5, 1e-8),
            "intercept = {}",
            res.params[0]
        );
        assert!(approx(res.params[1], 2.0, 1e-8), "b_x1 = {}", res.params[1]);
        assert!(
            approx(res.params[2], -0.5, 1e-8),
            "b_x2 = {}",
            res.params[2]
        );
        // Exact fit -> R^2 == 1.
        assert!(approx(res.rsquared, 1.0, 1e-10), "R^2 = {}", res.rsquared);
        assert_eq!(res.nobs as usize, 6);
    }

    #[test]
    fn ols_without_intercept_has_no_constant() {
        let df = df![
            "y" => [2.0_f64, 4.0, 6.0, 8.0],
            "x" => [1.0_f64, 2.0, 3.0, 4.0],
        ]
        .unwrap();
        let res = ols_from_frame(&df, "y", &["x"], false).unwrap();
        // Only the slope is estimated; y = 2*x.
        assert_eq!(res.params.len(), 1);
        assert!(
            approx(res.params[0], 2.0, 1e-9),
            "slope = {}",
            res.params[0]
        );
    }

    #[test]
    fn ols_intercept_is_skipped_when_column_already_constant() {
        // A user-supplied constant predictor column should not be duplicated.
        let df = df![
            "y" => [3.0_f64, 5.0, 7.0, 9.0],
            "ones" => [1.0_f64, 1.0, 1.0, 1.0],
            "x" => [1.0_f64, 2.0, 3.0, 4.0],
        ]
        .unwrap();
        let res = ols_from_frame(&df, "y", &["ones", "x"], true).unwrap();
        // add_constant(Skip) sees the existing constant column and adds nothing,
        // so we keep exactly two coefficients.
        assert_eq!(res.params.len(), 2);
        // y = 1 + 2*x  ->  intercept coeff (on `ones`) = 1, slope = 2.
        assert!(
            approx(res.params[0], 1.0, 1e-8),
            "const = {}",
            res.params[0]
        );
        assert!(
            approx(res.params[1], 2.0, 1e-8),
            "slope = {}",
            res.params[1]
        );
    }

    #[test]
    fn glm_poisson_from_frame_fits() {
        // Counts increasing with x; just confirm the bridge wires up and the
        // slope comes out positive and finite.
        let df = df![
            "count" => [1.0_f64, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0, 20.0],
            "x"     => [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ]
        .unwrap();
        let res = glm_from_frame(&df, "count", &["x"], true, Family::Poisson).unwrap();
        assert_eq!(res.params.len(), 2);
        assert!(res.params[1] > 0.0, "poisson slope = {}", res.params[1]);
        assert!(res.params.iter().all(|v| v.is_finite()));
    }
}
