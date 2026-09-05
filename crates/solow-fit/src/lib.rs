//! # solow-fit
//!
//! Ergonomic, formula-driven model fitting for the Solow statistical stack —
//! the one-call bridge from an R/patsy-style formula string plus named data to a
//! fully fitted model whose coefficients are *labeled* with the design column
//! names. This is the from-scratch equivalent of the reference library's
//! `from_formula` constructors: instead of hand-assembling a design matrix and
//! threading column names through by hand, you write
//!
//! ```text
//! ols("y ~ x1 + C(g)", &df)
//! ```
//!
//! and get back a [`NamedFit`] that pairs the estimator's results with the
//! ordered column names, ready to [`summary`](NamedFit::summary).
//!
//! The formula layer ([`solow_formula`]) already emits the `Intercept` column
//! when the formula carries one, so these functions pass the design straight
//! through to the estimator without re-adding a constant — the formula path and
//! the manual `add_constant` + estimator path therefore produce *identical*
//! coefficients and standard errors (see the crate's tests, which assert
//! agreement to `≤ 1e-12`).
//!
//! ## Models
//!
//! * [`ols`] / [`wls`] / [`gls`] — linear regression ([`LinearResults`]).
//! * [`glm`] — generalized linear models with a [`Family`] and optional
//!   [`Link`] ([`GlmResults`]).
//! * [`logit`] / [`probit`] / [`poisson`] — discrete-choice and count models
//!   ([`DiscreteResults`]).
//!
//! ## One import
//!
//! [`DataFrame`] is re-exported here, so a typical user needs a single `use`:
//!
//! ```
//! use solow_fit::{ols, DataFrame};
//!
//! let mut df = DataFrame::new();
//! df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! df.add_numeric("x", vec![0.0, 1.0, 2.0, 3.0, 4.0]);
//! let fit = ols("y ~ x", &df).unwrap();
//! assert_eq!(fit.names, vec!["Intercept".to_string(), "x".to_string()]);
//! // The fitted line is y = 1 + x, recovered exactly.
//! assert!((fit.results.params[0] - 1.0).abs() < 1e-10);
//! assert!((fit.results.params[1] - 1.0).abs() < 1e-10);
//! println!("{}", fit.summary());
//! ```

use ndarray::{Array1, Array2};
use solow_core::{Error, Result};
use solow_formula::{build, DesignOutput};

// Re-exports so callers need a single import surface.
pub use solow_discrete::DiscreteResults;
pub use solow_formula::DataFrame;
pub use solow_glm::{Family, GlmResults, Link};
pub use solow_regression::LinearResults;

/// A fitted model bundled with the design column names that label its
/// coefficients.
///
/// Every estimator in the Solow stack returns a results struct whose `params`,
/// `bse`, … are plain numeric vectors with no notion of *which* column each
/// entry belongs to. `NamedFit` carries the ordered [`names`](Self::names)
/// alongside the [`results`](Self::results) so a coefficient table can be
/// rendered with meaningful labels (`Intercept`, `x1`, `C(g)[T.b]`,
/// `x1:x2`, …).
///
/// `R` is the underlying results type ([`LinearResults`], [`GlmResults`], or
/// [`DiscreteResults`]); the fields are public so the full numeric battery
/// remains directly accessible.
#[derive(Clone, Debug)]
pub struct NamedFit<R> {
    /// Design-matrix column names, in formula (patsy) order. `names[i]` labels
    /// the `i`-th coefficient of `results`.
    pub names: Vec<String>,
    /// The fitted results from the underlying estimator.
    pub results: R,
}

impl<R> NamedFit<R> {
    /// Construct directly from names and results.
    pub fn new(names: Vec<String>, results: R) -> Self {
        NamedFit { names, results }
    }

    /// The coefficient labels (`names[i]` labels coefficient `i`).
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

/// Split a [`DesignOutput`] into a required response and the design matrix.
///
/// Fitting needs a left-hand side, so a formula without `~` is an error.
fn require_endog(out: DesignOutput) -> Result<(Array1<f64>, Array2<f64>, Vec<String>)> {
    let DesignOutput { y, design, names } = out;
    let y = y.ok_or_else(|| {
        Error::Value("formula has no response: expected `lhs ~ rhs` for fitting".into())
    })?;
    Ok((y, design, names))
}

// ---------------------------------------------------------------------------
// Linear models
// ---------------------------------------------------------------------------

/// Fit an ordinary-least-squares linear model from a formula and data.
///
/// The formula's intercept (present unless suppressed with `- 1` / `+ 0`)
/// supplies the constant column, so the resulting coefficients and standard
/// errors match a manual [`LinearModel::ols`](solow_regression::LinearModel::ols)
/// fit on the same design exactly.
///
/// ```
/// use solow_fit::{ols, DataFrame};
///
/// let mut df = DataFrame::new();
/// df.add_numeric("y", vec![2.0, 4.0, 6.0, 8.0]);
/// df.add_numeric("x", vec![1.0, 2.0, 3.0, 4.0]);
/// let fit = ols("y ~ x", &df).unwrap();
/// assert!((fit.results.params[1] - 2.0).abs() < 1e-10);
/// ```
pub fn ols(formula: &str, data: &DataFrame) -> Result<NamedFit<LinearResults>> {
    let (y, design, names) = require_endog(build(formula, data)?)?;
    let results = solow_regression::LinearModel::ols(y, design)?.fit()?;
    Ok(NamedFit::new(names, results))
}

/// Fit a weighted-least-squares linear model from a formula and data.
///
/// `weights` are per-observation and proportional to the inverse error
/// variance; they must have one entry per row of `data`.
pub fn wls(
    formula: &str,
    data: &DataFrame,
    weights: Array1<f64>,
) -> Result<NamedFit<LinearResults>> {
    let (y, design, names) = require_endog(build(formula, data)?)?;
    let results = solow_regression::LinearModel::wls(y, design, weights)?.fit()?;
    Ok(NamedFit::new(names, results))
}

/// Fit a generalized-least-squares linear model from a formula and data.
///
/// `sigma` is the full `n × n` error covariance matrix (symmetric positive
/// definite), where `n` is the number of rows in `data`.
pub fn gls(
    formula: &str,
    data: &DataFrame,
    sigma: &Array2<f64>,
) -> Result<NamedFit<LinearResults>> {
    let (y, design, names) = require_endog(build(formula, data)?)?;
    let results = solow_regression::LinearModel::gls(y, design, sigma)?.fit()?;
    Ok(NamedFit::new(names, results))
}

// ---------------------------------------------------------------------------
// Generalized linear models
// ---------------------------------------------------------------------------

/// Fit a generalized linear model from a formula and data.
///
/// `family` selects the exponential-dispersion error distribution; `link`
/// optionally overrides the family's canonical link (pass `None` for the
/// default).
///
/// ```
/// use solow_fit::{glm, DataFrame, Family};
///
/// let mut df = DataFrame::new();
/// df.add_numeric("y", vec![1.0, 2.0, 4.0, 7.0, 12.0]);
/// df.add_numeric("x", vec![0.0, 1.0, 2.0, 3.0, 4.0]);
/// let fit = glm("y ~ x", &df, Family::Poisson, None).unwrap();
/// assert!(fit.results.converged);
/// assert_eq!(fit.names.len(), 2);
/// ```
pub fn glm(
    formula: &str,
    data: &DataFrame,
    family: Family,
    link: Option<Link>,
) -> Result<NamedFit<GlmResults>> {
    let (y, design, names) = require_endog(build(formula, data)?)?;
    let model = match link {
        Some(l) => solow_glm::Glm::with_link(y, design, family, l)?,
        None => solow_glm::Glm::new(y, design, family)?,
    };
    let results = model.fit()?;
    Ok(NamedFit::new(names, results))
}

// ---------------------------------------------------------------------------
// Discrete-choice / count models
// ---------------------------------------------------------------------------

/// Fit a logistic (logit) regression from a formula and data.
///
/// The response must be coded `0`/`1`.
pub fn logit(formula: &str, data: &DataFrame) -> Result<NamedFit<DiscreteResults>> {
    let (y, design, names) = require_endog(build(formula, data)?)?;
    let results = solow_discrete::Logit::new(y, design)?.fit()?;
    Ok(NamedFit::new(names, results))
}

/// Fit a probit regression from a formula and data.
///
/// The response must be coded `0`/`1`.
pub fn probit(formula: &str, data: &DataFrame) -> Result<NamedFit<DiscreteResults>> {
    let (y, design, names) = require_endog(build(formula, data)?)?;
    let results = solow_discrete::Probit::new(y, design)?.fit()?;
    Ok(NamedFit::new(names, results))
}

/// Fit a Poisson count regression from a formula and data.
///
/// The response must be non-negative counts.
pub fn poisson(formula: &str, data: &DataFrame) -> Result<NamedFit<DiscreteResults>> {
    let (y, design, names) = require_endog(build(formula, data)?)?;
    let results = solow_discrete::Poisson::new(y, design)?.fit()?;
    Ok(NamedFit::new(names, results))
}

// ---------------------------------------------------------------------------
// Labeled summaries (reusing solow-summary)
// ---------------------------------------------------------------------------

use solow_summary::{HeaderStats, RegressionSummary, StatKind};

/// Build the `(lower, upper)` confidence-interval rows from a `k × 2` array.
fn conf_int_pairs(ci: &Array2<f64>) -> Vec<(f64, f64)> {
    (0..ci.nrows()).map(|i| (ci[[i, 0]], ci[[i, 1]])).collect()
}

impl NamedFit<LinearResults> {
    /// A labeled coefficient table with header fit statistics, rendered via
    /// [`solow_summary`]. Coefficient rows are labeled with [`names`](Self::names).
    pub fn summary(&self) -> String {
        let r = &self.results;
        let ci = conf_int_pairs(&r.conf_int(0.05));
        let header = HeaderStats {
            model: Some("OLS".into()),
            nobs: Some(r.nobs),
            df_model: Some(r.df_model),
            df_resid: Some(r.df_resid),
            rsquared: Some(r.rsquared),
            rsquared_adj: Some(r.rsquared_adj),
            fvalue: Some(r.fvalue),
            f_pvalue: Some(r.f_pvalue),
            llf: Some(r.llf),
            aic: Some(r.aic),
            bic: Some(r.bic),
            ..HeaderStats::new()
        };
        RegressionSummary::new(
            &self.names,
            r.params.as_slice().unwrap_or(&[]),
            r.bse.as_slice().unwrap_or(&[]),
            r.tvalues.as_slice().unwrap_or(&[]),
            r.pvalues.as_slice().unwrap_or(&[]),
            &ci,
            header,
        )
        .to_string()
    }
}

impl NamedFit<GlmResults> {
    /// A labeled coefficient table with header fit statistics, rendered via
    /// [`solow_summary`]. GLM inference uses the normal distribution, so the
    /// statistic column is reported as `z`.
    pub fn summary(&self) -> String {
        let r = &self.results;
        let ci = conf_int_pairs(&r.conf_int(0.05));
        let header = HeaderStats {
            model: Some("GLM".into()),
            nobs: Some(r.nobs),
            df_model: Some(r.df_model),
            df_resid: Some(r.df_resid),
            llf: Some(r.llf),
            aic: Some(r.aic),
            bic: Some(r.bic),
            ..HeaderStats::new()
        };
        RegressionSummary::new(
            &self.names,
            r.params.as_slice().unwrap_or(&[]),
            r.bse.as_slice().unwrap_or(&[]),
            r.tvalues.as_slice().unwrap_or(&[]),
            r.pvalues.as_slice().unwrap_or(&[]),
            &ci,
            header,
        )
        .with_stat_kind(StatKind::Z)
        .to_string()
    }
}

impl NamedFit<DiscreteResults> {
    /// A labeled coefficient table with header fit statistics, rendered via
    /// [`solow_summary`]. Discrete-model inference uses the normal
    /// distribution, so the statistic column is reported as `z`.
    pub fn summary(&self) -> String {
        let r = &self.results;
        let ci = conf_int_pairs(&r.conf_int(0.05));
        let header = HeaderStats {
            model: Some("Discrete".into()),
            nobs: Some(r.nobs),
            df_model: Some(r.df_model),
            df_resid: Some(r.df_resid),
            llf: Some(r.llf),
            aic: Some(r.aic),
            bic: Some(r.bic),
            ..HeaderStats::new()
        };
        RegressionSummary::new(
            &self.names,
            r.params.as_slice().unwrap_or(&[]),
            r.bse.as_slice().unwrap_or(&[]),
            r.tvalues.as_slice().unwrap_or(&[]),
            r.pvalues.as_slice().unwrap_or(&[]),
            &ci,
            header,
        )
        .with_stat_kind(StatKind::Z)
        .to_string()
    }
}
