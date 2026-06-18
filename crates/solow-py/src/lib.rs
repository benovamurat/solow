//! Python bindings for Solow (PyO3).
//!
//! Exposes a small, NumPy-friendly surface over the Solow modeling stack:
//!
//! * [`OLS`] / [`WLS`] / [`GLS`] — (generalized/weighted) least squares
//! * [`GLM`] — generalized linear models (Gaussian / Binomial / Poisson / Gamma)
//! * [`Logit`] / [`Probit`] / [`Poisson`] — maximum-likelihood discrete & count models
//! * [`AutoReg`] plus [`acf`] / [`pacf`] — time-series autoregression & diagnostics
//!
//! All estimators accept `float64` NumPy arrays (`endog` 1-D, `exog` 2-D where the
//! caller supplies any intercept column themselves, exactly like the canonical
//! reference) and `.fit()` returns a results object whose attributes are NumPy
//! arrays / Python floats. Rust `Result` errors are surfaced as `ValueError`.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::solow::discrete::{Logit as RsLogit, Poisson as RsPoisson, Probit as RsProbit};
use ::solow::glm::{Family, Glm as RsGlm, Link};
use ::solow::regression::{CovType, LinearModel};
use ::solow::tsa::{acf as rs_acf, pacf as rs_pacf, AutoReg as RsAutoReg, PacfMethod, Trend};

/// Convert any Solow error into a Python `ValueError`.
fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Copy a 1-D read-only NumPy view into an owned `Array1<f64>`.
fn vec1(a: &PyReadonlyArray1<f64>) -> Array1<f64> {
    a.as_array().to_owned()
}

/// Copy a 2-D read-only NumPy view into an owned `Array2<f64>`.
fn mat2(a: &PyReadonlyArray2<f64>) -> Array2<f64> {
    a.as_array().to_owned()
}

/// Build a [`CovType`] from a `(cov_type, **kwargs)` request.
///
/// * `"hc0"`/`"hc1"`/`"hc2"`/`"hc3"` — heteroskedasticity-consistent.
/// * `"hac"` — Newey–West; `maxlags` (required) and `use_correction` (default
///   `false`, matching the reference `HAC` default).
/// * `"cluster"` — one-way; `groups` (required `int64` array, length nobs) and
///   `use_correction` (default `true`, matching the reference default).
fn parse_cov_type(
    cov_type: &str,
    maxlags: Option<usize>,
    groups: Option<Vec<i64>>,
    use_correction: Option<bool>,
) -> PyResult<CovType> {
    match cov_type.to_ascii_lowercase().as_str() {
        "hc0" => Ok(CovType::Hc0),
        "hc1" => Ok(CovType::Hc1),
        "hc2" => Ok(CovType::Hc2),
        "hc3" => Ok(CovType::Hc3),
        "hac" => {
            let maxlags = maxlags.ok_or_else(|| {
                PyValueError::new_err("cov_type 'hac' requires the 'maxlags' keyword")
            })?;
            Ok(CovType::Hac {
                maxlags,
                use_correction: use_correction.unwrap_or(false),
            })
        }
        "cluster" => {
            let groups = groups.ok_or_else(|| {
                PyValueError::new_err("cov_type 'cluster' requires the 'groups' keyword")
            })?;
            Ok(CovType::Cluster {
                groups,
                use_correction: use_correction.unwrap_or(true),
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown cov_type '{other}' (expected hc0|hc1|hc2|hc3|hac|cluster)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// OLS / WLS / GLS (linear least squares)
// ---------------------------------------------------------------------------

/// Build an [`OLSResults`] from a fitted Solow [`LinearResults`].
///
/// Robust-covariance inputs (`robust_exog` / `robust_resid`) are left unset; the
/// OLS path fills them in afterwards, since the reference only offers robust
/// standard errors for plain OLS fits.
fn make_linear_results(r: &::solow::regression::LinearResults) -> OLSResults {
    OLSResults {
        params: r.params.clone(),
        bse: r.bse.clone(),
        tvalues: r.tvalues.clone(),
        pvalues: r.pvalues.clone(),
        conf_int_95: r.conf_int(0.05),
        fittedvalues: r.fittedvalues.clone(),
        resid: r.resid.clone(),
        rsquared: r.rsquared,
        rsquared_adj: r.rsquared_adj,
        fvalue: r.fvalue,
        f_pvalue: r.f_pvalue,
        aic: r.aic,
        bic: r.bic,
        llf: r.llf,
        scale: r.scale,
        nobs: r.nobs,
        df_model: r.df_model,
        df_resid: r.df_resid,
        normalized_cov_params: r.normalized_cov_params.clone(),
        robust_exog: None,
        robust_resid: None,
    }
}

/// Ordinary least squares: `OLS(endog, exog).fit()`.
///
/// `exog` must already include any intercept column (matching the reference's
/// `OLS` convention, which does not add one automatically).
#[pyclass]
struct OLS {
    endog: Array1<f64>,
    exog: Array2<f64>,
}

#[pymethods]
impl OLS {
    #[new]
    fn new(endog: PyReadonlyArray1<f64>, exog: PyReadonlyArray2<f64>) -> Self {
        OLS {
            endog: vec1(&endog),
            exog: mat2(&exog),
        }
    }

    /// Fit by least squares and return an [`OLSResults`].
    fn fit(&self) -> PyResult<OLSResults> {
        let model = LinearModel::ols(self.endog.clone(), self.exog.clone()).map_err(err)?;
        let r = model.fit().map_err(err)?;
        let mut out = make_linear_results(&r);
        // Robust SEs for OLS use the raw design and raw residual.
        out.robust_exog = Some(self.exog.clone());
        out.robust_resid = Some(r.resid.clone());
        Ok(out)
    }
}

/// Weighted least squares: `WLS(endog, exog, weights).fit()`.
///
/// `weights` are per-observation, proportional to the inverse error variance
/// (must be positive), matching the reference `WLS`.
#[pyclass]
struct WLS {
    endog: Array1<f64>,
    exog: Array2<f64>,
    weights: Array1<f64>,
}

#[pymethods]
impl WLS {
    #[new]
    fn new(
        endog: PyReadonlyArray1<f64>,
        exog: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
    ) -> Self {
        WLS {
            endog: vec1(&endog),
            exog: mat2(&exog),
            weights: vec1(&weights),
        }
    }

    /// Fit by weighted least squares and return an [`OLSResults`].
    fn fit(&self) -> PyResult<OLSResults> {
        let model = LinearModel::wls(self.endog.clone(), self.exog.clone(), self.weights.clone())
            .map_err(err)?;
        Ok(make_linear_results(&model.fit().map_err(err)?))
    }
}

/// Generalized least squares: `GLS(endog, exog, sigma).fit()`.
///
/// `sigma` is the full `nobs × nobs` error covariance (symmetric positive
/// definite), matching the reference `GLS`.
#[pyclass]
struct GLS {
    endog: Array1<f64>,
    exog: Array2<f64>,
    sigma: Array2<f64>,
}

#[pymethods]
impl GLS {
    #[new]
    fn new(
        endog: PyReadonlyArray1<f64>,
        exog: PyReadonlyArray2<f64>,
        sigma: PyReadonlyArray2<f64>,
    ) -> Self {
        GLS {
            endog: vec1(&endog),
            exog: mat2(&exog),
            sigma: mat2(&sigma),
        }
    }

    /// Fit by generalized least squares and return an [`OLSResults`].
    fn fit(&self) -> PyResult<OLSResults> {
        let model =
            LinearModel::gls(self.endog.clone(), self.exog.clone(), &self.sigma).map_err(err)?;
        Ok(make_linear_results(&model.fit().map_err(err)?))
    }
}

/// Fitted least-squares results (OLS / WLS / GLS). All array attributes are
/// NumPy `float64` arrays.
#[pyclass]
struct OLSResults {
    params: Array1<f64>,
    bse: Array1<f64>,
    tvalues: Array1<f64>,
    pvalues: Array1<f64>,
    conf_int_95: Array2<f64>,
    fittedvalues: Array1<f64>,
    resid: Array1<f64>,
    rsquared: f64,
    rsquared_adj: f64,
    fvalue: f64,
    f_pvalue: f64,
    aic: f64,
    bic: f64,
    llf: f64,
    scale: f64,
    nobs: f64,
    df_model: f64,
    df_resid: f64,
    normalized_cov_params: Array2<f64>,
    /// Design + residual for robust covariances (set only for OLS fits).
    robust_exog: Option<Array2<f64>>,
    robust_resid: Option<Array1<f64>>,
}

#[pymethods]
impl OLSResults {
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.params.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn bse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bse.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn tvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.tvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn pvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.pvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn fittedvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fittedvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn resid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.resid.clone().into_pyarray_bound(py)
    }
    /// 95% confidence interval (alpha = 0.05) as an `(k, 2)` array.
    #[getter]
    fn conf_int<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.conf_int_95.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn rsquared(&self) -> f64 {
        self.rsquared
    }
    #[getter]
    fn rsquared_adj(&self) -> f64 {
        self.rsquared_adj
    }
    #[getter]
    fn fvalue(&self) -> f64 {
        self.fvalue
    }
    #[getter]
    fn f_pvalue(&self) -> f64 {
        self.f_pvalue
    }
    #[getter]
    fn aic(&self) -> f64 {
        self.aic
    }
    #[getter]
    fn bic(&self) -> f64 {
        self.bic
    }
    #[getter]
    fn llf(&self) -> f64 {
        self.llf
    }
    #[getter]
    fn nobs(&self) -> f64 {
        self.nobs
    }
    #[getter]
    fn df_model(&self) -> f64 {
        self.df_model
    }
    #[getter]
    fn df_resid(&self) -> f64 {
        self.df_resid
    }
    #[getter]
    fn scale(&self) -> f64 {
        self.scale
    }

    /// Robust (sandwich) coefficient covariance, returned as a `(k, k)` array.
    ///
    /// `cov_type` is one of `hc0|hc1|hc2|hc3|hac|cluster`. HAC requires
    /// `maxlags`; cluster requires `groups` (an `int64` array of length nobs).
    /// `use_correction` toggles the small-sample factor (defaults match the
    /// reference: `False` for HAC, `True` for cluster). Available for OLS fits.
    #[pyo3(signature = (cov_type, maxlags=None, groups=None, use_correction=None))]
    fn cov_params_robust<'py>(
        &self,
        py: Python<'py>,
        cov_type: &str,
        maxlags: Option<usize>,
        groups: Option<PyReadonlyArray1<i64>>,
        use_correction: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let cov = self.robust_cov_inner(cov_type, maxlags, groups, use_correction)?;
        Ok(cov.into_pyarray_bound(py))
    }

    /// Robust standard errors: `√diag` of [`cov_params_robust`](Self::cov_params_robust).
    ///
    /// `OLS(y, X).fit().bse_robust("HC1")`, etc. See [`cov_params_robust`]
    /// (Self::cov_params_robust) for the `cov_type` and keyword semantics.
    #[pyo3(signature = (cov_type, maxlags=None, groups=None, use_correction=None))]
    fn bse_robust<'py>(
        &self,
        py: Python<'py>,
        cov_type: &str,
        maxlags: Option<usize>,
        groups: Option<PyReadonlyArray1<i64>>,
        use_correction: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let cov = self.robust_cov_inner(cov_type, maxlags, groups, use_correction)?;
        Ok(::solow::regression::bse_from_cov(&cov).into_pyarray_bound(py))
    }
}

impl OLSResults {
    /// Shared robust-covariance computation backing the two public methods.
    fn robust_cov_inner(
        &self,
        cov_type: &str,
        maxlags: Option<usize>,
        groups: Option<PyReadonlyArray1<i64>>,
        use_correction: Option<bool>,
    ) -> PyResult<Array2<f64>> {
        let exog = self.robust_exog.as_ref().ok_or_else(|| {
            PyValueError::new_err("robust standard errors are only available for OLS fits")
        })?;
        let resid = self
            .robust_resid
            .as_ref()
            .expect("OLS fit sets robust_resid");
        let groups = groups.map(|g| g.as_array().to_vec());
        let ct = parse_cov_type(cov_type, maxlags, groups, use_correction)?;
        ::solow::regression::robust_cov(
            exog,
            resid,
            &self.normalized_cov_params,
            self.df_resid,
            &ct,
        )
        .map_err(err)
    }
}

// ---------------------------------------------------------------------------
// GLM
// ---------------------------------------------------------------------------

/// Parse a family name (case-insensitive) into a Solow [`Family`].
fn parse_family(name: &str) -> PyResult<Family> {
    match name.to_ascii_lowercase().as_str() {
        "gaussian" | "normal" => Ok(Family::Gaussian),
        "binomial" | "logistic" => Ok(Family::Binomial),
        "poisson" => Ok(Family::Poisson),
        "gamma" => Ok(Family::Gamma),
        "inversegaussian" | "inverse_gaussian" => Ok(Family::InverseGaussian),
        other => Err(PyValueError::new_err(format!(
            "unknown family '{other}' (expected gaussian|binomial|poisson|gamma|inversegaussian)"
        ))),
    }
}

/// Parse an optional link name into a Solow [`Link`].
fn parse_link(name: &str) -> PyResult<Link> {
    match name.to_ascii_lowercase().as_str() {
        "identity" => Ok(Link::Identity),
        "log" => Ok(Link::Log),
        "logit" => Ok(Link::Logit),
        "probit" => Ok(Link::Probit),
        "cloglog" => Ok(Link::CLogLog),
        "inverse" | "inversepower" | "inverse_power" => Ok(Link::InversePower),
        "inversesquared" | "inverse_squared" => Ok(Link::InverseSquared),
        "sqrt" => Ok(Link::Sqrt),
        other => Err(PyValueError::new_err(format!("unknown link '{other}'"))),
    }
}

/// Generalized linear model: `GLM(endog, exog, family="poisson", link=None).fit()`.
///
/// `exog` includes any intercept column. `family` is one of
/// `gaussian|binomial|poisson|gamma|inversegaussian`. `link` is optional and
/// defaults to the family's canonical link.
#[pyclass]
struct GLM {
    endog: Array1<f64>,
    exog: Array2<f64>,
    family: Family,
    link: Link,
}

#[pymethods]
impl GLM {
    #[new]
    #[pyo3(signature = (endog, exog, family="gaussian", link=None))]
    fn new(
        endog: PyReadonlyArray1<f64>,
        exog: PyReadonlyArray2<f64>,
        family: &str,
        link: Option<&str>,
    ) -> PyResult<Self> {
        let fam = parse_family(family)?;
        let lnk = match link {
            Some(l) => parse_link(l)?,
            None => fam.default_link(),
        };
        Ok(GLM {
            endog: vec1(&endog),
            exog: mat2(&exog),
            family: fam,
            link: lnk,
        })
    }

    /// Fit by IRLS and return a [`GLMResults`].
    fn fit(&self) -> PyResult<GLMResults> {
        let model = RsGlm::with_link(
            self.endog.clone(),
            self.exog.clone(),
            self.family,
            self.link,
        )
        .map_err(err)?;
        let r = model.fit().map_err(err)?;
        Ok(GLMResults {
            params: r.params.clone(),
            bse: r.bse.clone(),
            deviance: r.deviance,
            llf: r.llf,
            aic: r.aic,
            converged: r.converged,
        })
    }
}

/// Fitted GLM results.
#[pyclass]
struct GLMResults {
    params: Array1<f64>,
    bse: Array1<f64>,
    deviance: f64,
    llf: f64,
    aic: f64,
    converged: bool,
}

#[pymethods]
impl GLMResults {
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.params.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn bse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bse.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn deviance(&self) -> f64 {
        self.deviance
    }
    #[getter]
    fn llf(&self) -> f64 {
        self.llf
    }
    #[getter]
    fn aic(&self) -> f64 {
        self.aic
    }
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
}

// ---------------------------------------------------------------------------
// Discrete (Logit / Poisson)
// ---------------------------------------------------------------------------

/// Maximum-likelihood discrete-choice or count results.
#[pyclass]
struct DiscreteResults {
    params: Array1<f64>,
    bse: Array1<f64>,
    tvalues: Array1<f64>,
    pvalues: Array1<f64>,
    llf: f64,
    aic: f64,
    bic: f64,
    converged: bool,
}

#[pymethods]
impl DiscreteResults {
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.params.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn bse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bse.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn tvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.tvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn pvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.pvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn llf(&self) -> f64 {
        self.llf
    }
    #[getter]
    fn aic(&self) -> f64 {
        self.aic
    }
    #[getter]
    fn bic(&self) -> f64 {
        self.bic
    }
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
}

/// Logistic regression: `Logit(endog, exog).fit()`.
#[pyclass]
struct Logit {
    endog: Array1<f64>,
    exog: Array2<f64>,
}

#[pymethods]
impl Logit {
    #[new]
    fn new(endog: PyReadonlyArray1<f64>, exog: PyReadonlyArray2<f64>) -> Self {
        Logit {
            endog: vec1(&endog),
            exog: mat2(&exog),
        }
    }

    fn fit(&self) -> PyResult<DiscreteResults> {
        let model = RsLogit::new(self.endog.clone(), self.exog.clone()).map_err(err)?;
        let r = model.fit().map_err(err)?;
        Ok(DiscreteResults {
            params: r.params.clone(),
            bse: r.bse.clone(),
            tvalues: r.tvalues.clone(),
            pvalues: r.pvalues.clone(),
            llf: r.llf,
            aic: r.aic,
            bic: r.bic,
            converged: r.converged,
        })
    }
}

/// Poisson count regression (MLE): `Poisson(endog, exog).fit()`.
#[pyclass]
struct Poisson {
    endog: Array1<f64>,
    exog: Array2<f64>,
}

#[pymethods]
impl Poisson {
    #[new]
    fn new(endog: PyReadonlyArray1<f64>, exog: PyReadonlyArray2<f64>) -> Self {
        Poisson {
            endog: vec1(&endog),
            exog: mat2(&exog),
        }
    }

    fn fit(&self) -> PyResult<DiscreteResults> {
        let model = RsPoisson::new(self.endog.clone(), self.exog.clone()).map_err(err)?;
        let r = model.fit().map_err(err)?;
        Ok(DiscreteResults {
            params: r.params.clone(),
            bse: r.bse.clone(),
            tvalues: r.tvalues.clone(),
            pvalues: r.pvalues.clone(),
            llf: r.llf,
            aic: r.aic,
            bic: r.bic,
            converged: r.converged,
        })
    }
}

/// Probit binary regression (MLE): `Probit(endog, exog).fit()`.
#[pyclass]
struct Probit {
    endog: Array1<f64>,
    exog: Array2<f64>,
}

#[pymethods]
impl Probit {
    #[new]
    fn new(endog: PyReadonlyArray1<f64>, exog: PyReadonlyArray2<f64>) -> Self {
        Probit {
            endog: vec1(&endog),
            exog: mat2(&exog),
        }
    }

    fn fit(&self) -> PyResult<DiscreteResults> {
        let model = RsProbit::new(self.endog.clone(), self.exog.clone()).map_err(err)?;
        let r = model.fit().map_err(err)?;
        Ok(DiscreteResults {
            params: r.params.clone(),
            bse: r.bse.clone(),
            tvalues: r.tvalues.clone(),
            pvalues: r.pvalues.clone(),
            llf: r.llf,
            aic: r.aic,
            bic: r.bic,
            converged: r.converged,
        })
    }
}

// ---------------------------------------------------------------------------
// Time series: AutoReg, acf, pacf
// ---------------------------------------------------------------------------

/// Parse a trend spec (case-insensitive) into a Solow [`Trend`].
fn parse_trend(name: &str) -> PyResult<Trend> {
    match name.to_ascii_lowercase().as_str() {
        "n" => Ok(Trend::N),
        "c" => Ok(Trend::C),
        "t" => Ok(Trend::T),
        "ct" => Ok(Trend::Ct),
        "ctt" => Ok(Trend::Ctt),
        other => Err(PyValueError::new_err(format!(
            "unknown trend '{other}' (expected n|c|t|ct|ctt)"
        ))),
    }
}

/// Autoregressive model: `AutoReg(endog, lags, trend="c").fit()`.
///
/// Conditional-least-squares AR(`lags`) with optional deterministic trend,
/// matching the reference `AutoReg`. Parameters are ordered
/// `[deterministic..., ar_lag_1, ...]`.
#[pyclass]
struct AutoReg {
    endog: Array1<f64>,
    lags: usize,
    trend: Trend,
}

#[pymethods]
impl AutoReg {
    #[new]
    #[pyo3(signature = (endog, lags, trend="c"))]
    fn new(endog: PyReadonlyArray1<f64>, lags: usize, trend: &str) -> PyResult<Self> {
        Ok(AutoReg {
            endog: vec1(&endog),
            lags,
            trend: parse_trend(trend)?,
        })
    }

    fn fit(&self) -> PyResult<AutoRegResults> {
        let model = RsAutoReg::new(self.endog.clone(), self.lags, self.trend).map_err(err)?;
        let r = model.fit().map_err(err)?;
        Ok(AutoRegResults {
            params: r.params.clone(),
            bse: r.bse.clone(),
            tvalues: r.tvalues.clone(),
            pvalues: r.pvalues.clone(),
            fittedvalues: r.fittedvalues.clone(),
            resid: r.resid.clone(),
            sigma2: r.sigma2,
            llf: r.llf,
            aic: r.aic,
            bic: r.bic,
            hqic: r.hqic,
            nobs: r.nobs as f64,
            df_model: r.df_model as f64,
        })
    }
}

/// Fitted [`AutoReg`] results.
#[pyclass]
struct AutoRegResults {
    params: Array1<f64>,
    bse: Array1<f64>,
    tvalues: Array1<f64>,
    pvalues: Array1<f64>,
    fittedvalues: Array1<f64>,
    resid: Array1<f64>,
    sigma2: f64,
    llf: f64,
    aic: f64,
    bic: f64,
    hqic: f64,
    nobs: f64,
    df_model: f64,
}

#[pymethods]
impl AutoRegResults {
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.params.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn bse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bse.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn tvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.tvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn pvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.pvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn fittedvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fittedvalues.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn resid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.resid.clone().into_pyarray_bound(py)
    }
    #[getter]
    fn sigma2(&self) -> f64 {
        self.sigma2
    }
    #[getter]
    fn llf(&self) -> f64 {
        self.llf
    }
    #[getter]
    fn aic(&self) -> f64 {
        self.aic
    }
    #[getter]
    fn bic(&self) -> f64 {
        self.bic
    }
    #[getter]
    fn hqic(&self) -> f64 {
        self.hqic
    }
    #[getter]
    fn nobs(&self) -> f64 {
        self.nobs
    }
    #[getter]
    fn df_model(&self) -> f64 {
        self.df_model
    }
}

/// Autocorrelation function for lags `0..=nlags` (`acf[0] == 1`).
///
/// Mirrors the reference `acf(x, nlags=nlags, adjusted=adjusted)` with
/// `demean=True`. Returns a length `nlags + 1` NumPy array.
#[pyfunction]
#[pyo3(signature = (x, nlags, adjusted=false))]
fn acf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
    adjusted: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = rs_acf(&vec1(&x), nlags, adjusted).map_err(err)?;
    Ok(a.into_pyarray_bound(py))
}

/// Partial autocorrelation function for lags `0..=nlags` (`pacf[0] == 1`).
///
/// `method` is `"yw"` (Yule–Walker, adjusted) or `"ols"`. Mirrors the reference
/// `pacf(x, nlags=nlags, method=method)`. Returns a length `nlags + 1` array.
#[pyfunction]
#[pyo3(signature = (x, nlags, method="yw"))]
fn pacf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = match method.to_ascii_lowercase().as_str() {
        "yw" | "ywadjusted" | "yule_walker" => PacfMethod::YuleWalker,
        "ols" => PacfMethod::Ols,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pacf method '{other}' (expected yw|ols)"
            )))
        }
    };
    let p = rs_pacf(&vec1(&x), nlags, m).map_err(err)?;
    Ok(p.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// The `solow` Python extension module.
#[pymodule]
fn solow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OLS>()?;
    m.add_class::<WLS>()?;
    m.add_class::<GLS>()?;
    m.add_class::<OLSResults>()?;
    m.add_class::<GLM>()?;
    m.add_class::<GLMResults>()?;
    m.add_class::<Logit>()?;
    m.add_class::<Probit>()?;
    m.add_class::<Poisson>()?;
    m.add_class::<DiscreteResults>()?;
    m.add_class::<AutoReg>()?;
    m.add_class::<AutoRegResults>()?;
    m.add_function(wrap_pyfunction!(acf, m)?)?;
    m.add_function(wrap_pyfunction!(pacf, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
