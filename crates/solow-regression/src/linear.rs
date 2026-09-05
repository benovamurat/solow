//! Linear regression models estimated by (generalized) least squares.
//!
//! The estimator follows the canonical whitening formulation: a model carries an
//! implied error covariance, the design and response are *whitened* so that the
//! transformed problem is ordinary least squares, and the pseudoinverse of the
//! whitened design yields the coefficients and the (normalized) covariance.
//!
//! * [`Ols`] — ordinary least squares (spherical errors).
//! * [`Wls`] — weighted least squares (diagonal error covariance `1/weights`).
//! * [`Gls`] — generalized least squares (full error covariance `sigma`).

use crate::robustcov::{self, CovType};
use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{chi2_sf, f_sf, norm_sf, t_ppf, t_sf};
use solow_linalg::{cholesky, eigh, inv, lstsq_qr, pinv};
use std::f64::consts::PI;

/// A linear regression model awaiting estimation.
#[derive(Clone, Debug)]
pub struct LinearModel {
    endog: Array1<f64>,
    exog: Array2<f64>,
    /// Per-observation weights (all ones for OLS/GLS).
    weights: Array1<f64>,
    /// Whitening matrix for GLS: `cholsigmainv` with `Σ⁻¹ = cholsigmainvᵀ cholsigmainv`.
    cholsigmainv: Option<Array2<f64>>,
    /// `0.5 · ln det Σ`, contributing to the GLS log-likelihood.
    half_logdet_sigma: f64,
    k_constant: usize,
}

impl LinearModel {
    fn validate(endog: &Array1<f64>, exog: &Array2<f64>) -> Result<()> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape(format!(
                "endog length {} != exog rows {}",
                endog.len(),
                exog.nrows()
            )));
        }
        if exog.nrows() == 0 {
            return Err(Error::Value("empty design matrix".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        Ok(())
    }

    fn detect_k_constant(exog: &Array2<f64>) -> usize {
        let (_, k) = exog.dim();
        for j in 0..k {
            let col = exog.column(j);
            let Some(&first) = col.iter().next() else {
                continue; // empty column (no rows): not a constant column
            };
            if first != 0.0 && col.iter().all(|&v| v == first) {
                return 1;
            }
        }
        0
    }

    /// Ordinary least squares.
    pub fn ols(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        Self::validate(&endog, &exog)?;
        let weights = Array1::ones(endog.len());
        let k_constant = Self::detect_k_constant(&exog);
        Ok(LinearModel {
            endog,
            exog,
            weights,
            cholsigmainv: None,
            half_logdet_sigma: 0.0,
            k_constant,
        })
    }

    /// Weighted least squares with per-observation `weights` (∝ inverse variance).
    pub fn wls(endog: Array1<f64>, exog: Array2<f64>, weights: Array1<f64>) -> Result<Self> {
        Self::validate(&endog, &exog)?;
        if weights.len() != endog.len() {
            return Err(Error::Shape("weights length != nobs".into()));
        }
        ensure_all_finite(&weights.view(), "weights")?;
        if weights.iter().any(|&w| w <= 0.0) {
            return Err(Error::Value("weights must be positive".into()));
        }
        let k_constant = Self::detect_k_constant(&exog);
        Ok(LinearModel {
            endog,
            exog,
            weights,
            cholsigmainv: None,
            half_logdet_sigma: 0.0,
            k_constant,
        })
    }

    /// Generalized least squares with full error covariance `sigma` (`n × n`, SPD).
    pub fn gls(endog: Array1<f64>, exog: Array2<f64>, sigma: &Array2<f64>) -> Result<Self> {
        Self::validate(&endog, &exog)?;
        let n = endog.len();
        if sigma.dim() != (n, n) {
            return Err(Error::Shape("sigma must be nobs × nobs".into()));
        }
        ensure_all_finite_2d(&sigma.view(), "sigma")?;
        // Σ⁻¹ = Lᵀ L with L = chol(Σ⁻¹)ᵀ; whiten(x) = cholsigmainv · x.
        let sigma_inv = inv(sigma)?;
        let l = cholesky(&sigma_inv)?; // lower, Σ⁻¹ = L Lᵀ
        let cholsigmainv = l.t().to_owned(); // upper
                                             // ln det Σ = -ln det Σ⁻¹ = -2 Σ ln L_ii
        let logdet_sigma_inv: f64 = (0..n).map(|i| l[[i, i]].ln()).sum::<f64>() * 2.0;
        let half_logdet_sigma = -0.5 * logdet_sigma_inv;
        let k_constant = Self::detect_k_constant(&exog);
        Ok(LinearModel {
            endog,
            exog,
            weights: Array1::ones(n),
            cholsigmainv: Some(cholsigmainv),
            half_logdet_sigma,
            k_constant,
        })
    }

    fn is_wls(&self) -> bool {
        self.weights.iter().any(|&w| w != 1.0)
    }

    fn whiten_mat(&self, x: &Array2<f64>) -> Array2<f64> {
        if let Some(ref c) = self.cholsigmainv {
            return c.dot(x);
        }
        if self.is_wls() {
            let mut out = x.clone();
            for i in 0..out.nrows() {
                let s = self.weights[i].sqrt();
                for j in 0..out.ncols() {
                    out[[i, j]] *= s;
                }
            }
            return out;
        }
        x.clone()
    }

    fn whiten_vec(&self, y: &Array1<f64>) -> Array1<f64> {
        if let Some(ref c) = self.cholsigmainv {
            return c.dot(y);
        }
        if self.is_wls() {
            let mut out = y.clone();
            for i in 0..out.len() {
                out[i] *= self.weights[i].sqrt();
            }
            return out;
        }
        y.clone()
    }

    /// Predict `exog · params` in the original (unwhitened) space.
    pub fn predict(&self, params: &Array1<f64>, exog: &Array2<f64>) -> Array1<f64> {
        exog.dot(params)
    }

    /// Estimate the model.
    pub fn fit(&self) -> Result<LinearResults> {
        let nobs = self.endog.len();
        let wexog = self.whiten_mat(&self.exog);
        let wendog = self.whiten_vec(&self.endog);

        // Fast path: a full-column-rank design is solved by Householder QR, which
        // yields the same least-squares solution and normalized covariance as the
        // pseudoinverse but is far faster on tall matrices (no SVD, no m×m Q). For a
        // rank-deficient / ill-conditioned design we fall back to the SVD-based
        // pseudoinverse, which matches the reference's rank handling.
        let (params, normalized_cov_params, rank) = match lstsq_qr(&wexog, &wendog)? {
            Some((p, ncp)) => (p, ncp, wexog.ncols()),
            None => {
                let (pinv_wexog, sv) = pinv(&wexog)?;
                let ncp = pinv_wexog.dot(&pinv_wexog.t());
                let p = pinv_wexog.dot(&wendog);
                // Rank via the singular values (reference convention:
                // tol = max(s) · len(s) · eps, matching matrix_rank(diag(s))).
                let smax = sv.iter().cloned().fold(0.0_f64, f64::max);
                let tol = smax * (sv.len() as f64) * f64::EPSILON;
                let rank = sv.iter().filter(|&&s| s > tol).count();
                (p, ncp, rank)
            }
        };

        Ok(LinearResults::compute(
            self,
            params,
            normalized_cov_params,
            rank,
            nobs,
        ))
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }
    /// Whether the design includes a constant.
    pub fn has_constant(&self) -> bool {
        self.k_constant == 1
    }
}

/// The fitted result of a [`LinearModel`], holding every standard quantity.
#[derive(Clone, Debug)]
pub struct LinearResults {
    /// Estimated coefficients.
    pub params: Array1<f64>,
    /// Standard errors of the coefficients.
    pub bse: Array1<f64>,
    /// t-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values from the t distribution with `df_resid` d.o.f.
    pub pvalues: Array1<f64>,

    /// Number of observations.
    pub nobs: f64,
    /// Rank of the design matrix.
    pub rank: usize,
    /// Whether a constant is present (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom (`rank − k_constant`).
    pub df_model: f64,
    /// Residual degrees of freedom (`nobs − rank`).
    pub df_resid: f64,

    /// Error variance estimate `ssr / df_resid`.
    pub scale: f64,
    /// Sum of squared (whitened) residuals.
    pub ssr: f64,
    /// Total sum of squares about the mean (weighted where applicable).
    pub centered_tss: f64,
    /// Total sum of squares about the origin.
    pub uncentered_tss: f64,
    /// Explained sum of squares.
    pub ess: f64,

    /// Coefficient of determination.
    pub rsquared: f64,
    /// Adjusted coefficient of determination.
    pub rsquared_adj: f64,
    /// Model mean square (`ess / df_model`).
    pub mse_model: f64,
    /// Residual mean square (`ssr / df_resid`).
    pub mse_resid: f64,
    /// Total mean square.
    pub mse_total: f64,
    /// Overall F statistic (all slopes jointly zero).
    pub fvalue: f64,
    /// p-value of the overall F statistic.
    pub f_pvalue: f64,

    /// Gaussian log-likelihood at the estimates.
    pub llf: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian (Schwarz) information criterion.
    pub bic: f64,

    /// Fitted values in the original space.
    pub fittedvalues: Array1<f64>,
    /// Residuals in the original space (`endog − fitted`).
    pub resid: Array1<f64>,
    /// Whitened residuals.
    pub wresid: Array1<f64>,

    /// `cov_params / scale`.
    pub normalized_cov_params: Array2<f64>,
    /// Coefficient covariance matrix `scale · normalized_cov_params`.
    pub cov_params: Array2<f64>,

    /// Whether t/F (rather than normal/chi²) inference is used.
    pub use_t: bool,
}

impl LinearResults {
    fn compute(
        model: &LinearModel,
        params: Array1<f64>,
        normalized_cov_params: Array2<f64>,
        rank: usize,
        nobs_usize: usize,
    ) -> LinearResults {
        let nobs = nobs_usize as f64;
        let k_constant = model.k_constant;
        let df_resid = nobs - rank as f64;
        let df_model = rank as f64 - k_constant as f64;

        let fittedvalues = model.predict(&params, &model.exog);
        let resid = &model.endog - &fittedvalues;

        let wexog = model.whiten_mat(&model.exog);
        let wendog = model.whiten_vec(&model.endog);
        let wresid = &wendog - &wexog.dot(&params);
        let ssr = wresid.dot(&wresid);

        let scale = ssr / df_resid;
        let cov_params = &normalized_cov_params * scale;
        let k = params.len();
        let mut bse = Array1::<f64>::zeros(k);
        for i in 0..k {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let mut pvalues = Array1::<f64>::zeros(k);
        for i in 0..k {
            pvalues[i] = 2.0 * t_sf(tvalues[i].abs(), df_resid);
        }

        // Total sums of squares.
        let (centered_tss, uncentered_tss) = Self::tss(model, &wendog);
        let used_tss = if k_constant == 1 {
            centered_tss
        } else {
            uncentered_tss
        };
        let ess = used_tss - ssr;
        let rsquared = 1.0 - ssr / used_tss;
        let rsquared_adj = 1.0 - (nobs - k_constant as f64) / df_resid * (1.0 - rsquared);

        let mse_model = if df_model > 0.0 {
            ess / df_model
        } else {
            f64::NAN
        };
        let mse_resid = ssr / df_resid;
        let mse_total = used_tss / (nobs - k_constant as f64);
        let fvalue = mse_model / mse_resid;
        let f_pvalue = if df_model > 0.0 {
            f_sf(fvalue, df_model, df_resid)
        } else {
            f64::NAN
        };

        // Gaussian log-likelihood (WLS form; GLS adds −½ ln det Σ).
        let nobs2 = nobs / 2.0;
        let mut llf = -ssr.ln() * nobs2 - (1.0 + (PI / nobs2).ln()) * nobs2;
        if model.is_wls() {
            llf += 0.5 * model.weights.iter().map(|w| w.ln()).sum::<f64>();
        }
        llf -= model.half_logdet_sigma;

        let k_params = df_model + k_constant as f64; // == rank
        let aic = -2.0 * llf + 2.0 * k_params;
        let bic = -2.0 * llf + nobs.ln() * k_params;

        LinearResults {
            params,
            bse,
            tvalues,
            pvalues,
            nobs,
            rank,
            k_constant,
            df_model,
            df_resid,
            scale,
            ssr,
            centered_tss,
            uncentered_tss,
            ess,
            rsquared,
            rsquared_adj,
            mse_model,
            mse_resid,
            mse_total,
            fvalue,
            f_pvalue,
            llf,
            aic,
            bic,
            fittedvalues,
            resid,
            wresid,
            normalized_cov_params,
            cov_params,
            use_t: true,
        }
    }

    fn tss(model: &LinearModel, wendog: &Array1<f64>) -> (f64, f64) {
        let uncentered: f64 = wendog.dot(wendog);
        let centered = if model.cholsigmainv.is_some() {
            // GLS: project the whitened response on the whitened constant.
            let ones = Array1::<f64>::ones(model.endog.len());
            let wones = model.whiten_vec(&ones);
            let denom = wones.dot(&wones);
            let mean = wendog.dot(&wones) / denom;
            let centered = wendog - &(mean * &wones);
            centered.dot(&centered)
        } else if model.is_wls() {
            let wsum: f64 = model.weights.sum();
            let wmean: f64 = model
                .weights
                .iter()
                .zip(model.endog.iter())
                .map(|(w, y)| w * y)
                .sum::<f64>()
                / wsum;
            model
                .weights
                .iter()
                .zip(model.endog.iter())
                .map(|(w, y)| w * (y - wmean).powi(2))
                .sum()
        } else {
            let mean = wendog.sum() / wendog.len() as f64;
            wendog.iter().map(|y| (y - mean).powi(2)).sum()
        };
        (centered, uncentered)
    }

    /// A textual results table (coefficients with inference, plus fit statistics).
    ///
    /// `names`, if given, labels the coefficients; otherwise `x0, x1, …` are used.
    pub fn summary(&self, names: Option<&[&str]>) -> String {
        self.summary_titled("y", "OLS", names)
    }

    /// A full results table in the canonical reference layout — a header block, the
    /// coefficient table, and a normality/autocorrelation diagnostics block — with the
    /// dependent-variable and model labels of your choosing.
    pub fn summary_titled(&self, dep: &str, model: &str, names: Option<&[&str]>) -> String {
        use std::fmt::Write as _;
        const W: usize = 78;
        let bar = "=".repeat(W);
        let dash = "-".repeat(W);
        let k = self.params.len();
        let ci = self.conf_int(0.05);
        let (tlab, plab) = if self.use_t {
            ("t", "P>|t|")
        } else {
            ("z", "P>|z|")
        };

        // Residual diagnostics (computed here so the summary is self-contained).
        let dw = resid_durbin_watson(&self.resid);
        let (skew, kurt) = resid_skew_kurt(&self.resid);
        let jb = (self.nobs / 6.0) * (skew * skew + 0.25 * (kurt - 3.0).powi(2));
        let jb_p = chi2_sf(jb, 2.0);
        let (omni, omni_p) = resid_omnibus(&self.resid);
        let cond = condition_number(&self.normalized_cov_params);
        let (date, time) = utc_now_strings();

        let mut s = String::new();
        // Title (centered).
        let title = format!("{} Regression Results", model);
        let pad = (W.saturating_sub(title.len())) / 2;
        let _ = writeln!(s, "{}{}", " ".repeat(pad), title);
        let _ = writeln!(s, "{bar}");

        // Header: two key/value pairs per line. The left block is a fixed 38
        // columns with the value right-justified, so a 17-char label such as
        // "No. Observations:" does not shove the right column one space over.
        let row = |l1: &str, v1: String, l2: &str, v2: String, s: &mut String| {
            let lw = 38usize.saturating_sub(l1.len());
            let _ = writeln!(s, "{}{:>width$}   {:<19}{:>17}", l1, v1, l2, v2, width = lw);
        };
        row(
            "Dep. Variable:",
            dep.to_string(),
            "R-squared:",
            fmt_f3(self.rsquared),
            &mut s,
        );
        row(
            "Model:",
            model.to_string(),
            "Adj. R-squared:",
            fmt_f3(self.rsquared_adj),
            &mut s,
        );
        row(
            "Method:",
            "Least Squares".into(),
            "F-statistic:",
            fmt_g(self.fvalue, 4),
            &mut s,
        );
        row(
            "Date:",
            date,
            "Prob (F-statistic):",
            fmt_g(self.f_pvalue, 3),
            &mut s,
        );
        row("Time:", time, "Log-Likelihood:", fmt_g(self.llf, 5), &mut s);
        row(
            "No. Observations:",
            format!("{:.0}", self.nobs),
            "AIC:",
            fmt_g(self.aic, 4),
            &mut s,
        );
        row(
            "Df Residuals:",
            format!("{:.0}", self.df_resid),
            "BIC:",
            fmt_g(self.bic, 4),
            &mut s,
        );
        row(
            "Df Model:",
            format!("{:.0}", self.df_model),
            "",
            String::new(),
            &mut s,
        );
        row(
            "Covariance Type:",
            "nonrobust".into(),
            "",
            String::new(),
            &mut s,
        );
        let _ = writeln!(s, "{bar}");

        // Coefficient table.
        let _ = writeln!(
            s,
            "{:<13}{:>10}{:>11}{:>10}{:>10}{:>12}{:>12}",
            "", "coef", "std err", tlab, plab, "[0.025", "0.975]"
        );
        let _ = writeln!(s, "{dash}");
        for i in 0..k {
            let name = names
                .and_then(|n| n.get(i).copied())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("x{i}"));
            let _ = writeln!(
                s,
                "{:<13}{:>10.4}{:>11.3}{:>10.3}{:>10.3}{:>12.3}{:>12.3}",
                name,
                self.params[i],
                self.bse[i],
                self.tvalues[i],
                self.pvalues[i],
                ci[[i, 0]],
                ci[[i, 1]]
            );
        }
        let _ = writeln!(s, "{bar}");

        // Diagnostics block.
        row(
            "Omnibus:",
            fmt_f3(omni),
            "Durbin-Watson:",
            fmt_f3(dw),
            &mut s,
        );
        row(
            "Prob(Omnibus):",
            fmt_f3(omni_p),
            "Jarque-Bera (JB):",
            fmt_f3(jb),
            &mut s,
        );
        row("Skew:", fmt_f3(skew), "Prob(JB):", fmt_f3(jb_p), &mut s);
        row(
            "Kurtosis:",
            fmt_f3(kurt),
            "Cond. No.",
            fmt_g(cond, 3),
            &mut s,
        );
        let _ = writeln!(s, "{bar}");
        let _ = writeln!(s, "\nNotes:");
        let _ = write!(
            s,
            "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
        );
        s
    }

    /// Confidence interval for each coefficient at level `1 − alpha`.
    /// Returns a `k × 2` array of `[lower, upper]` rows.
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        let k = self.params.len();
        let q = if self.use_t {
            t_ppf(1.0 - alpha / 2.0, self.df_resid)
        } else {
            solow_distributions::norm_ppf(1.0 - alpha / 2.0)
        };
        let mut out = Array2::<f64>::zeros((k, 2));
        for i in 0..k {
            out[[i, 0]] = self.params[i] - q * self.bse[i];
            out[[i, 1]] = self.params[i] + q * self.bse[i];
        }
        out
    }

    /// Robust (sandwich) coefficient covariance matrix.
    ///
    /// `exog` is the design used in the fit (for OLS the same matrix passed to
    /// [`LinearModel::ols`]); the residuals `self.resid` supply the score and
    /// leverage terms. Supported `cov_type` variants are HC0–HC3, HAC
    /// (Newey–West, Bartlett kernel), and one-way cluster — see [`CovType`].
    ///
    /// The result reproduces the reference `get_robustcov_results(...).cov_params()`
    /// to closed-form precision for OLS fits. For WLS/GLS, use the free
    /// [`robustcov::robust_cov`] with the whitened design and residuals.
    pub fn cov_params_robust(&self, exog: &Array2<f64>, cov_type: &CovType) -> Result<Array2<f64>> {
        robustcov::robust_cov(
            exog,
            &self.resid,
            &self.normalized_cov_params,
            self.df_resid,
            cov_type,
        )
    }

    /// Robust standard errors: `√diag` of [`cov_params_robust`](Self::cov_params_robust).
    pub fn bse_robust(&self, exog: &Array2<f64>, cov_type: &CovType) -> Result<Array1<f64>> {
        let cov = self.cov_params_robust(exog, cov_type)?;
        Ok(robustcov::bse_from_cov(&cov))
    }

    /// Degrees of freedom used for robust *inference* (t/z and p-values).
    ///
    /// HC0–HC3 and HAC keep `df_resid = n − k`. One-way clustering switches to
    /// `G − 1` (number of groups minus one), mirroring the reference's
    /// `df_resid_inference` under its default `df_correction`.
    fn robust_inference_df(&self, cov_type: &CovType) -> f64 {
        match cov_type {
            CovType::Cluster { groups, .. } => {
                let mut uniq = groups.clone();
                uniq.sort_unstable();
                uniq.dedup();
                (uniq.len() as f64) - 1.0
            }
            _ => self.df_resid,
        }
    }

    /// Robust t-statistics `params / bse_robust` for the given covariance type.
    pub fn tvalues_robust(&self, exog: &Array2<f64>, cov_type: &CovType) -> Result<Array1<f64>> {
        let bse = self.bse_robust(exog, cov_type)?;
        Ok(&self.params / &bse)
    }

    /// Two-sided robust p-values.
    ///
    /// When [`use_t`](Self::use_t) is set, a Student-t distribution with the
    /// inference degrees of freedom is used (`n − k`, or `G − 1` for cluster);
    /// otherwise the standard normal is used.
    pub fn pvalues_robust(&self, exog: &Array2<f64>, cov_type: &CovType) -> Result<Array1<f64>> {
        let t = self.tvalues_robust(exog, cov_type)?;
        let df = self.robust_inference_df(cov_type);
        let mut p = Array1::<f64>::zeros(t.len());
        for i in 0..t.len() {
            p[i] = if self.use_t {
                2.0 * t_sf(t[i].abs(), df)
            } else {
                2.0 * norm_sf(t[i].abs())
            };
        }
        Ok(p)
    }
}

// ---------------------------------------------------------------------------
// Summary helpers — small, self-contained diagnostics and formatting used by
// `LinearResults::summary`. The canonical, reference-verified versions of these
// statistics live in `solow-stats`; this crate carries its own copy because
// `solow-stats` depends on `solow-regression` (so a dependency the other way
// would cycle). The formulas are identical.
// ---------------------------------------------------------------------------

fn central_moment(x: &Array1<f64>, mean: f64, k: i32) -> f64 {
    let n = x.len() as f64;
    x.iter().map(|&v| (v - mean).powi(k)).sum::<f64>() / n
}

fn resid_durbin_watson(resid: &Array1<f64>) -> f64 {
    let n = resid.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut num = 0.0;
    let mut den = resid[0] * resid[0];
    for i in 1..n {
        let d = resid[i] - resid[i - 1];
        num += d * d;
        den += resid[i] * resid[i];
    }
    num / den
}

fn resid_skew_kurt(resid: &Array1<f64>) -> (f64, f64) {
    let n = resid.len() as f64;
    let mean = resid.sum() / n;
    let m2 = central_moment(resid, mean, 2);
    let m3 = central_moment(resid, mean, 3);
    let m4 = central_moment(resid, mean, 4);
    (m3 / m2.powf(1.5), m4 / (m2 * m2))
}

fn skew_z(skew: f64, n: f64) -> f64 {
    let y = skew * (((n + 1.0) * (n + 3.0)) / (6.0 * (n - 2.0))).sqrt();
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let y = if y == 0.0 { 1.0 } else { y };
    let ya = y / alpha;
    delta * (ya + (ya * ya + 1.0).sqrt()).ln()
}

fn kurtosis_z(kurt: f64, n: f64) -> f64 {
    let e = 3.0 * (n - 1.0) / (n + 1.0);
    let varb2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    let x = (kurt - e) / varb2.sqrt();
    let sqrtbeta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a =
        6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + (1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1)).sqrt());
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let denom = 1.0 + x * (2.0 / (a - 4.0)).sqrt();
    let term2 = denom.signum() * ((1.0 - 2.0 / a) / denom.abs()).powf(1.0 / 3.0);
    (term1 - term2) / (2.0 / (9.0 * a)).sqrt()
}

fn resid_omnibus(resid: &Array1<f64>) -> (f64, f64) {
    let n = resid.len();
    if n < 8 {
        return (f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let (skew, kurt) = resid_skew_kurt(resid);
    let zs = skew_z(skew, nf);
    let zk = kurtosis_z(kurt, nf);
    let stat = zs * zs + zk * zk;
    (stat, solow_distributions::chi2_sf(stat, 2.0))
}

/// Condition number of the design = `sqrt(max eig / min eig of XᵀX)`, recovered from
/// `normalized_cov_params = (XᵀX)⁻¹` (so `eig(XᵀX) = 1/eig(ncp)`), matching the reference.
fn condition_number(ncp: &Array2<f64>) -> f64 {
    match eigh(ncp) {
        Ok((w, _)) => {
            let lo = w.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if lo > 0.0 {
                (hi / lo).sqrt()
            } else {
                f64::INFINITY
            }
        }
        Err(_) => f64::NAN,
    }
}

/// Format with `sig` significant figures keeping trailing zeros, matching the
/// reference's `%#.Ng`: scientific when the exponent is `< -4` or `>= sig`, fixed
/// otherwise.
fn fmt_g(v: f64, sig: usize) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return format!("{:.*}", sig.saturating_sub(1), 0.0);
    }
    let exp = v.abs().log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        format!("{:.*e}", sig.saturating_sub(1), v)
    } else {
        let dec = (sig as i32 - 1 - exp).max(0) as usize;
        format!("{:.*}", dec, v)
    }
}

/// Fixed three-decimal formatting (the reference's `%#6.3f`) used for the
/// normality / autocorrelation diagnostics and their p-values.
fn fmt_f3(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    format!("{v:.3}")
}

/// Current UTC date/time as `("Fri, 05 Dec 2025", "18:07:35")`, matching the reference's
/// header look. Uses only `std::time` (no calendar dependency) via the civil-from-days
/// algorithm.
fn utc_now_strings() -> (String, String) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (h, mi, sc) = (tod / 3600, (tod % 3600) / 60, tod % 60);

    // Howard Hinnant's civil_from_days (1970-01-01 == day 0).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let wdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    // 1970-01-01 was a Thursday (index 3).
    let wd = (days.rem_euclid(7) + 3).rem_euclid(7) as usize;
    let date = format!(
        "{}, {:02} {} {}",
        wdays[wd],
        d,
        months[(m as usize - 1).min(11)],
        year
    );
    let time = format!("{h:02}:{mi:02}:{sc:02}");
    (date, time)
}

#[cfg(test)]
mod summary_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn summary_matches_reference_layout() {
        // A small full-rank OLS; the summary must carry every block of the canonical
        // reference layout.
        let x = array![
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.4],
            [1.0, 2.0, -0.3],
            [1.0, 3.0, 0.8],
            [1.0, 4.0, -0.6],
            [1.0, 5.0, 0.2],
            [1.0, 6.0, 0.9],
            [1.0, 7.0, -0.1],
            [1.0, 8.0, 0.5],
            [1.0, 9.0, -0.4],
        ];
        let y = array![1.0, 1.6, 2.1, 3.0, 3.4, 4.1, 5.2, 5.6, 6.5, 6.9];
        let res = LinearModel::ols(y, x).unwrap().fit().unwrap();
        let s = res.summary(Some(&["const", "x1", "x2"]));

        // Header / title.
        assert!(s.contains("OLS Regression Results"));
        assert!(s.contains("R-squared:"));
        assert!(s.contains("Adj. R-squared:"));
        assert!(s.contains("F-statistic:"));
        assert!(s.contains("Log-Likelihood:"));
        assert!(s.contains("AIC:") && s.contains("BIC:"));
        assert!(s.contains("Covariance Type:"));
        // Coefficient table.
        assert!(s.contains("coef") && s.contains("std err") && s.contains("P>|t|"));
        assert!(s.contains("const") && s.contains("x1") && s.contains("x2"));
        // Diagnostics block.
        assert!(s.contains("Omnibus:") && s.contains("Durbin-Watson:"));
        assert!(s.contains("Jarque-Bera (JB):") && s.contains("Prob(JB):"));
        assert!(s.contains("Skew:") && s.contains("Kurtosis:"));
        assert!(s.contains("Cond. No."));
        // Notes.
        assert!(s.contains("Notes:") && s.contains("Standard Errors assume"));
    }
}
