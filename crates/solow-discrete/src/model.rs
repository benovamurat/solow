//! Discrete-choice and count MLE models fit by Newton's method.
//!
//! Three response families share the same Newton driver and result type, differing
//! only in their log-likelihood, score, and Hessian:
//!
//! * [`Logit`] — Bernoulli response, logistic link `p = 1/(1 + e^{-Xβ})`.
//! * [`Probit`] — Bernoulli response, Gaussian link `p = Φ(Xβ)`.
//! * [`Poisson`] — count response, log link `μ = e^{Xβ}`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{chi2_sf, lgamma, norm_cdf, norm_pdf, norm_ppf, norm_sf};
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// The response/link family selected for a discrete model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Logit,
    Probit,
    Poisson,
}

/// Standard logistic CDF `1/(1 + e^{-x})`, numerically stable in both tails.
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// A discrete-choice / count model awaiting estimation.
///
/// Construct via [`Logit::new`], [`Probit::new`], or [`Poisson::new`]; each is a
/// thin wrapper returning this shared estimator with the right [`Kind`].
#[derive(Clone, Debug)]
pub struct DiscreteModel {
    endog: Array1<f64>,
    exog: Array2<f64>,
    kind: Kind,
    k_constant: usize,
    maxiter: usize,
    gtol: f64,
}

/// Binary logistic regression: `P(y=1) = 1/(1 + e^{-Xβ})`.
#[derive(Clone, Debug)]
pub struct Logit(DiscreteModel);

/// Binary probit regression: `P(y=1) = Φ(Xβ)`.
#[derive(Clone, Debug)]
pub struct Probit(DiscreteModel);

/// Poisson count regression: `E[y] = e^{Xβ}`.
#[derive(Clone, Debug)]
pub struct Poisson(DiscreteModel);

macro_rules! wrapper {
    ($name:ident, $kind:expr, $doc:literal) => {
        impl $name {
            #[doc = $doc]
            ///
            /// `exog` should already contain an intercept column if one is wanted
            /// (see `solow_core::tools::add_constant`). Returns an error when the
            /// endog/exog shapes are inconsistent.
            pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
                Ok($name(DiscreteModel::new(endog, exog, $kind)?))
            }

            /// Estimate by Newton's method and return the fitted results.
            pub fn fit(&self) -> Result<DiscreteResults> {
                self.0.fit()
            }

            /// The underlying generic model.
            pub fn model(&self) -> &DiscreteModel {
                &self.0
            }
        }
    };
}

wrapper!(Logit, Kind::Logit, "Build a logistic regression model.");
wrapper!(Probit, Kind::Probit, "Build a probit regression model.");
wrapper!(Poisson, Kind::Poisson, "Build a Poisson regression model.");

impl DiscreteModel {
    fn new(endog: Array1<f64>, exog: Array2<f64>, kind: Kind) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        let k_constant = detect_k_constant(&exog);
        Ok(DiscreteModel {
            endog,
            exog,
            kind,
            k_constant,
            maxiter: 100,
            gtol: 1e-12,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Log-likelihood at coefficient vector `beta`.
    fn loglike(&self, beta: &Array1<f64>) -> f64 {
        let eta = self.exog.dot(beta);
        let y = &self.endog;
        let mut ll = 0.0;
        match self.kind {
            Kind::Logit => {
                for i in 0..y.len() {
                    // log p if y=1, log(1-p) if y=0; written via -log(1+e^{∓η}).
                    let e = eta[i];
                    let q = 2.0 * y[i] - 1.0; // +1 / -1
                    ll += -softplus(-q * e);
                }
            }
            Kind::Probit => {
                for i in 0..y.len() {
                    let q = 2.0 * y[i] - 1.0;
                    ll += norm_cdf(q * eta[i]).max(1e-300).ln();
                }
            }
            Kind::Poisson => {
                for i in 0..y.len() {
                    let mu = eta[i].exp();
                    ll += -mu + y[i] * eta[i] - lgamma(y[i] + 1.0);
                }
            }
        }
        ll
    }

    /// Score (gradient of the log-likelihood) at `beta`.
    fn score(&self, beta: &Array1<f64>) -> Array1<f64> {
        let eta = self.exog.dot(beta);
        let n = self.endog.len();
        let mut w = Array1::<f64>::zeros(n); // residual-like weight per obs
        match self.kind {
            Kind::Logit => {
                for i in 0..n {
                    w[i] = self.endog[i] - logistic(eta[i]);
                }
            }
            Kind::Probit => {
                for i in 0..n {
                    let xb = eta[i];
                    let phi = norm_pdf(xb);
                    let cdf = norm_cdf(xb);
                    // λ = y φ/Φ − (1−y) φ/(1−Φ)
                    let lam = if self.endog[i] > 0.5 {
                        phi / cdf.max(1e-300)
                    } else {
                        -phi / (1.0 - cdf).max(1e-300)
                    };
                    w[i] = lam;
                }
            }
            Kind::Poisson => {
                for i in 0..n {
                    w[i] = self.endog[i] - eta[i].exp();
                }
            }
        }
        self.exog.t().dot(&w)
    }

    /// Hessian of the log-likelihood at `beta` (negative definite at the optimum).
    fn hessian(&self, beta: &Array1<f64>) -> Array2<f64> {
        let eta = self.exog.dot(beta);
        let n = self.endog.len();
        let p = self.exog.ncols();
        let mut d = Array1::<f64>::zeros(n); // diagonal weight: H = -Xᵀ diag(d) X
        match self.kind {
            Kind::Logit => {
                for i in 0..n {
                    let pr = logistic(eta[i]);
                    d[i] = pr * (1.0 - pr);
                }
            }
            Kind::Probit => {
                for i in 0..n {
                    let xb = eta[i];
                    let phi = norm_pdf(xb);
                    let cdf = norm_cdf(xb);
                    let lam = if self.endog[i] > 0.5 {
                        phi / cdf.max(1e-300)
                    } else {
                        -phi / (1.0 - cdf).max(1e-300)
                    };
                    // -∂²ℓ/∂η² = λ (λ + xb)  (positive); see standard probit results.
                    d[i] = lam * (lam + xb);
                }
            }
            Kind::Poisson => {
                for i in 0..n {
                    d[i] = eta[i].exp();
                }
            }
        }
        // H = -Xᵀ diag(d) X.
        let mut h = Array2::<f64>::zeros((p, p));
        for a in 0..p {
            for b in a..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += self.exog[[i, a]] * d[i] * self.exog[[i, b]];
                }
                h[[a, b]] = -s;
                h[[b, a]] = -s;
            }
        }
        h
    }

    /// Reasonable Newton starting point: zeros work well for all three families.
    fn start(&self) -> Array1<f64> {
        Array1::<f64>::zeros(self.exog.ncols())
    }

    /// The intercept-only (null) log-likelihood.
    ///
    /// With a constant in the model the MLE of the intercept-only model produces a
    /// constant fitted mean equal to ȳ, so the null log-likelihood is available in
    /// closed form without a second fit.
    fn llnull(&self) -> f64 {
        let n = self.endog.len() as f64;
        let ybar = self.endog.sum() / n;
        match self.kind {
            Kind::Logit | Kind::Probit => {
                // Σ log p̂ where p̂ = ȳ; equals n[ȳ log ȳ + (1−ȳ) log(1−ȳ)].
                let mut ll = 0.0;
                for &y in self.endog.iter() {
                    let p = ybar;
                    ll += y * p.max(1e-300).ln() + (1.0 - y) * (1.0 - p).max(1e-300).ln();
                }
                ll
            }
            Kind::Poisson => {
                // μ̂ = ȳ for every observation.
                let mut ll = 0.0;
                for &y in self.endog.iter() {
                    ll += -ybar + y * ybar.ln() - lgamma(y + 1.0);
                }
                ll
            }
        }
    }

    /// Estimate the model by full Newton steps and assemble [`DiscreteResults`].
    pub fn fit(&self) -> Result<DiscreteResults> {
        let fgh = |b: &Array1<f64>| {
            // Newton minimizes; minimize the negative log-likelihood so the
            // stationary point is the MLE. score/hessian are of +ℓ, so negate.
            let f = -self.loglike(b);
            let g = self.score(b).mapv(|v| -v);
            let h = self.hessian(b).mapv(|v| -v);
            (f, g, h)
        };
        let opt = newton_stationary(&self.start(), fgh, self.maxiter, self.gtol)?;
        let params = opt.x;

        // Covariance = (-H)^{-1} evaluated at the optimum (observed information).
        let h = self.hessian(&params);
        let neg_h = h.mapv(|v| -v);
        let cov = inv(&neg_h)?;

        Ok(DiscreteResults::new(self, params, cov, opt.converged))
    }
}

/// `log(1 + e^x)`, evaluated without overflow.
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
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

/// The fitted result of a discrete-choice / count model.
#[derive(Clone, Debug)]
pub struct DiscreteResults {
    /// Estimated coefficients `β̂`.
    pub params: Array1<f64>,
    /// Standard errors `√diag((−H)^{-1})`.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`.
    pub pvalues: Array1<f64>,

    /// Number of observations.
    pub nobs: f64,
    /// Whether a constant is present (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom (regressors excluding the constant).
    pub df_model: f64,
    /// Residual degrees of freedom (`nobs − df_model − 1`).
    pub df_resid: f64,

    /// Maximized log-likelihood.
    pub llf: f64,
    /// Intercept-only (null) log-likelihood.
    pub llnull: f64,
    /// Likelihood-ratio statistic `2(llf − llnull)`.
    pub llr: f64,
    /// p-value of the LR statistic, `χ²_{df_model}` survival.
    pub llr_pvalue: f64,
    /// McFadden's pseudo-R², `1 − llf/llnull`.
    pub prsquared: f64,

    /// Akaike information criterion `−2 llf + 2(df_model + 1)`.
    pub aic: f64,
    /// Bayesian information criterion `−2 llf + (df_model + 1) ln n`.
    pub bic: f64,

    /// Fitted linear predictor `Xβ̂` (matches the reference `fittedvalues`).
    pub fittedvalues: Array1<f64>,
    /// Predicted probabilities (Logit/Probit) or means (Poisson).
    pub predicted: Array1<f64>,

    /// Coefficient covariance `(−H)^{-1}`.
    pub cov_params: Array2<f64>,

    /// Whether Newton converged.
    pub converged: bool,

    /// Model display name (`"Logit"`, `"Probit"`, or `"Poisson"`), for the
    /// summary title and `Model:` field.
    pub model_name: &'static str,
}

impl DiscreteResults {
    fn new(
        model: &DiscreteModel,
        params: Array1<f64>,
        cov_params: Array2<f64>,
        converged: bool,
    ) -> DiscreteResults {
        let nobs = model.endog.len() as f64;
        let k = params.len();
        let k_constant = model.k_constant;
        let df_model = k as f64 - k_constant as f64;
        let df_resid = nobs - df_model - 1.0;

        let mut bse = Array1::<f64>::zeros(k);
        for i in 0..k {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        let llf = model.loglike(&params);
        let llnull = model.llnull();
        let llr = 2.0 * (llf - llnull);
        let llr_pvalue = chi2_sf(llr, df_model);
        let prsquared = 1.0 - llf / llnull;

        let k_params = df_model + 1.0;
        let aic = -2.0 * llf + 2.0 * k_params;
        let bic = -2.0 * llf + k_params * nobs.ln();

        let eta = model.exog.dot(&params);
        let predicted = match model.kind {
            Kind::Logit => eta.mapv(logistic),
            Kind::Probit => eta.mapv(norm_cdf),
            Kind::Poisson => eta.mapv(f64::exp),
        };

        DiscreteResults {
            params,
            bse,
            tvalues,
            pvalues,
            nobs,
            k_constant,
            df_model,
            df_resid,
            llf,
            llnull,
            llr,
            llr_pvalue,
            prsquared,
            aic,
            bic,
            fittedvalues: eta,
            predicted,
            cov_params,
            converged,
            model_name: match model.kind {
                Kind::Logit => "Logit",
                Kind::Probit => "Probit",
                Kind::Poisson => "Poisson",
            },
        }
    }

    /// Confidence interval for each coefficient at level `1 − alpha` (normal).
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        let q = norm_ppf(1.0 - alpha / 2.0);
        let k = self.params.len();
        let mut out = Array2::<f64>::zeros((k, 2));
        for i in 0..k {
            out[[i, 0]] = self.params[i] - q * self.bse[i];
            out[[i, 1]] = self.params[i] + q * self.bse[i];
        }
        out
    }

    /// A full results table in the canonical reference discrete layout: the
    /// "`<Model>` Regression Results" header block followed by the coefficient
    /// table (with `z`, `P>|z|`, and the 95 % confidence interval).
    ///
    /// `names`, if given, labels the coefficients; otherwise `x0, x1, …` are
    /// used. The output matches the reference field-for-field (the only volatile
    /// fields are the `Date:`/`Time:` stamps).
    pub fn summary(&self, names: Option<&[&str]>) -> String {
        self.summary_titled("y", names)
    }

    /// As [`summary`](Self::summary), with the dependent-variable label of your
    /// choosing.
    pub fn summary_titled(&self, dep: &str, names: Option<&[&str]>) -> String {
        use crate::summary::{centered, coef_header, coef_row, fmt_g, header_row, utc_now_strings};
        use std::fmt::Write as _;
        const W: usize = 78;
        let bar = "=".repeat(W);
        let dash = "-".repeat(W);
        let k = self.params.len();
        let ci = self.conf_int(0.05);
        let (date, time) = utc_now_strings();

        let mut s = String::new();
        let title = format!("{} Regression Results", self.model_name);
        let _ = writeln!(s, "{}", centered(&title, W));
        let _ = writeln!(s, "{bar}");

        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Dep. Variable:",
                dep,
                "No. Observations:",
                &format!("{:.0}", self.nobs)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Model:",
                self.model_name,
                "Df Residuals:",
                &format!("{:.0}", self.df_resid)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Method:",
                "MLE",
                "Df Model:",
                &format!("{:.0}", self.df_model)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row("Date:", &date, "Pseudo R-squ.:", &fmt_g(self.prsquared, 4))
        );
        let _ = writeln!(
            s,
            "{}",
            header_row("Time:", &time, "Log-Likelihood:", &fmt_g(self.llf, 5))
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "converged:",
                if self.converged { "True" } else { "False" },
                "LL-Null:",
                &fmt_g(self.llnull, 5)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Covariance Type:",
                "nonrobust",
                "LLR p-value:",
                &fmt_g(self.llr_pvalue, 4)
            )
        );
        let _ = writeln!(s, "{bar}");

        // Coefficient table.
        let _ = writeln!(s, "{}", coef_header());
        let _ = writeln!(s, "{dash}");
        for i in 0..k {
            let name = names
                .and_then(|n| n.get(i).copied())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("x{i}"));
            let _ = writeln!(
                s,
                "{}",
                coef_row(
                    &name,
                    self.params[i],
                    self.bse[i],
                    self.tvalues[i],
                    self.pvalues[i],
                    ci[[i, 0]],
                    ci[[i, 1]]
                )
            );
        }
        let _ = write!(s, "{bar}");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn design() -> (Array1<f64>, Array2<f64>) {
        let xcol = array![0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6];
        let mut x = Array2::<f64>::ones((10, 2));
        x.column_mut(1).assign(&xcol);
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        (y, x)
    }

    #[test]
    fn logit_score_zero_at_optimum() {
        let (y, x) = design();
        let m = Logit::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let g = m.0.score(&res.params);
        let gnorm = g.dot(&g).sqrt();
        assert!(gnorm < 1e-9, "score not zero: {gnorm}");
    }

    #[test]
    fn poisson_score_zero_at_optimum() {
        let xcol = array![0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6];
        let mut x = Array2::<f64>::ones((10, 2));
        x.column_mut(1).assign(&xcol);
        let y = array![2.0, 0.0, 3.0, 8.0, 1.0, 5.0, 1.0, 4.0, 0.0, 3.0];
        let m = Poisson::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let g = m.0.score(&res.params);
        assert!(g.dot(&g).sqrt() < 1e-9);
    }

    #[test]
    fn analytic_hessian_matches_finite_difference() {
        // Compare analytic Hessian to a central difference of the score.
        let (y, x) = design();
        let m = Probit::new(y, x).unwrap();
        let b = array![0.1, -0.3];
        let h_an = m.0.hessian(&b);
        let eps = 1e-6;
        let p = b.len();
        for j in 0..p {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[j] += eps;
            bm[j] -= eps;
            let gp = m.0.score(&bp);
            let gm = m.0.score(&bm);
            for i in 0..p {
                let fd = (gp[i] - gm[i]) / (2.0 * eps);
                assert_abs_diff_eq!(h_an[[i, j]], fd, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn logit_llr_nonnegative() {
        let (y, x) = design();
        let res = Logit::new(y, x).unwrap().fit().unwrap();
        assert!(res.llr >= -1e-9);
        assert!(res.prsquared >= 0.0 && res.prsquared <= 1.0);
    }

    /// The summary must carry every block/label of the reference's discrete
    /// layout, for each of the three families, with z-inference (no t/F block).
    #[test]
    fn summary_has_every_block() {
        let (y, x) = design();
        for (name, s) in [
            (
                "Logit",
                Logit::new(y.clone(), x.clone())
                    .unwrap()
                    .fit()
                    .unwrap()
                    .summary(Some(&["const", "x1"])),
            ),
            (
                "Probit",
                Probit::new(y.clone(), x.clone())
                    .unwrap()
                    .fit()
                    .unwrap()
                    .summary(Some(&["const", "x1"])),
            ),
        ] {
            assert!(
                s.contains(&format!("{name} Regression Results")),
                "{name} title"
            );
            assert!(s.contains("Dep. Variable:"));
            assert!(s.contains("Model:") && s.contains(name));
            assert!(s.contains("Method:") && s.contains("MLE"));
            assert!(s.contains("Date:") && s.contains("Time:"));
            assert!(s.contains("converged:"));
            assert!(s.contains("No. Observations:"));
            assert!(s.contains("Df Residuals:") && s.contains("Df Model:"));
            assert!(s.contains("Pseudo R-squ.:"));
            assert!(s.contains("Log-Likelihood:"));
            assert!(s.contains("LL-Null:"));
            assert!(s.contains("LLR p-value:"));
            assert!(s.contains("Covariance Type:") && s.contains("nonrobust"));
            // Coefficient table: z / P>|z| (no t), CI columns, labels.
            assert!(s.contains("coef") && s.contains("std err"));
            assert!(s.contains("P>|z|") && !s.contains("P>|t|"));
            assert!(s.contains("[0.025") && s.contains("0.975]"));
            assert!(s.contains("const") && s.contains("x1"));
            // No OLS-style normality / autocorrelation block.
            assert!(!s.contains("Omnibus") && !s.contains("Durbin-Watson"));
        }

        // Poisson family carries the same blocks under its own title.
        let yp = array![2.0, 0.0, 3.0, 8.0, 1.0, 5.0, 1.0, 4.0, 0.0, 3.0];
        let sp = Poisson::new(yp, x)
            .unwrap()
            .fit()
            .unwrap()
            .summary(Some(&["const", "x1"]));
        assert!(sp.contains("Poisson Regression Results"));
        assert!(sp.contains("Method:") && sp.contains("MLE"));
        assert!(sp.contains("LLR p-value:") && sp.contains("P>|z|"));
    }
}
