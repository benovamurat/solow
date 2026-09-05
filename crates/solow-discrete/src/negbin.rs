//! Negative-binomial (NB2) count regression with the dispersion estimated
//! jointly by maximum likelihood.
//!
//! [`NegativeBinomial`] fits the NB2 model: counts `yᵢ` have mean
//! `μᵢ = exp(xᵢ · β)` and variance `μᵢ + α μᵢ²`, where `α > 0` is an
//! over-dispersion parameter estimated alongside the regression coefficients.
//! Writing `r = 1/α`, the per-observation log-likelihood is
//!
//! ```text
//! ℓᵢ = lnΓ(r + yᵢ) − lnΓ(yᵢ + 1) − lnΓ(r)
//!      + r ln(r / (r + μᵢ)) + yᵢ ln(μᵢ / (r + μᵢ)),
//! ```
//!
//! the negative-binomial probability mass with size `r` and success
//! probability `r/(r + μᵢ)`. Coefficients and `α` are stacked into a single
//! parameter vector `[β, α]` and maximized with a full Newton iteration using
//! the analytic score and Hessian, so the estimate matches the canonical
//! reference to machine precision.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{chi2_sf, digamma, lgamma, norm_ppf, norm_sf};
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// Trigamma function `ψ'(x) = Σ_{k≥0} 1/(x+k)²`, via the asymptotic expansion
/// with upward recurrence (matches `scipy.special.polygamma(1, ·)` to ~1e-13).
fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Recurrence ψ'(x) = ψ'(x+1) + 1/x² to push the argument large.
    while x < 15.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic (Euler–Maclaurin) series for large x:
    //   ψ'(x) ≈ 1/x + 1/(2x²) + B₂/x³ + B₄/x⁵ + B₆/x⁷ + B₈/x⁹ + …
    // with Bernoulli numbers B₂=1/6, B₄=−1/30, B₆=1/42, B₈=−1/30, B₁₀=5/66.
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result
        + inv
        + 0.5 * inv2
        + inv2
            * inv
            * ((1.0 / 6.0)
                + inv2
                    * ((-1.0 / 30.0)
                        + inv2 * ((1.0 / 42.0) + inv2 * ((-1.0 / 30.0) + inv2 * (5.0 / 66.0)))))
}

/// A negative-binomial (NB2) model awaiting estimation.
///
/// Construct with [`NegativeBinomial::new`]; estimate with
/// [`NegativeBinomial::fit`].
#[derive(Clone, Debug)]
pub struct NegativeBinomial {
    endog: Array1<f64>,
    exog: Array2<f64>,
    k_constant: usize,
    maxiter: usize,
    gtol: f64,
}

impl NegativeBinomial {
    /// Build an NB2 negative-binomial count model.
    ///
    /// `exog` should already contain an intercept column if one is wanted (see
    /// `solow_core::tools::add_constant`). Returns an error on shape mismatch.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        let k_constant = detect_k_constant(&exog);
        Ok(NegativeBinomial {
            endog,
            exog,
            k_constant,
            maxiter: 200,
            gtol: 1e-11,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Linear-predictor mean `μ = exp(Xβ)` for coefficient block `beta`.
    fn mean(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.exog.dot(beta).mapv(f64::exp)
    }

    /// Log-likelihood at the stacked parameter vector `[β, α]`.
    fn loglike(&self, params: &Array1<f64>) -> f64 {
        let p = params.len();
        let alpha = params[p - 1];
        let beta = params.slice(ndarray::s![..p - 1]).to_owned();
        let a1 = 1.0 / alpha; // r = 1/α
        let mu = self.mean(&beta);
        let y = &self.endog;
        let mut ll = 0.0;
        for i in 0..y.len() {
            let size = a1;
            let prob = size / (size + mu[i]);
            ll += lgamma(size + y[i]) - lgamma(y[i] + 1.0) - lgamma(size)
                + size * prob.ln()
                + y[i] * (1.0 - prob).ln();
        }
        ll
    }

    /// Analytic score at `[β, α]`.
    fn score(&self, params: &Array1<f64>) -> Array1<f64> {
        let p = params.len();
        let alpha = params[p - 1];
        let beta = params.slice(ndarray::s![..p - 1]).to_owned();
        let a1 = 1.0 / alpha;
        let mu = self.mean(&beta);
        let y = &self.endog;
        let n = y.len();
        let kb = p - 1;
        let mut g = Array1::<f64>::zeros(p);

        // dℓ/dβ = Xᵀ ((y − μ)/(1 + α μ))
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            w[i] = (y[i] - mu[i]) / (1.0 + alpha * mu[i]);
        }
        let gb = self.exog.t().dot(&w);
        for a in 0..kb {
            g[a] = gb[a];
        }

        // dℓ/dα via the size = 1/α reparameterization.
        let da1 = -alpha.powi(-2); // d(1/α)/dα
        let mut s = 0.0;
        for i in 0..n {
            let dgpart = digamma(a1 + y[i]) - digamma(a1);
            let prob = a1 / (a1 + mu[i]);
            s += da1 * (dgpart + prob.ln() - (y[i] - mu[i]) / (a1 + mu[i]));
        }
        g[p - 1] = s;
        g
    }

    /// Analytic Hessian at `[β, α]`.
    fn hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let p = params.len();
        let alpha = params[p - 1];
        let beta = params.slice(ndarray::s![..p - 1]).to_owned();
        let a1 = 1.0 / alpha;
        let mu = self.mean(&beta);
        let y = &self.endog;
        let n = y.len();
        let dim = p; // kb + 1
        let kb = p - 1;
        let mut h = Array2::<f64>::zeros((dim, dim));

        // ∂²ℓ/∂β∂β = −Xᵀ diag(c) X with c = a1·μ·(a1+y)/(μ+a1)².
        let mut c = Array1::<f64>::zeros(n);
        for i in 0..n {
            let den = mu[i] + a1;
            c[i] = a1 * mu[i] * (a1 + y[i]) / (den * den);
        }
        for a in 0..kb {
            for b in a..kb {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += self.exog[[i, a]] * c[i] * self.exog[[i, b]];
                }
                h[[a, b]] = -sm;
                h[[b, a]] = -sm;
            }
        }

        // ∂²ℓ/∂β∂α  = −Σ μ x (y − μ) a1² / (μ + a1)²
        for a in 0..kb {
            let mut sm = 0.0;
            for i in 0..n {
                let den = mu[i] + a1;
                sm += mu[i] * self.exog[[i, a]] * (y[i] - mu[i]) * a1 * a1 / (den * den);
            }
            h[[kb, a]] = -sm;
            h[[a, kb]] = -sm;
        }

        // ∂²ℓ/∂α² (size = 1/α reparameterization, matches the reference).
        let da1 = -alpha.powi(-2);
        let da2 = 2.0 * alpha.powi(-3);
        let mut dada = 0.0;
        for i in 0..n {
            let dgpart = digamma(a1 + y[i]) - digamma(a1);
            let prob = a1 / (a1 + mu[i]);
            let dalpha = da1 * (dgpart + prob.ln() - (y[i] - mu[i]) / (a1 + mu[i]));
            let inner = trigamma(a1 + y[i]) - trigamma(a1) + 1.0 / a1 - 1.0 / (a1 + mu[i])
                + (y[i] - mu[i]) / ((mu[i] + a1) * (mu[i] + a1));
            dada += da2 * dalpha / da1 + da1 * da1 * inner;
        }
        h[[kb, kb]] = dada;
        h
    }

    /// Starting parameters: a few Poisson-style Newton steps for `β`, then a
    /// moment-based `α₀` from the resulting Pearson dispersion.
    fn start(&self) -> Array1<f64> {
        let n = self.endog.len();
        let kb = self.exog.ncols();
        // --- Poisson IRLS / Newton for the mean coefficients. ---
        let mut beta = Array1::<f64>::zeros(kb);
        for _ in 0..50 {
            let mu = self.mean(&beta);
            // score = Xᵀ(y − μ); −H = Xᵀ diag(μ) X
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n {
                w[i] = self.endog[i] - mu[i];
            }
            let g = self.exog.t().dot(&w);
            let mut hh = Array2::<f64>::zeros((kb, kb));
            for a in 0..kb {
                for b in a..kb {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += self.exog[[i, a]] * mu[i] * self.exog[[i, b]];
                    }
                    hh[[a, b]] = s;
                    hh[[b, a]] = s;
                }
            }
            let step = match solow_linalg::solve(&hh, &g) {
                Ok(s) => s,
                Err(_) => break,
            };
            beta = &beta + &step;
            if step.dot(&step).sqrt() < 1e-12 {
                break;
            }
        }
        // --- Moment estimate of α from Pearson over-dispersion. ---
        let mu = self.mean(&beta);
        let df_resid = (n as f64 - kb as f64).max(1.0);
        let mut num = 0.0;
        for i in 0..n {
            let r = self.endog[i] - mu[i];
            num += (r * r / mu[i] - 1.0) / mu[i];
        }
        let a0 = (num / df_resid).max(0.05);
        let mut out = Array1::<f64>::zeros(kb + 1);
        for a in 0..kb {
            out[a] = beta[a];
        }
        out[kb] = a0;
        out
    }

    /// Intercept-only (null) log-likelihood, fit by Newton with a single
    /// constant regressor.
    fn llnull(&self) -> Result<f64> {
        let n = self.endog.len();
        let ones = Array2::<f64>::ones((n, 1));
        let null = NegativeBinomial {
            endog: self.endog.clone(),
            exog: ones,
            k_constant: 1,
            maxiter: self.maxiter,
            gtol: self.gtol,
        };
        let (params, _, _) = null.fit_params()?;
        Ok(null.loglike(&params))
    }

    /// Core Newton fit returning `(params, cov, converged)`.
    fn fit_params(&self) -> Result<(Array1<f64>, Array2<f64>, bool)> {
        let fgh = |pp: &Array1<f64>| {
            // Keep α strictly positive by clamping the value fed to the model;
            // at the optimum it is well inside the feasible region.
            let mut q = pp.clone();
            let last = q.len() - 1;
            if q[last] < 1e-8 {
                q[last] = 1e-8;
            }
            let f = -self.loglike(&q);
            let g = self.score(&q).mapv(|v| -v);
            let h = self.hessian(&q).mapv(|v| -v);
            (f, g, h)
        };
        let opt = newton_stationary(&self.start(), fgh, self.maxiter, self.gtol)?;
        let params = opt.x;
        let h = self.hessian(&params);
        let neg_h = h.mapv(|v| -v);
        let cov = inv(&neg_h)?;
        Ok((params, cov, opt.converged))
    }

    /// Estimate by full Newton steps and assemble [`NegativeBinomialResults`].
    pub fn fit(&self) -> Result<NegativeBinomialResults> {
        let (params, cov, converged) = self.fit_params()?;
        let llnull = self.llnull()?;
        Ok(NegativeBinomialResults::new(
            self, params, cov, llnull, converged,
        ))
    }
}

fn detect_k_constant(exog: &Array2<f64>) -> usize {
    let (_, k) = exog.dim();
    for j in 0..k {
        let col = exog.column(j);
        let Some(&first) = col.iter().next() else {
            continue;
        };
        if first != 0.0 && col.iter().all(|&v| v == first) {
            return 1;
        }
    }
    0
}

/// The fitted result of an NB2 negative-binomial model.
#[derive(Clone, Debug)]
pub struct NegativeBinomialResults {
    /// Estimated parameters `[β, α]` (the last entry is the dispersion `α`).
    pub params: Array1<f64>,
    /// Standard errors `√diag((−H)^{-1})` (the last is `se(α)`).
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`.
    pub pvalues: Array1<f64>,

    /// Number of observations.
    pub nobs: f64,
    /// Whether a constant is present (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom (regressors excluding the constant; `α`
    /// excluded).
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

    /// Akaike information criterion `−2 llf + 2(k_exog + 1)`.
    pub aic: f64,
    /// Bayesian information criterion `−2 llf + (k_exog + 1) ln n`.
    pub bic: f64,

    /// Fitted means `μ = exp(Xβ̂)`.
    pub predicted: Array1<f64>,

    /// Parameter covariance `(−H)^{-1}` (includes the `α` row/column).
    pub cov_params: Array2<f64>,

    /// Whether Newton converged.
    pub converged: bool,
}

impl NegativeBinomialResults {
    fn new(
        model: &NegativeBinomial,
        params: Array1<f64>,
        cov_params: Array2<f64>,
        llnull: f64,
        converged: bool,
    ) -> NegativeBinomialResults {
        let nobs = model.endog.len() as f64;
        let p = params.len();
        let kb = p - 1; // number of regression coefficients (incl. constant)
        let k_constant = model.k_constant;
        let df_model = kb as f64 - k_constant as f64;
        let df_resid = nobs - df_model - 1.0;

        let mut bse = Array1::<f64>::zeros(p);
        for i in 0..p {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        let llf = model.loglike(&params);
        let llr = 2.0 * (llf - llnull);
        let llr_pvalue = chi2_sf(llr, df_model);
        let prsquared = 1.0 - llf / llnull;

        // AIC/BIC count k_exog + 1 parameters (the +1 is α).
        let k_params = kb as f64 + 1.0;
        let aic = -2.0 * llf + 2.0 * k_params;
        let bic = -2.0 * llf + k_params * nobs.ln();

        let beta = params.slice(ndarray::s![..kb]).to_owned();
        let predicted = model.mean(&beta);

        NegativeBinomialResults {
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
            predicted,
            cov_params,
            converged,
        }
    }

    /// Confidence interval for each parameter at level `1 − alpha` (normal).
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

    /// A full results table in the canonical reference discrete layout, titled
    /// "NegativeBinomial Regression Results".
    ///
    /// `names` labels the `k_exog` regression coefficients; the estimated
    /// dispersion is shown as a trailing `alpha` row (matching the reference, so
    /// you supply names only for the regressors). The output matches the
    /// reference field-for-field (only the `Date:`/`Time:` stamps vary).
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
        let _ = writeln!(s, "{}", centered("NegativeBinomial Regression Results", W));
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
                "NegativeBinomial",
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

        // Coefficient table: regressors, then the estimated dispersion `alpha`.
        let _ = writeln!(s, "{}", coef_header());
        let _ = writeln!(s, "{dash}");
        for i in 0..k {
            let name = if i + 1 == k {
                "alpha".to_string()
            } else {
                names
                    .and_then(|n| n.get(i).copied())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("x{i}"))
            };
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
        let xcol = array![0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4];
        let mut x = Array2::<f64>::ones((12, 2));
        x.column_mut(1).assign(&xcol);
        let y = array![2.0, 0.0, 3.0, 8.0, 1.0, 5.0, 1.0, 4.0, 0.0, 3.0, 2.0, 1.0];
        (y, x)
    }

    #[test]
    fn trigamma_matches_known_values() {
        // ψ'(1) = π²/6, ψ'(2) = π²/6 − 1, ψ'(0.5) = π²/2.
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        assert_abs_diff_eq!(trigamma(1.0), pi2 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(trigamma(2.0), pi2 / 6.0 - 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(trigamma(0.5), pi2 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn score_zero_at_optimum() {
        let (y, x) = design();
        let m = NegativeBinomial::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let g = m.score(&res.params);
        assert!(g.dot(&g).sqrt() < 1e-7, "score norm {}", g.dot(&g).sqrt());
        assert!(res.params[res.params.len() - 1] > 0.0); // alpha positive
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let (y, x) = design();
        let m = NegativeBinomial::new(y, x).unwrap();
        let p = array![0.4, -0.1, 0.6];
        let g_an = m.score(&p);
        let eps = 1e-6;
        for j in 0..p.len() {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp[j] += eps;
            pm[j] -= eps;
            let fd = (m.loglike(&pp) - m.loglike(&pm)) / (2.0 * eps);
            assert_abs_diff_eq!(g_an[j], fd, epsilon = 1e-5);
        }
    }

    #[test]
    fn analytic_hessian_matches_finite_difference() {
        let (y, x) = design();
        let m = NegativeBinomial::new(y, x).unwrap();
        let p = array![0.4, -0.1, 0.6];
        let h_an = m.hessian(&p);
        let eps = 1e-6;
        for j in 0..p.len() {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp[j] += eps;
            pm[j] -= eps;
            let gp = m.score(&pp);
            let gm = m.score(&pm);
            for i in 0..p.len() {
                let fd = (gp[i] - gm[i]) / (2.0 * eps);
                assert_abs_diff_eq!(h_an[[i, j]], fd, epsilon = 1e-4);
            }
        }
    }

    /// The summary must carry the discrete header block and a coefficient table
    /// whose final row is the estimated dispersion `alpha`.
    #[test]
    fn summary_has_alpha_row_and_blocks() {
        let (y, x) = design();
        let res = NegativeBinomial::new(y, x).unwrap().fit().unwrap();
        let s = res.summary(Some(&["const", "x1"]));
        assert!(s.contains("NegativeBinomial Regression Results"));
        assert!(s.contains("Method:") && s.contains("MLE"));
        assert!(s.contains("No. Observations:"));
        assert!(s.contains("Df Residuals:") && s.contains("Df Model:"));
        assert!(s.contains("Pseudo R-squ.:") && s.contains("LL-Null:"));
        assert!(s.contains("LLR p-value:"));
        assert!(s.contains("P>|z|") && !s.contains("P>|t|"));
        assert!(s.contains("[0.025") && s.contains("0.975]"));
        assert!(s.contains("const") && s.contains("x1"));
        // The dispersion appears as a trailing `alpha` parameter row.
        assert!(s.contains("alpha"));
    }
}
