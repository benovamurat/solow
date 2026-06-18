//! Generalized Poisson (GP-1) count regression ([`GeneralizedPoisson`]).
//!
//! The generalized Poisson model extends Poisson regression with a dispersion
//! parameter `α` that allows the conditional variance to differ from the mean.
//! With the default parameterization `p = 1` (so the reference's internal
//! `parameterization = p − 1 = 0`), the mean is `μᵢ = exp(xᵢ · β)` and the
//! per-observation log-likelihood is
//!
//! ```text
//! ℓᵢ = ln μᵢ + (yᵢ − 1) ln(μᵢ + α yᵢ) − yᵢ ln(1 + α) − ln(yᵢ!)
//!      − (μᵢ + α yᵢ) / (1 + α).
//! ```
//!
//! (Writing `a₁ = 1 + α`, `a₂ = μ + α y`.) The implied variance is
//! `μ (1 + α)²`, so `α > 0` is over-dispersion. Coefficients and `α` are stacked
//! into a single vector `[β, α]` and maximized by a full Newton iteration using
//! the analytic score and Hessian, matching the canonical reference to machine
//! precision.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{chi2_sf, lgamma, norm_ppf, norm_sf};
use solow_linalg::inv;
use solow_optimize::{minimize_bfgs, newton_stationary};

/// A generalized Poisson (GP-1) model awaiting estimation.
///
/// Construct with [`GeneralizedPoisson::new`]; estimate with
/// [`GeneralizedPoisson::fit`].
#[derive(Clone, Debug)]
pub struct GeneralizedPoisson {
    endog: Array1<f64>,
    exog: Array2<f64>,
    k_constant: usize,
    maxiter: usize,
    gtol: f64,
}

impl GeneralizedPoisson {
    /// Build a GP-1 generalized-Poisson count model.
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
        Ok(GeneralizedPoisson {
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

    /// Mean `μ = exp(Xβ)` for the coefficient block of `params`.
    fn mean(&self, params: &Array1<f64>) -> Array1<f64> {
        let kb = params.len() - 1;
        let beta = params.slice(s![..kb]).to_owned();
        self.exog.dot(&beta).mapv(f64::exp)
    }

    /// Log-likelihood at the stacked parameter vector `[β, α]` (GP-1, p = 1).
    fn loglike(&self, params: &Array1<f64>) -> f64 {
        let kb = params.len() - 1;
        let alpha = params[kb];
        let mu = self.mean(params);
        let y = &self.endog;
        let a1 = 1.0 + alpha; // mu_p = mu^0 = 1
        let mut ll = 0.0;
        for i in 0..y.len() {
            let a2 = (mu[i] + alpha * y[i]).max(1e-20);
            let a1c = a1.max(1e-20);
            ll += mu[i].ln() + (y[i] - 1.0) * a2.ln()
                - y[i] * a1c.ln()
                - lgamma(y[i] + 1.0)
                - a2 / a1c;
        }
        ll
    }

    /// Analytic score at `[β, α]` (GP-1, p = 1; `mu_p = 1`, `a3 = a4 = 0`).
    fn score(&self, params: &Array1<f64>) -> Array1<f64> {
        let kb = params.len() - 1;
        let alpha = params[kb];
        let mu = self.mean(params);
        let y = &self.endog;
        let n = y.len();
        let a1 = 1.0 + alpha; // mu_p = 1
        let mut g = Array1::<f64>::zeros(kb + 1);

        // dparams = dmudb * ( (y-1)/a2 - 1/a1 + 1/mu )   (a3 = a4 = 0 at p=1)
        // dmudb = mu * x  →  contribution per obs is w_i = mu_i * ((y-1)/a2 - 1/a1 + 1/mu)
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let a2 = mu[i] + alpha * y[i];
            w[i] = mu[i] * ((y[i] - 1.0) / a2 - 1.0 / a1 + 1.0 / mu[i]);
        }
        let gb = self.exog.t().dot(&w);
        for j in 0..kb {
            g[j] = gb[j];
        }

        // dalpha = mu_p * ( y*((y-1)/a2 - 2/a1) + a2/a1^2 ), mu_p = 1.
        let mut da = 0.0;
        for i in 0..n {
            let a2 = mu[i] + alpha * y[i];
            da += y[i] * ((y[i] - 1.0) / a2 - 2.0 / a1) + a2 / (a1 * a1);
        }
        g[kb] = da;
        g
    }

    /// Analytic Hessian at `[β, α]` (GP-1, p = 1).
    fn hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let kb = params.len() - 1;
        let alpha = params[kb];
        let mu = self.mean(params);
        let y = &self.endog;
        let n = y.len();
        let a1 = 1.0 + alpha; // mu_p = 1
        let dim = kb + 1;
        let mut h = Array2::<f64>::zeros((dim, dim));

        // β–β block. With p = 1: a3 = a4 = a5 = 0, and the reference's bracket
        // collapses to
        //   mu * exog_i exog_j * ( mu * (-(y-1)(1)^2 / a2^2)
        //                          + ((y-1)/a2 - 1/a1) )
        // because (1 + a4) = 1 here.
        // weight_i = mu_i * ( mu_i * (-(y-1)/a2^2) + (y-1)/a2 - 1/a1 )
        let mut wbb = Array1::<f64>::zeros(n);
        for i in 0..n {
            let a2 = mu[i] + alpha * y[i];
            wbb[i] = mu[i] * (mu[i] * (-(y[i] - 1.0) / (a2 * a2)) + (y[i] - 1.0) / a2 - 1.0 / a1);
        }
        for a in 0..kb {
            for b in a..kb {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += self.exog[[i, a]] * wbb[i] * self.exog[[i, b]];
                }
                h[[a, b]] = sm;
                h[[b, a]] = sm;
            }
        }

        // β–α block.  dldpda summand (mu_p = 1, a5 = 0):
        //   ( -mu_p*y*(y-1)*(1+a4)/a2^2 + mu_p*(1+a4)/a1^2 ) * dmudb
        // with a4 = 0, dmudb_i = mu_i x_i:
        //   ( -y(y-1)/a2^2 + 1/a1^2 ) * mu_i x_i
        for a in 0..kb {
            let mut sm = 0.0;
            for i in 0..n {
                let a2 = mu[i] + alpha * y[i];
                sm += (-y[i] * (y[i] - 1.0) / (a2 * a2) + 1.0 / (a1 * a1))
                    * mu[i]
                    * self.exog[[i, a]];
            }
            h[[kb, a]] = sm;
            h[[a, kb]] = sm;
        }

        // α–α term.  dldada = mu_p^2 * ( 3y/a1^2 - (y/a2)^2 (y-1) - 2 a2/a1^3 ).
        let mut dada = 0.0;
        for i in 0..n {
            let a2 = mu[i] + alpha * y[i];
            dada += 3.0 * y[i] / (a1 * a1)
                - (y[i] / a2).powi(2) * (y[i] - 1.0)
                - 2.0 * a2 / (a1 * a1 * a1);
        }
        h[[kb, kb]] = dada;
        h
    }

    /// Starting parameters: Poisson Newton for `β`, then a small positive `α₀`.
    fn start(&self) -> Array1<f64> {
        let n = self.endog.len();
        let kb = self.exog.ncols();
        let mut beta = Array1::<f64>::zeros(kb);
        for _ in 0..50 {
            let mu = self.exog.dot(&beta).mapv(f64::exp);
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
        // Pearson dispersion → a small positive starting alpha.
        let mu = self.exog.dot(&beta).mapv(f64::exp);
        let df = (n as f64 - kb as f64).max(1.0);
        let mut chi2 = 0.0;
        for i in 0..n {
            let r = self.endog[i] - mu[i];
            chi2 += r * r / mu[i];
        }
        // For GP-1, Var = mu (1+alpha)^2; method-of-moments ⇒ 1+alpha ≈ sqrt(chi2/df).
        let a0 = ((chi2 / df).sqrt() - 1.0).clamp(0.05, 5.0);
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
        let null = GeneralizedPoisson {
            endog: self.endog.clone(),
            exog: ones,
            k_constant: 1,
            maxiter: self.maxiter,
            gtol: self.gtol,
        };
        let (params, _, _) = null.fit_params()?;
        Ok(null.loglike(&params))
    }

    /// Core fit returning `(params, cov, converged)`.
    ///
    /// A BFGS pass on the analytic gradient provides a globally robust route to
    /// the interior optimum (Newton alone can diverge on small or
    /// under-dispersed samples, mirroring the reference); a final Newton polish
    /// drives the gradient to zero for machine-precision agreement.
    fn fit_params(&self) -> Result<(Array1<f64>, Array2<f64>, bool)> {
        let clamp = |pp: &Array1<f64>| {
            let mut q = pp.clone();
            let last = q.len() - 1;
            // Keep α in the feasible region (a1 = 1+α must stay positive).
            if q[last] < -0.99 {
                q[last] = -0.99;
            }
            q
        };
        let nll = |pp: &Array1<f64>| -self.loglike(&clamp(pp));
        let neg_score = |pp: &Array1<f64>| self.score(&clamp(pp)).mapv(|v| -v);
        let fgh = |pp: &Array1<f64>| {
            let q = clamp(pp);
            let f = -self.loglike(&q);
            let g = self.score(&q).mapv(|v| -v);
            let h = self.hessian(&q).mapv(|v| -v);
            (f, g, h)
        };

        let start = self.start();
        // Newton (analytic Hessian) converges quadratically and is tried first.
        // Only if it fails to converge do we fall back to a robust BFGS pass and
        // a final Newton polish — this keeps the common case fast.
        let mut opt = newton_stationary(&start, fgh, self.maxiter, self.gtol)?;
        if !opt.converged {
            let bfgs = minimize_bfgs(&start, nll, neg_score, 5000, 1e-10)?;
            let polished = newton_stationary(&bfgs.x, fgh, self.maxiter, self.gtol)?;
            // Keep whichever feasible point has the larger log-likelihood.
            let ll_bfgs = self.loglike(&clamp(&bfgs.x));
            let ll_pol = self.loglike(&clamp(&polished.x));
            let conv = polished.converged || bfgs.converged;
            if ll_pol >= ll_bfgs - 1e-6 {
                opt = polished;
            } else {
                opt.x = bfgs.x;
            }
            opt.converged = conv;
        }

        let params = clamp(&opt.x);
        let h = self.hessian(&params);
        let neg_h = h.mapv(|v| -v);
        let cov = inv(&neg_h)?;
        Ok((params, cov, opt.converged))
    }

    /// Estimate by full Newton steps and assemble [`GeneralizedPoissonResults`].
    pub fn fit(&self) -> Result<GeneralizedPoissonResults> {
        let (params, cov, converged) = self.fit_params()?;
        let llnull = self.llnull()?;
        Ok(GeneralizedPoissonResults::new(
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

/// The fitted result of a generalized-Poisson (GP-1) model.
#[derive(Clone, Debug)]
pub struct GeneralizedPoissonResults {
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

    /// Akaike information criterion `−2 llf + 2·(k_exog + 1)`.
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

impl GeneralizedPoissonResults {
    fn new(
        model: &GeneralizedPoisson,
        params: Array1<f64>,
        cov_params: Array2<f64>,
        llnull: f64,
        converged: bool,
    ) -> GeneralizedPoissonResults {
        let nobs = model.endog.len() as f64;
        let p = params.len();
        let kb = p - 1;
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

        let k_params = kb as f64 + 1.0;
        let aic = -2.0 * llf + 2.0 * k_params;
        let bic = -2.0 * llf + k_params * nobs.ln();

        let predicted = model.mean(&params);

        GeneralizedPoissonResults {
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
    /// "GeneralizedPoisson Regression Results".
    ///
    /// `names` labels the `k_exog` regression coefficients; the estimated
    /// dispersion is shown as a trailing `alpha` row (matching the reference, so
    /// supply names only for the regressors). The output matches the reference
    /// field-for-field (only the `Date:`/`Time:` stamps vary).
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
        let _ = writeln!(
            s,
            "{}",
            centered("GeneralizedPoisson Regression Results", W)
        );
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
                "GeneralizedPoisson",
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
        // A moderately over-dispersed count sample (20 obs) on which the GP-1
        // likelihood has a well-separated interior optimum.
        let xcol = array![
            0.5, -1.2, 0.3, 1.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4, 0.8, -0.6, 0.4, -0.9,
            0.2, 1.0, -0.3, 0.7
        ];
        let mut x = Array2::<f64>::ones((20, 2));
        x.column_mut(1).assign(&xcol);
        let y = array![
            3.0, 0.0, 4.0, 6.0, 1.0, 7.0, 2.0, 5.0, 0.0, 4.0, 3.0, 1.0, 5.0, 0.0, 3.0, 1.0, 2.0,
            8.0, 1.0, 4.0
        ];
        (y, x)
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let (y, x) = design();
        let m = GeneralizedPoisson::new(y, x).unwrap();
        let p = array![0.4, -0.1, 0.6];
        let g = m.score(&p);
        let eps = 1e-6;
        for j in 0..p.len() {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp[j] += eps;
            pm[j] -= eps;
            let fd = (m.loglike(&pp) - m.loglike(&pm)) / (2.0 * eps);
            assert_abs_diff_eq!(g[j], fd, epsilon = 1e-5);
        }
    }

    #[test]
    fn analytic_hessian_matches_finite_difference() {
        let (y, x) = design();
        let m = GeneralizedPoisson::new(y, x).unwrap();
        let p = array![0.4, -0.1, 0.6];
        let h = m.hessian(&p);
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
                assert_abs_diff_eq!(h[[i, j]], fd, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn score_zero_at_optimum() {
        let (y, x) = design();
        let m = GeneralizedPoisson::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let g = m.score(&res.params);
        assert!(g.dot(&g).sqrt() < 1e-7, "score norm {}", g.dot(&g).sqrt());
    }

    /// The summary must carry every block/label of the reference's discrete
    /// layout (z-inference, no t/F block) with the dispersion as a trailing
    /// `alpha` row.
    #[test]
    fn summary_has_every_block() {
        let (y, x) = design();
        let res = GeneralizedPoisson::new(y, x).unwrap().fit().unwrap();
        let s = res.summary(Some(&["const", "x1"]));

        assert!(s.contains("GeneralizedPoisson Regression Results"));
        assert!(s.contains("Dep. Variable:"));
        assert!(s.contains("Model:") && s.contains("GeneralizedPoisson"));
        assert!(s.contains("Method:") && s.contains("MLE"));
        assert!(s.contains("Date:") && s.contains("Time:"));
        assert!(s.contains("converged:"));
        assert!(s.contains("No. Observations:"));
        assert!(s.contains("Df Residuals:") && s.contains("Df Model:"));
        assert!(s.contains("Pseudo R-squ.:"));
        assert!(s.contains("Log-Likelihood:") && s.contains("LL-Null:"));
        assert!(s.contains("LLR p-value:"));
        assert!(s.contains("Covariance Type:") && s.contains("nonrobust"));
        assert!(s.contains("coef") && s.contains("std err"));
        assert!(s.contains("P>|z|") && !s.contains("P>|t|"));
        assert!(s.contains("[0.025") && s.contains("0.975]"));
        assert!(s.contains("const") && s.contains("x1") && s.contains("alpha"));
        assert!(!s.contains("Omnibus") && !s.contains("Durbin-Watson"));
    }
}
