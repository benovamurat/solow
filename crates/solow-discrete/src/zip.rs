//! Zero-inflated Poisson count regression ([`ZeroInflatedPoisson`]).
//!
//! A zero-inflated Poisson mixes a degenerate "always-zero" state with an
//! ordinary Poisson count process. With probability `wᵢ` the observation is a
//! structural zero; otherwise it is drawn from `Poisson(μᵢ)`. The inflation
//! probability is modeled with a logit link on a (constant-only by default)
//! inflation design,
//!
//! ```text
//! wᵢ = 1 / (1 + e^{−zᵢ·γ}),   μᵢ = exp(xᵢ·β),
//! ```
//!
//! and the per-observation log-likelihood is
//!
//! ```text
//! ℓᵢ = log(wᵢ + (1 − wᵢ) e^{−μᵢ})                     if yᵢ = 0,
//! ℓᵢ = log(1 − wᵢ) + yᵢ log μᵢ − μᵢ − log(yᵢ!)        if yᵢ > 0.
//! ```
//!
//! ## Parameter layout (matches the reference exactly)
//!
//! The stacked parameter vector is `[γ, β]`: first the `k_inflate` inflation
//! (logit) coefficients, then the `k_exog` Poisson coefficients. By default the
//! inflation design is a single constant column, so `k_inflate = 1` and `γ₀` is
//! the inflation intercept (`inflate_const` in the reference). The model is
//! maximized by a full Newton iteration on the analytic score and Hessian.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{chi2_sf, lgamma, norm_ppf, norm_sf};
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// Standard logistic CDF `1/(1 + e^{-x})`, numerically stable in both tails.
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// A zero-inflated Poisson model awaiting estimation.
///
/// Construct with [`ZeroInflatedPoisson::new`] (constant-only inflation) or
/// [`ZeroInflatedPoisson::with_inflation`]; estimate with
/// [`ZeroInflatedPoisson::fit`].
#[derive(Clone, Debug)]
pub struct ZeroInflatedPoisson {
    endog: Array1<f64>,
    /// Poisson (main) design, `n × k_exog`.
    exog: Array2<f64>,
    /// Inflation (logit) design, `n × k_inflate`.
    exog_infl: Array2<f64>,
    k_constant: usize,
    maxiter: usize,
    gtol: f64,
}

impl ZeroInflatedPoisson {
    /// Build a zero-inflated Poisson model with a **constant-only** inflation
    /// model (`k_inflate = 1`).
    ///
    /// `exog` is the main (count) design and should already contain an
    /// intercept column if wanted. Returns an error on shape mismatch.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        let n = endog.len();
        let infl = Array2::<f64>::ones((n, 1));
        Self::with_inflation(endog, exog, infl)
    }

    /// Build a zero-inflated Poisson model with an explicit inflation design.
    pub fn with_inflation(
        endog: Array1<f64>,
        exog: Array2<f64>,
        exog_infl: Array2<f64>,
    ) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if endog.len() != exog_infl.nrows() {
            return Err(Error::Shape("endog length != exog_infl rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        ensure_all_finite_2d(&exog_infl.view(), "exog_infl")?;
        let k_constant = detect_k_constant(&exog);
        Ok(ZeroInflatedPoisson {
            endog,
            exog,
            exog_infl,
            k_constant,
            maxiter: 200,
            gtol: 1e-11,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    fn k_inflate(&self) -> usize {
        self.exog_infl.ncols()
    }

    fn k_exog(&self) -> usize {
        self.exog.ncols()
    }

    /// Split a stacked `[γ, β]` vector into inflation and main blocks.
    fn split(&self, params: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let ki = self.k_inflate();
        let gamma = params.slice(s![..ki]).to_owned();
        let beta = params.slice(s![ki..]).to_owned();
        (gamma, beta)
    }

    /// Inflation probabilities `w = logistic(exog_infl · γ)`.
    fn infl_prob(&self, gamma: &Array1<f64>) -> Array1<f64> {
        self.exog_infl.dot(gamma).mapv(logistic)
    }

    /// Poisson means `μ = exp(exog · β)`.
    fn mean(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.exog.dot(beta).mapv(f64::exp)
    }

    /// Total log-likelihood at the stacked `[γ, β]` parameters.
    fn loglike(&self, params: &Array1<f64>) -> f64 {
        let (gamma, beta) = self.split(params);
        let w = self
            .infl_prob(&gamma)
            .mapv(|v| v.clamp(f64::EPSILON, 1.0 - f64::EPSILON));
        let mu = self.mean(&beta);
        let y = &self.endog;
        let mut ll = 0.0;
        for i in 0..y.len() {
            // Poisson loglikeobs for the main model.
            let llf_main = -mu[i] + y[i] * mu[i].ln() - lgamma(y[i] + 1.0);
            if y[i] == 0.0 {
                ll += (w[i] + (1.0 - w[i]) * llf_main.exp()).ln();
            } else {
                ll += (1.0 - w[i]).ln() + llf_main;
            }
        }
        ll
    }

    /// Analytic score at the stacked `[γ, β]` parameters.
    ///
    /// Mirrors the reference: with `pmf_i = e^{ℓ_i}` the full ZIP per-obs
    /// likelihood and `score_main` the Poisson score `(y − μ) x`,
    ///
    /// * for the main block, zero observations are reweighted by
    ///   `1 − w/pmf` and positive observations keep the Poisson score;
    /// * for the logit inflation block, zeros contribute
    ///   `z · w(1−w)(1 − e^{ℓ_main})/pmf` and positives contribute `−z·w`.
    fn score(&self, params: &Array1<f64>) -> Array1<f64> {
        let ki = self.k_inflate();
        let ke = self.k_exog();
        let (gamma, beta) = self.split(params);
        let w = self
            .infl_prob(&gamma)
            .mapv(|v| v.clamp(f64::EPSILON, 1.0 - f64::EPSILON));
        let mu = self.mean(&beta);
        let y = &self.endog;
        let n = y.len();

        let mut g = Array1::<f64>::zeros(ki + ke);

        // Main (Poisson) block.
        let mut dldp_w = Array1::<f64>::zeros(n); // per-obs weight on x_i for main score
                                                  // Inflation block per-obs weight on z_i.
        let mut dldw_w = Array1::<f64>::zeros(n);

        for i in 0..n {
            let llf_main = -mu[i] + y[i] * mu[i].ln() - lgamma(y[i] + 1.0);
            let p_main = llf_main.exp();
            let score_main = y[i] - mu[i]; // Poisson score per unit x
            if y[i] == 0.0 {
                let pmf = w[i] + (1.0 - w[i]) * p_main;
                dldp_w[i] = score_main * (1.0 - w[i] / pmf);
                dldw_w[i] = w[i] * (1.0 - w[i]) * (1.0 - p_main) / pmf;
            } else {
                dldp_w[i] = score_main;
                dldw_w[i] = -w[i];
            }
        }

        let gb = self.exog.t().dot(&dldp_w); // length k_exog
        let gw = self.exog_infl.t().dot(&dldw_w); // length k_inflate
        for a in 0..ki {
            g[a] = gw[a];
        }
        for a in 0..ke {
            g[ki + a] = gb[a];
        }
        g
    }

    /// Analytic Hessian at the stacked `[γ, β]` parameters (reference parity).
    fn hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let ki = self.k_inflate();
        let ke = self.k_exog();
        let (gamma, beta) = self.split(params);
        let w = self
            .infl_prob(&gamma)
            .mapv(|v| v.clamp(f64::EPSILON, 1.0 - f64::EPSILON));
        let mu = self.mean(&beta);
        let y = &self.endog;
        let n = y.len();
        let dim = ki + ke;
        let mut h = Array2::<f64>::zeros((dim, dim));

        // Precompute per-obs Poisson llf and full pmf.
        let mut llf_main = Array1::<f64>::zeros(n);
        let mut pmf = Array1::<f64>::zeros(n);
        for i in 0..n {
            llf_main[i] = -mu[i] + y[i] * mu[i].ln() - lgamma(y[i] + 1.0);
            pmf[i] = if y[i] == 0.0 {
                w[i] + (1.0 - w[i]) * llf_main[i].exp()
            } else {
                ((1.0 - w[i]).ln() + llf_main[i]).exp()
            };
        }

        // ----- main–main block (d2l/dβ dβ), index offset by k_inflate. -----
        // For zeros: coeff = 1 + w*(e^{mu} - 1);
        //   term = x_i x_j mu (w-1) (1/coeff - w mu e^{mu}/coeff^2).
        // For positives: -mu x_i x_j.
        for i in 0..ke {
            for j in 0..=i {
                let mut s = 0.0;
                for r in 0..n {
                    if y[r] == 0.0 {
                        let coeff = 1.0 + w[r] * (mu[r].exp() - 1.0);
                        s += self.exog[[r, i]]
                            * self.exog[[r, j]]
                            * mu[r]
                            * (w[r] - 1.0)
                            * (1.0 / coeff - w[r] * mu[r] * mu[r].exp() / (coeff * coeff));
                    } else {
                        s -= mu[r] * self.exog[[r, i]] * self.exog[[r, j]];
                    }
                }
                h[[ki + i, ki + j]] = s;
                h[[ki + j, ki + i]] = s;
            }
        }

        // ----- inflation–inflation block (d2l/dγ dγ). -----
        for i in 0..ki {
            for j in 0..=i {
                let mut s = 0.0;
                for r in 0..n {
                    if y[r] == 0.0 {
                        let em = llf_main[r].exp();
                        let val = w[r]
                            * (1.0 - w[r])
                            * ((1.0 - em) * (1.0 - 2.0 * w[r]) * pmf[r]
                                - (w[r] - w[r] * w[r]) * (1.0 - em).powi(2))
                            / (pmf[r] * pmf[r]);
                        s += self.exog_infl[[r, i]] * self.exog_infl[[r, j]] * val;
                    } else {
                        s -= self.exog_infl[[r, i]] * self.exog_infl[[r, j]] * w[r] * (1.0 - w[r]);
                    }
                }
                h[[i, j]] = s;
                h[[j, i]] = s;
            }
        }

        // ----- inflation–main cross block (d2l/dγ dβ). -----
        // Only zero observations contribute. For a zero obs the inflation score
        // is  dℓ/dγ = z_i · w(1−w)(1 − e^{m})/pmf  with m = −μ; differentiating
        // w.r.t. β (through e^{m} and pmf, with dm/dβ = −μ x_j) gives the exact
        // second derivative below (verified against finite differences).
        for i in 0..ki {
            for j in 0..ke {
                let mut s = 0.0;
                for r in 0..n {
                    if y[r] == 0.0 {
                        let em = llf_main[r].exp();
                        let a = w[r] * (1.0 - w[r]) * self.exog_infl[[r, i]];
                        let dem_db = em * (-mu[r]) * self.exog[[r, j]];
                        let dpmf_db = (1.0 - w[r]) * dem_db;
                        let d = (-dem_db * pmf[r] - (1.0 - em) * dpmf_db) / (pmf[r] * pmf[r]);
                        s += a * d;
                    }
                }
                h[[i, ki + j]] = s;
                h[[ki + j, i]] = s;
            }
        }

        h
    }

    /// Starting parameters: Poisson Newton for `β`, a small inflation intercept.
    fn start(&self) -> Array1<f64> {
        let n = self.endog.len();
        let ke = self.k_exog();
        let ki = self.k_inflate();
        // Poisson Newton for the main coefficients.
        let mut beta = Array1::<f64>::zeros(ke);
        for _ in 0..50 {
            let mu = self.mean(&beta);
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n {
                w[i] = self.endog[i] - mu[i];
            }
            let gscore = self.exog.t().dot(&w);
            let mut hh = Array2::<f64>::zeros((ke, ke));
            for a in 0..ke {
                for b in a..ke {
                    let mut sm = 0.0;
                    for i in 0..n {
                        sm += self.exog[[i, a]] * mu[i] * self.exog[[i, b]];
                    }
                    hh[[a, b]] = sm;
                    hh[[b, a]] = sm;
                }
            }
            let step = match solow_linalg::solve(&hh, &gscore) {
                Ok(s) => s,
                Err(_) => break,
            };
            beta = &beta + &step;
            if step.dot(&step).sqrt() < 1e-12 {
                break;
            }
        }
        let mut out = Array1::<f64>::zeros(ki + ke);
        // Small positive inflation intercept (reference uses 0.1 on every infl coef).
        for a in 0..ki {
            out[a] = 0.1;
        }
        for a in 0..ke {
            out[ki + a] = beta[a];
        }
        out
    }

    /// Intercept-only (null) log-likelihood: a constant-only main model with the
    /// same constant-only inflation, fit by Newton.
    fn llnull(&self) -> Result<f64> {
        let n = self.endog.len();
        let ones = Array2::<f64>::ones((n, 1));
        let infl = Array2::<f64>::ones((n, 1));
        let null = ZeroInflatedPoisson {
            endog: self.endog.clone(),
            exog: ones,
            exog_infl: infl,
            k_constant: 1,
            maxiter: self.maxiter,
            gtol: self.gtol,
        };
        let (params, _, _) = null.fit_params()?;
        Ok(null.loglike(&params))
    }

    /// Hessian with the inflation/main cross-block zeroed.
    ///
    /// The canonical reference assembles its analytic zero-inflated Hessian in a
    /// way that discards the inflation–main cross derivatives (the lower-left
    /// cross block is never filled and the symmetrization step overwrites the
    /// upper-left cross block with zeros). Its reported standard errors are
    /// therefore computed from this block-diagonal information matrix. We mirror
    /// that exactly so `bse` agrees to machine precision; the cross block does
    /// not affect the parameter estimates, only their standard errors.
    fn hessian_blockdiag(&self, params: &Array1<f64>) -> Array2<f64> {
        let ki = self.k_inflate();
        let mut h = self.hessian(params);
        let dim = h.nrows();
        for i in 0..ki {
            for j in ki..dim {
                h[[i, j]] = 0.0;
                h[[j, i]] = 0.0;
            }
        }
        h
    }

    /// Core Newton fit returning `(params, cov, converged)`.
    ///
    /// The Newton step uses the full analytic Hessian (including the
    /// inflation–main cross block) for fast, accurate convergence to the true
    /// optimum, but the covariance is formed from the block-diagonal Hessian to
    /// reproduce the reference's standard errors.
    fn fit_params(&self) -> Result<(Array1<f64>, Array2<f64>, bool)> {
        let fgh = |pp: &Array1<f64>| {
            let f = -self.loglike(pp);
            let g = self.score(pp).mapv(|v| -v);
            let h = self.hessian(pp).mapv(|v| -v);
            (f, g, h)
        };
        let opt = newton_stationary(&self.start(), fgh, self.maxiter, self.gtol)?;
        let params = opt.x;
        let h = self.hessian_blockdiag(&params);
        let neg_h = h.mapv(|v| -v);
        let cov = inv(&neg_h)?;
        Ok((params, cov, opt.converged))
    }

    /// Estimate by full Newton steps and assemble [`ZeroInflatedPoissonResults`].
    pub fn fit(&self) -> Result<ZeroInflatedPoissonResults> {
        let (params, cov, converged) = self.fit_params()?;
        let llnull = self.llnull()?;
        Ok(ZeroInflatedPoissonResults::new(
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

/// The fitted result of a zero-inflated Poisson model.
#[derive(Clone, Debug)]
pub struct ZeroInflatedPoissonResults {
    /// Estimated parameters `[γ, β]` (inflation logit coefs, then Poisson coefs).
    pub params: Array1<f64>,
    /// Standard errors `√diag((−H)^{-1})`.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`.
    pub pvalues: Array1<f64>,

    /// Number of inflation parameters `k_inflate`.
    pub k_inflate: usize,
    /// Number of observations.
    pub nobs: f64,
    /// Whether a constant is present in the main design (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom (main regressors excluding constant + inflation
    /// extras), matching the reference `df_model`.
    pub df_model: f64,
    /// Residual degrees of freedom.
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

    /// Akaike information criterion `−2 llf + 2·k_params`.
    pub aic: f64,
    /// Bayesian information criterion `−2 llf + k_params · ln n`.
    pub bic: f64,

    /// Expected response `E[y] = (1 − w) μ`.
    pub predicted: Array1<f64>,

    /// Parameter covariance `(−H)^{-1}`.
    pub cov_params: Array2<f64>,

    /// Whether Newton converged.
    pub converged: bool,
}

impl ZeroInflatedPoissonResults {
    fn new(
        model: &ZeroInflatedPoisson,
        params: Array1<f64>,
        cov_params: Array2<f64>,
        llnull: f64,
        converged: bool,
    ) -> ZeroInflatedPoissonResults {
        let nobs = model.endog.len() as f64;
        let ki = model.k_inflate();
        let ke = model.k_exog();
        let k_params = params.len(); // k_inflate + k_exog

        let mut bse = Array1::<f64>::zeros(k_params);
        for i in 0..k_params {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        // Reference convention (DiscreteModel.initialize): df_model and df_resid
        // are derived from the *main* count design only — `df_model = k_exog − 1`
        // (a constant is assumed) and `df_resid = nobs − k_exog`. The inflation
        // parameters do not enter these counts.
        let _ = ki;
        let df_model = ke as f64 - 1.0;
        let df_resid = nobs - ke as f64;

        let llf = model.loglike(&params);
        let llr = 2.0 * (llf - llnull);
        let llr_pvalue = chi2_sf(llr, df_model);
        let prsquared = 1.0 - llf / llnull;

        let kp = k_params as f64;
        let aic = -2.0 * llf + 2.0 * kp;
        let bic = -2.0 * llf + kp * nobs.ln();

        // Predicted mean E[y] = (1 - w) mu.
        let gamma = params.slice(s![..ki]).to_owned();
        let beta = params.slice(s![ki..]).to_owned();
        let w = model.infl_prob(&gamma);
        let mu = model.mean(&beta);
        let predicted = (1.0 - &w) * &mu;

        ZeroInflatedPoissonResults {
            params,
            bse,
            tvalues,
            pvalues,
            k_inflate: ki,
            nobs,
            k_constant: model.k_constant,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn design() -> (Array1<f64>, Array2<f64>) {
        let xcol = array![
            0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4, 1.1, -0.9, 0.8, -0.3
        ];
        let mut x = Array2::<f64>::ones((16, 2));
        x.column_mut(1).assign(&xcol);
        let y =
            array![0.0, 0.0, 3.0, 8.0, 0.0, 5.0, 1.0, 4.0, 0.0, 3.0, 0.0, 1.0, 6.0, 0.0, 4.0, 0.0];
        (y, x)
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let (y, x) = design();
        let m = ZeroInflatedPoisson::new(y, x).unwrap();
        let p = array![-0.5, 0.3, 0.5];
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
        let m = ZeroInflatedPoisson::new(y, x).unwrap();
        let p = array![-0.5, 0.3, 0.5];
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
        let m = ZeroInflatedPoisson::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let g = m.score(&res.params);
        assert!(g.dot(&g).sqrt() < 1e-6, "score norm {}", g.dot(&g).sqrt());
    }
}
