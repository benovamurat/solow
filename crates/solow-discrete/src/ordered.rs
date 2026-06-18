//! Ordinal regression with estimated thresholds ([`OrderedModel`]).
//!
//! An ordered (ordinal) response `yᵢ ∈ {0, 1, …, K−1}` is modeled through a
//! latent linear index `ηᵢ = xᵢ · β` and a set of `K−1` increasing cutpoints
//! `a₀ < a₁ < … < a_{K−2}` (with `a₋₁ = −∞`, `a_{K−1} = +∞`). The choice
//! probabilities are interval probabilities of the latent error distribution
//! `F` (logistic for ordinal logit, standard normal for ordinal probit):
//!
//! ```text
//! P(yᵢ = k) = F(a_k − ηᵢ) − F(a_{k−1} − ηᵢ).
//! ```
//!
//! There is **no intercept** in `β` (it is absorbed into the cutpoints), so
//! `exog` must not contain a constant column.
//!
//! ## Parameter layout (matches the reference exactly)
//!
//! The free parameter vector is `[β, c₀, d₁, d₂, …, d_{K−2}]`: first the `k_exog`
//! regression coefficients, then the **cutpoint transforms**. The first cutpoint
//! `c₀ = a₀` is unconstrained; each later entry `dⱼ = ln(aⱼ − a_{j−1})` is the log
//! of a positive increment, guaranteeing strictly increasing thresholds for any
//! real parameter values. Concretely
//!
//! ```text
//! a₀ = c₀,   aⱼ = aⱼ₋₁ + exp(dⱼ)   (j = 1 … K−2).
//! ```
//!
//! [`OrderedResults::params`] reports the free vector in this layout (so it
//! agrees element-by-element with the reference's `params`), while
//! [`OrderedResults::thresholds`] exposes the back-transformed cutpoints.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{chi2_sf, norm_cdf, norm_ppf, norm_sf};
use solow_optimize::{approx_hess, minimize_bfgs, newton_stationary};

/// Latent error distribution selecting ordinal logit vs. ordinal probit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distr {
    /// Logistic latent errors: ordinal **logit**.
    Logit,
    /// Gaussian latent errors: ordinal **probit**.
    Probit,
}

/// Standard logistic CDF `1/(1 + e^{-x})`, numerically stable in both tails.
fn logistic_cdf(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Standard logistic PDF `e^{-x}/(1 + e^{-x})² = F(x)(1 − F(x))`.
fn logistic_pdf(x: f64) -> f64 {
    let f = logistic_cdf(x);
    f * (1.0 - f)
}

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// An ordinal regression model awaiting estimation.
///
/// Construct with [`OrderedModel::new`] (ordinal logit) or
/// [`OrderedModel::with_distr`]; estimate with [`OrderedModel::fit`].
#[derive(Clone, Debug)]
pub struct OrderedModel {
    /// Integer-coded ordinal response in `0 … K−1`, stored as `f64`.
    endog: Array1<f64>,
    /// Design matrix `n × k_exog`; must NOT contain a constant column.
    exog: Array2<f64>,
    /// Number of ordered levels `K`.
    k_levels: usize,
    /// Latent error family.
    distr: Distr,
    maxiter: usize,
    gtol: f64,
}

impl OrderedModel {
    /// Build an **ordinal logit** model (logistic latent errors).
    ///
    /// `endog` holds integer level codes `0 … K−1` (as `f64`); `K` is inferred
    /// as `max(endog) + 1`. `exog` must NOT contain a constant column — the
    /// level cutpoints play the role of intercepts. Returns an error on shape
    /// mismatch, a non-integer/negative code, fewer than two levels, or a
    /// detected constant column.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        Self::with_distr(endog, exog, Distr::Logit)
    }

    /// Build an ordinal model with an explicit latent error family.
    pub fn with_distr(endog: Array1<f64>, exog: Array2<f64>, distr: Distr) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        if has_constant(&exog) {
            return Err(Error::Shape(
                "OrderedModel exog must not contain a constant column".into(),
            ));
        }
        let mut kmax = 0usize;
        for &v in endog.iter() {
            if v < 0.0 || v.fract() != 0.0 {
                return Err(Error::Shape("endog must hold integer level codes".into()));
            }
            let c = v as usize;
            if c > kmax {
                kmax = c;
            }
        }
        let k_levels = kmax + 1;
        if k_levels < 2 {
            return Err(Error::Shape(
                "OrderedModel needs at least two levels".into(),
            ));
        }
        Ok(OrderedModel {
            endog,
            exog,
            k_levels,
            distr,
            maxiter: 200,
            gtol: 1e-10,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Number of ordered levels `K`.
    pub fn k_levels(&self) -> usize {
        self.k_levels
    }

    /// Number of regression coefficients (`k_exog`, no intercept).
    fn k_vars(&self) -> usize {
        self.exog.ncols()
    }

    /// Latent CDF `F`.
    fn cdf(&self, x: f64) -> f64 {
        match self.distr {
            Distr::Logit => logistic_cdf(x),
            Distr::Probit => norm_cdf(x),
        }
    }

    /// Latent PDF `f = F'`.
    fn pdf(&self, x: f64) -> f64 {
        match self.distr {
            Distr::Logit => logistic_pdf(x),
            Distr::Probit => normal_pdf(x),
        }
    }

    /// Back-transform the cutpoint-transform block into the actual increasing
    /// thresholds `[a₀, a₁, …, a_{K−2}]`.
    fn thresholds(&self, params: &Array1<f64>) -> Array1<f64> {
        let kv = self.k_vars();
        let th = params.slice(s![kv..]); // length K-1
        let m = th.len();
        let mut out = Array1::<f64>::zeros(m);
        let mut acc = th[0];
        out[0] = acc;
        for j in 1..m {
            acc += th[j].exp();
            out[j] = acc;
        }
        out
    }

    /// Latent index `η = Xβ` for the coefficient block of `params`.
    fn linpred(&self, params: &Array1<f64>) -> Array1<f64> {
        let kv = self.k_vars();
        let beta = params.slice(s![..kv]).to_owned();
        self.exog.dot(&beta)
    }

    /// Per-observation interval probability `P(yᵢ = cᵢ)`.
    fn probs(&self, params: &Array1<f64>) -> Array1<f64> {
        let thr = self.thresholds(params);
        let eta = self.linpred(params);
        let n = self.endog.len();
        let kl = self.k_levels;
        let mut p = Array1::<f64>::zeros(n);
        for i in 0..n {
            let c = self.endog[i] as usize;
            let upp = if c == kl - 1 {
                1.0
            } else {
                self.cdf(thr[c] - eta[i])
            };
            let low = if c == 0 {
                0.0
            } else {
                self.cdf(thr[c - 1] - eta[i])
            };
            p[i] = (upp - low).max(1e-300);
        }
        p
    }

    /// Total log-likelihood at `params`.
    fn loglike(&self, params: &Array1<f64>) -> f64 {
        // Reference adds 1e-20 inside the log for numerical safety; we mirror it
        // so the maximized value agrees to machine precision.
        self.probs(params).iter().map(|&pi| (pi + 1e-20).ln()).sum()
    }

    /// Analytic score (gradient of the log-likelihood) at `params`.
    ///
    /// Differentiating `ℓ = Σ log p_i` with `p_i = F(u_i) − F(l_i)`, where
    /// `u_i = a_{c} − η_i` and `l_i = a_{c−1} − η_i`:
    ///
    /// * `∂ℓ/∂β = Σ (−1/p_i)(f(u_i) − f(l_i)) x_i`  (the `−x_i` is `∂u/∂β`),
    /// * `∂ℓ/∂a_t` collects `f(u_i)/p_i` from obs whose upper cut is `a_t` and
    ///   `−f(l_i)/p_i` from obs whose lower cut is `a_t`; the chain rule through
    ///   the increment transform turns these into the reported derivatives.
    fn score(&self, params: &Array1<f64>) -> Array1<f64> {
        let kv = self.k_vars();
        let kl = self.k_levels;
        let nthr = kl - 1;
        let thr = self.thresholds(params);
        let eta = self.linpred(params);
        let n = self.endog.len();

        let mut g = Array1::<f64>::zeros(params.len());
        // Accumulate dℓ/dη_i into a per-obs weight, and dℓ/da_t into a per-cut
        // accumulator; then map cuts → transform parameters via the chain rule.
        let mut dl_deta = Array1::<f64>::zeros(n);
        let mut dl_da = Array1::<f64>::zeros(nthr);

        for i in 0..n {
            let c = self.endog[i] as usize;
            let (u, fu) = if c == kl - 1 {
                (f64::INFINITY, 0.0)
            } else {
                let z = thr[c] - eta[i];
                (z, self.pdf(z))
            };
            let (_l, fl) = if c == 0 {
                (f64::NEG_INFINITY, 0.0)
            } else {
                let z = thr[c - 1] - eta[i];
                (z, self.pdf(z))
            };
            let p = self.probs_one(c, kl, &thr, eta[i]);
            let inv = 1.0 / p;
            // dη contribution: ∂u/∂η = ∂l/∂η = −1, so dℓ/dη = −(fu − fl)/p.
            dl_deta[i] = -(fu - fl) * inv;
            // upper cut a_c (only if c < K-1): +fu/p
            if c < kl - 1 {
                dl_da[c] += fu * inv;
            }
            // lower cut a_{c-1} (only if c > 0): −fl/p
            if c > 0 {
                dl_da[c - 1] -= fl * inv;
            }
            let _ = u;
        }

        // dℓ/dβ = Xᵀ (diag handling): dℓ/dβ_j = Σ_i dl_deta_i · x_{ij}.
        let gb = self.exog.t().dot(&dl_deta);
        for j in 0..kv {
            g[j] = gb[j];
        }

        // Chain rule: thresholds a_t depend on transform params p_t.
        // a_0 = c_0;  a_t = c_0 + Σ_{m=1..t} exp(d_m).
        // ∂a_t/∂c_0 = 1 for all t.
        // ∂a_t/∂d_m = exp(d_m) for t ≥ m (m ≥ 1).
        let th_block = params.slice(s![kv..]);
        // d c_0
        let mut s0 = 0.0;
        for t in 0..nthr {
            s0 += dl_da[t];
        }
        g[kv] = s0;
        for m in 1..nthr {
            let em = th_block[m].exp();
            let mut sm = 0.0;
            for t in m..nthr {
                sm += dl_da[t];
            }
            g[kv + m] = sm * em;
        }
        g
    }

    /// Interval probability for a single observation (helper for [`Self::score`]).
    fn probs_one(&self, c: usize, kl: usize, thr: &Array1<f64>, eta: f64) -> f64 {
        let upp = if c == kl - 1 {
            1.0
        } else {
            self.cdf(thr[c] - eta)
        };
        let low = if c == 0 {
            0.0
        } else {
            self.cdf(thr[c - 1] - eta)
        };
        (upp - low).max(1e-300)
    }

    /// Starting values: zeros for `β`, and evenly spaced empirical cutpoints
    /// transformed into the increment parameterization.
    fn start(&self) -> Array1<f64> {
        let kv = self.k_vars();
        let kl = self.k_levels;
        let nthr = kl - 1;
        let n = self.endog.len() as f64;
        // Empirical cumulative frequencies → cutpoints via F⁻¹.
        let mut counts = vec![0.0f64; kl];
        for &v in self.endog.iter() {
            counts[v as usize] += 1.0;
        }
        let mut cum = 0.0;
        let mut cuts = Vec::with_capacity(nthr);
        for &ck in counts.iter().take(nthr) {
            cum += ck;
            let q = (cum / n).clamp(1e-3, 1.0 - 1e-3);
            let a = match self.distr {
                Distr::Logit => (q / (1.0 - q)).ln(),
                Distr::Probit => norm_ppf(q),
            };
            cuts.push(a);
        }
        // Transform cutpoints → increment params.
        let mut out = Array1::<f64>::zeros(kv + nthr);
        out[kv] = cuts[0];
        for j in 1..nthr {
            let inc = (cuts[j] - cuts[j - 1]).max(1e-3);
            out[kv + j] = inc.ln();
        }
        out
    }

    /// Intercept-only (null) log-likelihood: cutpoints set to the empirical
    /// cumulative log-odds with `β = 0`, giving `P(y=k) = n_k / n` exactly.
    fn llnull(&self) -> f64 {
        let n = self.endog.len() as f64;
        let kl = self.k_levels;
        let mut counts = vec![0.0f64; kl];
        for &v in self.endog.iter() {
            counts[v as usize] += 1.0;
        }
        let mut ll = 0.0;
        for c in counts {
            if c > 0.0 {
                ll += c * (c / n).ln();
            }
        }
        ll
    }

    /// Estimate by maximum likelihood and assemble [`OrderedResults`].
    ///
    /// Optimization uses a Newton iteration on the analytic gradient with a
    /// numeric Hessian, falling back to BFGS if a Newton step misbehaves; the
    /// covariance is the inverse negative numeric Hessian of the
    /// log-likelihood, matching the reference's `bse`.
    pub fn fit(&self) -> Result<OrderedResults> {
        let start = self.start();

        // BFGS on the analytic gradient reaches the optimum quickly and
        // robustly; a short analytic-Hessian Newton polish then drives the
        // gradient to zero for machine-precision agreement. The expensive
        // finite-difference Hessian is formed only once, for the covariance.
        let nll = |p: &Array1<f64>| -self.loglike(p);
        let neg_score = |p: &Array1<f64>| self.score(p).mapv(|v| -v);

        let fgh = |p: &Array1<f64>| {
            let f = nll(p);
            let g = neg_score(p);
            // A finite-difference Jacobian of the analytic gradient is the
            // Hessian; it costs only O(p) score evaluations.
            let h = score_jacobian(p, &neg_score);
            (f, g, h)
        };

        // Newton on the analytic gradient converges in a handful of iterations
        // from the empirical-cutpoint start; BFGS is only a robustness fallback.
        let mut opt = newton_stationary(&start, fgh, self.maxiter, self.gtol)?;
        if !opt.converged {
            let bfgs = minimize_bfgs(&start, nll, neg_score, 5000, 1e-10)?;
            let polished = newton_stationary(&bfgs.x, fgh, self.maxiter, self.gtol)?;
            let conv = polished.converged || bfgs.converged;
            if self.loglike(&polished.x) >= self.loglike(&bfgs.x) - 1e-6 {
                opt = polished;
            } else {
                opt.x = bfgs.x;
            }
            opt.converged = conv;
        }
        let params = opt.x;

        // Covariance: inverse of the negative numeric Hessian of the
        // log-likelihood in the transformed parameter space (reference parity).
        let h = approx_hess(&params, |p| self.loglike(p));
        let neg_h = h.mapv(|v| -v);
        let cov = solow_linalg::inv(&neg_h)?;

        Ok(OrderedResults::new(self, params, cov, opt.converged))
    }
}

/// Central-difference Jacobian of a vector field `g` at `x` (used as the
/// Hessian of the negative log-likelihood in the Newton polish). Costs `2·p`
/// gradient evaluations and is symmetrized for stability.
fn score_jacobian<G>(x: &Array1<f64>, g: &G) -> Array2<f64>
where
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = x.len();
    let mut h = Array2::<f64>::zeros((p, p));
    let base = 1e-6;
    let mut xp = x.clone();
    for j in 0..p {
        let step = base * (1.0 + x[j].abs());
        xp[j] = x[j] + step;
        let gp = g(&xp);
        xp[j] = x[j] - step;
        let gm = g(&xp);
        xp[j] = x[j];
        for i in 0..p {
            h[[i, j]] = (gp[i] - gm[i]) / (2.0 * step);
        }
    }
    // Symmetrize.
    for i in 0..p {
        for j in (i + 1)..p {
            let v = 0.5 * (h[[i, j]] + h[[j, i]]);
            h[[i, j]] = v;
            h[[j, i]] = v;
        }
    }
    h
}

/// Detect whether any column of `exog` is constant (an intercept).
fn has_constant(exog: &Array2<f64>) -> bool {
    let (_, k) = exog.dim();
    for j in 0..k {
        let col = exog.column(j);
        let Some(&first) = col.iter().next() else {
            continue;
        };
        if first != 0.0 && col.iter().all(|&v| v == first) {
            return true;
        }
    }
    false
}

/// The fitted result of an ordinal regression model.
#[derive(Clone, Debug)]
pub struct OrderedResults {
    /// Free parameters `[β, cutpoint-transforms]` in the reference layout.
    pub params: Array1<f64>,
    /// Standard errors `√diag((−H)^{-1})` for the free parameters.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`.
    pub pvalues: Array1<f64>,

    /// Back-transformed increasing thresholds `[a₀, …, a_{K−2}]`.
    pub thresholds: Array1<f64>,

    /// Number of observations.
    pub nobs: f64,
    /// Number of ordered levels `K`.
    pub k_levels: usize,
    /// Model degrees of freedom (`k_exog`).
    pub df_model: f64,
    /// Residual degrees of freedom (`nobs − k_exog − (K−1)`).
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

    /// Predicted choice probabilities, shape `n × K`.
    pub predicted: Array2<f64>,

    /// Parameter covariance `(−H)^{-1}` for the free parameters.
    pub cov_params: Array2<f64>,

    /// Whether the optimizer converged.
    pub converged: bool,
}

impl OrderedResults {
    fn new(
        model: &OrderedModel,
        params: Array1<f64>,
        cov_params: Array2<f64>,
        converged: bool,
    ) -> OrderedResults {
        let nobs = model.endog.len() as f64;
        let kv = model.k_vars();
        let kl = model.k_levels;
        let k_params = params.len(); // k_exog + (K-1)

        let mut bse = Array1::<f64>::zeros(k_params);
        for i in 0..k_params {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        let thresholds = model.thresholds(&params);

        let df_model = kv as f64;
        let df_resid = nobs - (kv as f64 + (kl as f64 - 1.0));

        let llf = model.loglike(&params);
        let llnull = model.llnull();
        let llr = 2.0 * (llf - llnull);
        let llr_pvalue = chi2_sf(llr, df_model);
        let prsquared = 1.0 - llf / llnull;

        let kp = k_params as f64;
        let aic = -2.0 * llf + 2.0 * kp;
        let bic = -2.0 * llf + kp * nobs.ln();

        // Full predicted-probability matrix n × K.
        let thr = &thresholds;
        let eta = model.linpred(&params);
        let n = model.endog.len();
        let mut predicted = Array2::<f64>::zeros((n, kl));
        for i in 0..n {
            for c in 0..kl {
                let upp = if c == kl - 1 {
                    1.0
                } else {
                    model.cdf(thr[c] - eta[i])
                };
                let low = if c == 0 {
                    0.0
                } else {
                    model.cdf(thr[c - 1] - eta[i])
                };
                predicted[[i, c]] = upp - low;
            }
        }

        OrderedResults {
            params,
            bse,
            tvalues,
            pvalues,
            thresholds: thresholds.clone(),
            nobs,
            k_levels: kl,
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

    /// Confidence interval for each free parameter at level `1 − alpha`.
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
        let x1 = array![
            0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4, 1.1, -0.9, 0.8, -0.3
        ];
        let x2 = array![
            -0.3, 0.8, 1.1, -0.5, 0.2, -1.0, 0.4, 0.6, -0.8, 1.3, -0.6, 0.5, -0.2, 0.9, -1.1, 0.7
        ];
        let mut x = Array2::<f64>::zeros((16, 2));
        x.column_mut(0).assign(&x1);
        x.column_mut(1).assign(&x2);
        let y = array![0., 2., 1., 0., 1., 0., 1., 2., 0., 2., 1., 1., 2., 0., 0., 2.];
        (y, x)
    }

    #[test]
    fn score_matches_finite_difference() {
        let (y, x) = design();
        let m = OrderedModel::new(y, x).unwrap();
        let p = array![0.4, -0.2, -0.6, 0.3];
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
    fn thresholds_increasing() {
        let (y, x) = design();
        let m = OrderedModel::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        for w in res.thresholds.windows(2).into_iter() {
            assert!(w[1] > w[0], "thresholds not increasing");
        }
    }

    #[test]
    fn score_zero_at_optimum() {
        let (y, x) = design();
        let m = OrderedModel::new(y, x).unwrap();
        let res = m.fit().unwrap();
        let g = m.score(&res.params);
        assert!(g.dot(&g).sqrt() < 1e-5, "score norm {}", g.dot(&g).sqrt());
    }

    #[test]
    fn probit_runs_and_probs_sum_to_one() {
        let (y, x) = design();
        let m = OrderedModel::with_distr(y, x, Distr::Probit).unwrap();
        let res = m.fit().unwrap();
        for i in 0..res.predicted.nrows() {
            let s: f64 = res.predicted.row(i).sum();
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn rejects_constant_column() {
        let mut x = Array2::<f64>::ones((6, 2));
        x.column_mut(1)
            .assign(&array![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]);
        let y = array![0., 1., 2., 0., 1., 2.];
        assert!(OrderedModel::new(y, x).is_err());
    }
}
