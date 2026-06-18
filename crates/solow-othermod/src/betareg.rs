//! Beta regression (the `BetaModel` of the reference modeling package).
//!
//! Both the conditional mean `μ ∈ (0, 1)` and the precision `φ > 0` are modeled
//! with their own linear predictors and link functions:
//!
//! ```text
//! g_μ(μ_i) = x_iᵀ β ,   g_φ(φ_i) = z_iᵀ γ
//! ```
//!
//! with the default mean link `logit` and the default precision link `log`. The
//! response is Beta-distributed with shape parameters `a = μφ` and `b = (1−μ)φ`,
//! so that `E[y] = μ` and `Var[y] = μ(1−μ)/(1+φ)`. Estimation is by maximum
//! likelihood; the per-observation log-likelihood is
//!
//! ```text
//! ℓ_i = lnΓ(φ) − lnΓ(μφ) − lnΓ((1−μ)φ)
//!       + (μφ − 1) ln y_i + ((1−μ)φ − 1) ln(1 − y_i).
//! ```
//!
//! The parameter vector concatenates the mean coefficients `β` first and the
//! precision coefficients `γ` second, matching the reference ordering.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::{digamma, lgamma, norm_ppf, norm_sf};
use solow_linalg::{inv, lstsq};
use solow_optimize::{approx_fprime, minimize_bfgs, newton_stationary};

/// A monotone link mapping a model quantity to its linear predictor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    /// `logit(p) = ln(p / (1 − p))`; inverse is the logistic function. Default
    /// link for the mean submodel.
    Logit,
    /// `log(x)`; inverse is `exp`. Default link for the precision submodel.
    Log,
}

/// `f64::EPSILON`; the reference clips link arguments to `[FLOAT_EPS, ∞)`
/// (or `[FLOAT_EPS, 1 − FLOAT_EPS]` for logit) before applying the transform.
const FLOAT_EPS: f64 = f64::EPSILON;

impl Link {
    /// The link `g(v)` mapping the quantity onto the linear-predictor scale.
    ///
    /// The argument is first cleaned into the link's valid domain, matching the
    /// reference's `_clean`, so that out-of-range starting-value proxies (e.g. a
    /// negative precision estimate) map to a large finite linear predictor rather
    /// than a NaN.
    fn link(self, v: f64) -> f64 {
        match self {
            Link::Logit => {
                let p = v.clamp(FLOAT_EPS, 1.0 - FLOAT_EPS);
                (p / (1.0 - p)).ln()
            }
            Link::Log => v.max(FLOAT_EPS).ln(),
        }
    }

    /// The inverse link `g⁻¹(η)` mapping a linear predictor back to the quantity.
    fn inverse(self, eta: f64) -> f64 {
        match self {
            Link::Logit => 1.0 / (1.0 + (-eta).exp()),
            Link::Log => eta.exp(),
        }
    }
}

/// A beta-regression model awaiting estimation.
#[derive(Clone, Debug)]
pub struct BetaModel {
    endog: Array1<f64>,
    exog: Array2<f64>,
    exog_precision: Array2<f64>,
    link: Link,
    link_precision: Link,
    maxiter: usize,
    gtol: f64,
}

impl BetaModel {
    /// A beta-regression model with the default links (`logit` for the mean,
    /// `log` for the precision).
    ///
    /// `endog` must lie strictly in `(0, 1)`. `exog` and `exog_precision` are the
    /// design matrices for the mean and precision submodels; both must have the
    /// same number of rows as `endog`.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>, exog_precision: Array2<f64>) -> Result<Self> {
        Self::with_links(endog, exog, exog_precision, Link::Logit, Link::Log)
    }

    /// A beta-regression model with explicit mean and precision links.
    pub fn with_links(
        endog: Array1<f64>,
        exog: Array2<f64>,
        exog_precision: Array2<f64>,
        link: Link,
        link_precision: Link,
    ) -> Result<Self> {
        let n = endog.len();
        if exog.nrows() != n {
            return Err(Error::Shape("exog rows != endog length".into()));
        }
        if exog_precision.nrows() != n {
            return Err(Error::Shape("exog_precision rows != endog length".into()));
        }
        if endog.iter().any(|&y| !(y > 0.0 && y < 1.0)) {
            return Err(Error::Shape("endog must lie strictly in (0, 1)".into()));
        }
        Ok(BetaModel {
            endog,
            exog,
            exog_precision,
            link,
            link_precision,
            maxiter: 1000,
            gtol: 1e-10,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Number of mean coefficients.
    fn k_mean(&self) -> usize {
        self.exog.ncols()
    }

    /// Number of precision coefficients.
    fn k_prec(&self) -> usize {
        self.exog_precision.ncols()
    }

    /// Split a stacked parameter vector into `(μ, φ)` evaluated per observation.
    fn mu_phi(&self, params: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let km = self.k_mean();
        let beta = params.slice(ndarray::s![..km]).to_owned();
        let gamma = params.slice(ndarray::s![km..]).to_owned();
        let lin = self.exog.dot(&beta);
        let lin_prec = self.exog_precision.dot(&gamma);
        let mu = lin.mapv(|e| self.link.inverse(e));
        let phi = lin_prec.mapv(|e| self.link_precision.inverse(e));
        (mu, phi)
    }

    /// Total log-likelihood at `params`.
    pub fn loglike(&self, params: &Array1<f64>) -> f64 {
        let (mu, phi) = self.mu_phi(params);
        let mut ll = 0.0;
        for i in 0..self.endog.len() {
            let y = self.endog[i];
            let m = mu[i];
            let p = phi[i];
            let alpha = (m * p).max(1e-200);
            let beta = ((1.0 - m) * p).max(1e-200);
            ll += lgamma(p) - lgamma(alpha) - lgamma(beta)
                + (m * p - 1.0) * y.ln()
                + ((1.0 - m) * p - 1.0) * (1.0 - y).ln();
        }
        ll
    }

    /// Analytic score (gradient of the log-likelihood) at `params`.
    ///
    /// The derivative w.r.t. each linear predictor is formed in closed form using
    /// `digamma`, then chained through the link derivatives and the design
    /// matrices. Coefficient order is `[∂/∂β, ∂/∂γ]`.
    pub fn score(&self, params: &Array1<f64>) -> Array1<f64> {
        let (mu, phi) = self.mu_phi(params);
        let n = self.endog.len();
        let km = self.k_mean();
        let kp = self.k_prec();
        let mut g = Array1::<f64>::zeros(km + kp);
        for i in 0..n {
            let y = self.endog[i];
            let m = mu[i];
            let p = phi[i];
            let alpha = (m * p).max(1e-200);
            let beta = ((1.0 - m) * p).max(1e-200);

            let ystar = (y / (1.0 - y)).ln();
            let yt = (1.0 - y).ln();
            let dig_beta = digamma(beta);
            let mustar = digamma(alpha) - dig_beta;
            let mut_ = dig_beta - digamma(p);

            // 1 / g'(·): the derivative of the inverse link w.r.t. the linear
            // predictor. For logit: μ(1−μ); for log: φ.
            let t = match self.link {
                Link::Logit => m * (1.0 - m),
                Link::Log => m,
            };
            let h = match self.link_precision {
                Link::Logit => p * (1.0 - p),
                Link::Log => p,
            };

            // dℓ/dη_μ and dℓ/dη_φ.
            let sf1 = p * t * (ystar - mustar);
            let sf2 = h * (m * (ystar - mustar) + yt - mut_);

            for j in 0..km {
                g[j] += sf1 * self.exog[[i, j]];
            }
            for j in 0..kp {
                g[km + j] += sf2 * self.exog_precision[[i, j]];
            }
        }
        g
    }

    /// Weighted-least-squares starting values, mirroring the reference's
    /// `_start_params` (two refinement iterations).
    fn start_params(&self) -> Array1<f64> {
        let n = self.endog.len();
        let km = self.k_mean();
        let kp = self.k_prec();

        // Mean equation: OLS of g_μ(y) on X.
        let g_y = self.endog.mapv(|y| self.link.link(y));
        let beta = ols(&self.exog, &g_y);
        let mut fitted = self.exog.dot(&beta).mapv(|e| self.link.inverse(e));

        // Precision equation: OLS of g_φ(prec_i) on Z.
        let prec_i = precision_proxy(&self.endog, &fitted);
        let g_prec = prec_i.mapv(|v| self.link_precision.link(v));
        let mut gamma = ols(&self.exog_precision, &g_prec);
        let mut prec_fitted = self
            .exog_precision
            .dot(&gamma)
            .mapv(|e| self.link_precision.inverse(e));

        for _ in 0..2 {
            // Reweighted mean equation.
            let mut w_m = Array1::<f64>::zeros(n);
            for i in 0..n {
                let f = fitted[i];
                let y_var_inv = (1.0 + prec_fitted[i]) / (f * (1.0 - f));
                let dlink = match self.link {
                    Link::Logit => 1.0 / (f * (1.0 - f)),
                    Link::Log => 1.0 / f,
                };
                w_m[i] = y_var_inv / (dlink * dlink);
            }
            let beta2 = wls(&self.exog, &g_y, &w_m);
            fitted = self.exog.dot(&beta2).mapv(|e| self.link.inverse(e));

            // Reweighted precision equation.
            let prec_i2 = precision_proxy(&self.endog, &fitted);
            let g_prec2 = prec_i2.mapv(|v| self.link_precision.link(v));
            let mut w_p = Array1::<f64>::zeros(n);
            for i in 0..n {
                let dlink = match self.link_precision {
                    Link::Logit => 1.0 / (prec_fitted[i] * (1.0 - prec_fitted[i])),
                    Link::Log => 1.0 / prec_fitted[i],
                };
                w_p[i] = 1.0 / (dlink * dlink);
            }
            let gamma2 = wls(&self.exog_precision, &g_prec2, &w_p);
            prec_fitted = self
                .exog_precision
                .dot(&gamma2)
                .mapv(|e| self.link_precision.inverse(e));
            gamma = gamma2;
        }

        let beta_final = wls_final_mean(self, &g_y, &fitted, &prec_fitted);
        let mut out = Array1::<f64>::zeros(km + kp);
        for j in 0..km {
            out[j] = beta_final[j];
        }
        for j in 0..kp {
            out[km + j] = gamma[j];
        }
        out
    }

    /// Estimate the model by maximum likelihood.
    ///
    /// BFGS is run first from WLS starting values to get into the basin of the
    /// optimum, then a short Newton phase (driving the analytic score to zero
    /// with a finite-difference Hessian of the score) polishes to full
    /// convergence. The coefficient covariance is the inverse of the observed
    /// information, `−H⁻¹`, where `H` is the Hessian of the log-likelihood
    /// obtained by differentiating the analytic score.
    pub fn fit(&self) -> Result<BetaResults> {
        let start = self.start_params();

        // Minimize the negative log-likelihood with the analytic gradient.
        let neg_ll = |p: &Array1<f64>| -self.loglike(p);
        let neg_score = |p: &Array1<f64>| self.score(p).mapv(|v| -v);
        let bfgs = minimize_bfgs(&start, neg_ll, neg_score, self.maxiter, 1e-8)?;

        // Polish: drive the analytic score to zero with a Newton step whose
        // Hessian is the (finite-difference) Jacobian of the analytic score.
        let fgh = |p: &Array1<f64>| {
            let f = -self.loglike(p);
            let g = self.score(p).mapv(|v| -v);
            let hess = self.neg_hessian(p);
            (f, g, hess)
        };
        let newton = newton_stationary(&bfgs.x, fgh, 100, self.gtol)?;

        let params = newton.x;
        let converged = newton.converged || newton.grad_norm < 1e-6;

        // Observed information: H = ∂score/∂params (symmetric); cov = −H⁻¹.
        let neg_h = self.neg_hessian(&params);
        let cov = inv(&neg_h)?;

        let k = params.len();
        let mut bse = Array1::<f64>::zeros(k);
        for i in 0..k {
            bse[i] = cov[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        let (mu, _phi) = self.mu_phi(&params);
        let llf = self.loglike(&params);

        Ok(BetaResults {
            params,
            bse,
            tvalues,
            pvalues,
            cov_params: cov,
            llf,
            fittedvalues: mu,
            k_mean: self.k_mean(),
            k_prec: self.k_prec(),
            nobs: self.nobs() as f64,
            converged,
        })
    }

    /// `−H` where `H` is the Hessian of the log-likelihood, formed as the
    /// (symmetrized) finite-difference Jacobian of the analytic score. This
    /// equals the observed information matrix.
    fn neg_hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let k = params.len();
        // Jacobian of the score: column j is ∂score/∂param_j.
        let mut jac = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            let component = |p: &Array1<f64>| self.score(p)[j];
            let row = approx_fprime(params, component); // ∂score_j/∂params
            for i in 0..k {
                jac[[j, i]] = row[i];
            }
        }
        // Symmetrize and negate: observed information = −∂²ℓ/∂θ∂θᵀ.
        let mut nh = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                nh[[i, j]] = -(jac[[i, j]] + jac[[j, i]]) / 2.0;
            }
        }
        nh
    }
}

/// Per-observation precision proxy used by the WLS starting-value scheme:
/// `μ(1−μ)/max(|y−μ|, 1e-2)² − 1`.
fn precision_proxy(y: &Array1<f64>, fitted: &Array1<f64>) -> Array1<f64> {
    let n = y.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let f = fitted[i];
        let resid = (y[i] - f).abs().max(1e-2);
        out[i] = f * (1.0 - f) / (resid * resid) - 1.0;
    }
    out
}

/// Ordinary least squares `(XᵀX)⁻¹Xᵀy` via a least-squares solve.
fn ols(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    lstsq(&xtx, &xty).expect("normal equations solvable")
}

/// Weighted least squares with diagonal weights `w`.
fn wls(x: &Array2<f64>, y: &Array1<f64>, w: &Array1<f64>) -> Array1<f64> {
    let (n, p) = x.dim();
    let mut xtwx = Array2::<f64>::zeros((p, p));
    let mut xtwy = Array1::<f64>::zeros(p);
    for i in 0..n {
        let wi = w[i];
        for a in 0..p {
            xtwy[a] += wi * x[[i, a]] * y[i];
            for b in 0..p {
                xtwx[[a, b]] += wi * x[[i, a]] * x[[i, b]];
            }
        }
    }
    lstsq(&xtwx, &xtwy).expect("weighted normal equations solvable")
}

/// Final mean-equation WLS using the implied inverse-variance weights, matching
/// the reference's last `res_m2` step (the precision equation's last fit is the
/// returned `gamma`, while the mean fit gets one more reweight).
fn wls_final_mean(
    model: &BetaModel,
    g_y: &Array1<f64>,
    fitted: &Array1<f64>,
    prec_fitted: &Array1<f64>,
) -> Array1<f64> {
    let n = model.endog.len();
    let mut w_m = Array1::<f64>::zeros(n);
    for i in 0..n {
        let f = fitted[i];
        let y_var_inv = (1.0 + prec_fitted[i]) / (f * (1.0 - f));
        let dlink = match model.link {
            Link::Logit => 1.0 / (f * (1.0 - f)),
            Link::Log => 1.0 / f,
        };
        w_m[i] = y_var_inv / (dlink * dlink);
    }
    wls(&model.exog, g_y, &w_m)
}

/// The fitted result of a [`BetaModel`].
#[derive(Clone, Debug)]
pub struct BetaResults {
    /// Estimated coefficients: mean coefficients first, precision coefficients
    /// second.
    pub params: Array1<f64>,
    /// Standard errors `sqrt(diag(−H⁻¹))`.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values from the standard normal.
    pub pvalues: Array1<f64>,
    /// Coefficient covariance matrix `−H⁻¹`.
    pub cov_params: Array2<f64>,
    /// Maximized log-likelihood.
    pub llf: f64,
    /// Fitted conditional means `μ`.
    pub fittedvalues: Array1<f64>,
    /// Number of mean coefficients.
    pub k_mean: usize,
    /// Number of precision coefficients.
    pub k_prec: usize,
    /// Number of observations.
    pub nobs: f64,
    /// Whether the MLE converged.
    pub converged: bool,
}

impl BetaResults {
    /// The mean-submodel coefficients `β`.
    pub fn params_mean(&self) -> Array1<f64> {
        self.params.slice(ndarray::s![..self.k_mean]).to_owned()
    }

    /// The precision-submodel coefficients `γ`.
    pub fn params_precision(&self) -> Array1<f64> {
        self.params.slice(ndarray::s![self.k_mean..]).to_owned()
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Build a small deterministic beta-regression data set with a known DGP.
    fn toy() -> (Array1<f64>, Array2<f64>, Array2<f64>) {
        // 20 observations, intercept + one mean covariate, intercept + one
        // precision covariate. Responses computed from the mean of the Beta.
        let xs = [
            -1.2, -0.7, -0.3, 0.1, 0.4, 0.9, 1.3, -0.5, 0.2, 0.6, -1.0, 0.7, 1.1, -0.2, 0.3, -0.8,
            0.5, 1.0, -0.4, 0.8,
        ];
        let zs = [
            0.3, -0.2, 0.5, -0.6, 0.1, 0.4, -0.3, 0.2, -0.5, 0.6, 0.0, -0.1, 0.3, -0.4, 0.2, 0.5,
            -0.2, 0.1, 0.4, -0.3,
        ];
        let n = xs.len();
        let mut x = Array2::<f64>::ones((n, 2));
        let mut z = Array2::<f64>::ones((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            x[[i, 1]] = xs[i];
            z[[i, 1]] = zs[i];
            let eta = 0.3 + 0.8 * xs[i];
            let mu = 1.0 / (1.0 + (-eta).exp());
            // Map the mean into a clean (0,1) response perturbed mildly so the
            // MLE is well defined but deterministic.
            let jitter = 0.03 * ((i as f64 * 1.7).sin());
            y[i] = (mu + jitter).clamp(0.02, 0.98);
        }
        (y, x, z)
    }

    #[test]
    fn links_roundtrip() {
        for &v in &[0.1, 0.5, 0.9] {
            let e = Link::Logit.link(v);
            assert!((Link::Logit.inverse(e) - v).abs() < 1e-12);
        }
        for &v in &[0.5, 1.0, 5.0] {
            let e = Link::Log.link(v);
            assert!((Link::Log.inverse(e) - v).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_out_of_range_endog() {
        let y = array![0.5, 1.0, 0.3];
        let x = Array2::<f64>::ones((3, 1));
        let z = Array2::<f64>::ones((3, 1));
        assert!(BetaModel::new(y, x, z).is_err());
    }

    #[test]
    fn score_matches_finite_difference() {
        let (y, x, z) = toy();
        let m = BetaModel::new(y, x, z).unwrap();
        let p = array![0.2, 0.5, 2.0, 0.3];
        let analytic = m.score(&p);
        let numeric = approx_fprime(&p, |q| m.loglike(q));
        for i in 0..p.len() {
            let e = (analytic[i] - numeric[i]).abs() / (1.0 + numeric[i].abs());
            assert!(e < 1e-5, "score[{i}] mismatch: {analytic} vs {numeric}");
        }
    }

    #[test]
    fn fit_converges_and_score_is_zero() {
        let (y, x, z) = toy();
        let m = BetaModel::new(y, x, z).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let g = m.score(&res.params);
        let gnorm = g.dot(&g).sqrt();
        assert!(gnorm < 1e-6, "score not zero at optimum: {gnorm}");
        // Fitted means must be valid probabilities.
        for &f in res.fittedvalues.iter() {
            assert!(f > 0.0 && f < 1.0);
        }
        // Splitting params reproduces the full vector.
        assert_eq!(res.params_mean().len(), 2);
        assert_eq!(res.params_precision().len(), 2);
    }
}
