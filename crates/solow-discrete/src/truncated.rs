//! Zero-truncated Poisson and Poisson hurdle count models.
//!
//! * [`TruncatedLFPoisson`] — a **zero-truncated** Poisson regression. The data
//!   contain only strictly positive counts (any `y = 0` rows are discarded), and
//!   the likelihood renormalizes the Poisson mass onto the support `y ≥ 1`:
//!   ```text
//!   ℓᵢ = −μᵢ + yᵢ log μᵢ − log(yᵢ!) − log(1 − e^{−μᵢ}),   μᵢ = exp(xᵢ·β).
//!   ```
//! * [`HurdleCountModel`] — a Poisson **hurdle** model: a binary "zero hurdle"
//!   model for whether the count crosses zero, combined with a zero-truncated
//!   Poisson for the positive counts. The two parts share no parameters, so the
//!   full log-likelihood separates and the two components are estimated
//!   independently:
//!   ```text
//!   ℓ = ℓ_zero(β_z) + ℓ_trunc(β_c),
//!   ℓ_zero,i = −μ_{z,i}                 if yᵢ = 0   (log P(Y=0)),
//!   ℓ_zero,i = log(1 − e^{−μ_{z,i}})    if yᵢ > 0   (log P(Y>0)),
//!   ```
//!   with the positive counts entering `ℓ_trunc` exactly as in
//!   [`TruncatedLFPoisson`]. The reported parameter vector is the concatenation
//!   `[β_z, β_c]` and the covariance is block-diagonal across the two parts,
//!   matching the reference.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{lgamma, norm_ppf, norm_sf};
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// `log(1 − e^{−m})` evaluated stably for `m > 0` (`log(-expm1(-m))`).
fn log1m_exp_neg(m: f64) -> f64 {
    // -expm1(-m) = 1 - e^{-m} ∈ (0,1) for m>0.
    (-((-m).exp_m1())).ln()
}

// =========================================================================== //
//  Zero-truncated Poisson                                                     //
// =========================================================================== //

/// A zero-truncated Poisson model awaiting estimation.
///
/// Construct with [`TruncatedLFPoisson::new`]; estimate with
/// [`TruncatedLFPoisson::fit`]. Rows with `y = 0` are dropped on construction
/// (the model has support `y ≥ 1`).
#[derive(Clone, Debug)]
pub struct TruncatedLFPoisson {
    endog: Array1<f64>,
    exog: Array2<f64>,
    k_constant: usize,
    maxiter: usize,
    gtol: f64,
}

impl TruncatedLFPoisson {
    /// Build a zero-truncated Poisson model.
    ///
    /// `exog` should already contain an intercept column if one is wanted.
    /// Observations with `y = 0` are discarded. Returns an error on shape
    /// mismatch or if no positive observations remain.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        let k = exog.ncols();
        let keep: Vec<usize> = (0..endog.len()).filter(|&i| endog[i] > 0.0).collect();
        if keep.is_empty() {
            return Err(Error::Shape(
                "no positive observations to truncate on".into(),
            ));
        }
        let mut y = Array1::<f64>::zeros(keep.len());
        let mut x = Array2::<f64>::zeros((keep.len(), k));
        for (r, &i) in keep.iter().enumerate() {
            y[r] = endog[i];
            for c in 0..k {
                x[[r, c]] = exog[[i, c]];
            }
        }
        let k_constant = detect_k_constant(&x);
        Ok(TruncatedLFPoisson {
            endog: y,
            exog: x,
            k_constant,
            maxiter: 200,
            gtol: 1e-12,
        })
    }

    /// Number of (positive) observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    fn mean(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.exog.dot(beta).mapv(f64::exp)
    }

    fn loglike(&self, beta: &Array1<f64>) -> f64 {
        let mu = self.mean(beta);
        let y = &self.endog;
        let mut ll = 0.0;
        for i in 0..y.len() {
            ll += -mu[i] + y[i] * mu[i].ln() - lgamma(y[i] + 1.0) - log1m_exp_neg(mu[i]);
        }
        ll
    }

    /// Analytic score: per obs weight `wᵢ = yᵢ − μᵢ − μᵢ e^{−μᵢ}/(1 − e^{−μᵢ})`
    /// on the row `xᵢ` (chain rule through `dμ/dβ = μ x`).
    fn score(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mu = self.mean(beta);
        let y = &self.endog;
        let n = y.len();
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let em = (-mu[i]).exp();
            let s = -(-mu[i]).exp_m1(); // 1 - e^{-mu}
            w[i] = y[i] - mu[i] - mu[i] * em / s;
        }
        self.exog.t().dot(&w)
    }

    /// Analytic Hessian `Σ dᵢ xᵢ xᵢᵀ`, `dᵢ = (−1 − da/dμ) μᵢ` where
    /// `a = μ e^{−μ}/(1 − e^{−μ})` is the truncation correction.
    fn hessian(&self, beta: &Array1<f64>) -> Array2<f64> {
        let mu = self.mean(beta);
        let n = self.endog.len();
        let mut d = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i];
            let em = (-m).exp();
            let s = -(-m).exp_m1(); // 1 - e^{-m}
                                    // a = m e^{-m}/s ; da/dm = e^{-m}/s - m e^{-m}/s^2
            let da_dm = em / s - m * em / (s * s);
            d[i] = (-1.0 - da_dm) * m;
        }
        weighted_gram(&self.exog, &d)
    }

    fn start(&self) -> Array1<f64> {
        // Plain Poisson Newton on the kept (positive) data is a good start.
        poisson_newton(&self.endog, &self.exog, 50)
    }

    fn fit_params(&self) -> Result<(Array1<f64>, Array2<f64>, bool)> {
        let fgh = |b: &Array1<f64>| {
            let f = -self.loglike(b);
            let g = self.score(b).mapv(|v| -v);
            let h = self.hessian(b).mapv(|v| -v);
            (f, g, h)
        };
        let opt = newton_stationary(&self.start(), fgh, self.maxiter, self.gtol)?;
        let params = opt.x;
        let neg_h = self.hessian(&params).mapv(|v| -v);
        let cov = inv(&neg_h)?;
        Ok((params, cov, opt.converged))
    }

    /// Estimate by full Newton steps and assemble [`TruncatedLFPoissonResults`].
    pub fn fit(&self) -> Result<TruncatedLFPoissonResults> {
        let (params, cov, converged) = self.fit_params()?;
        Ok(TruncatedLFPoissonResults::new(self, params, cov, converged))
    }
}

/// The fitted result of a zero-truncated Poisson model.
#[derive(Clone, Debug)]
pub struct TruncatedLFPoissonResults {
    /// Estimated coefficients `β̂`.
    pub params: Array1<f64>,
    /// Standard errors `√diag((−H)⁻¹)`.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`.
    pub pvalues: Array1<f64>,

    /// Number of (positive) observations.
    pub nobs: f64,
    /// Whether a constant is present (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom (regressors excluding the constant).
    pub df_model: f64,
    /// Residual degrees of freedom `nobs − (df_model + 1)`.
    pub df_resid: f64,

    /// Maximized log-likelihood.
    pub llf: f64,
    /// Akaike information criterion `−2 llf + 2 k_params`.
    pub aic: f64,
    /// Bayesian information criterion `−2 llf + k_params ln n`.
    pub bic: f64,

    /// Conditional mean of the truncated distribution `μ/(1 − e^{−μ})`.
    pub predicted: Array1<f64>,

    /// Coefficient covariance `(−H)⁻¹`.
    pub cov_params: Array2<f64>,

    /// Whether Newton converged.
    pub converged: bool,
}

impl TruncatedLFPoissonResults {
    fn new(
        model: &TruncatedLFPoisson,
        params: Array1<f64>,
        cov_params: Array2<f64>,
        converged: bool,
    ) -> TruncatedLFPoissonResults {
        let nobs = model.endog.len() as f64;
        let k = params.len();
        let df_model = k as f64 - model.k_constant as f64;
        let df_resid = nobs - df_model - 1.0;

        let mut bse = Array1::<f64>::zeros(k);
        for i in 0..k {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        let llf = model.loglike(&params);
        let kp = k as f64;
        let aic = -2.0 * llf + 2.0 * kp;
        let bic = -2.0 * llf + kp * nobs.ln();

        // Mean of the zero-truncated Poisson: μ / (1 − e^{−μ}).
        let mu = model.mean(&params);
        let predicted = mu.mapv(|m| m / -(-m).exp_m1());

        TruncatedLFPoissonResults {
            params,
            bse,
            tvalues,
            pvalues,
            nobs,
            k_constant: model.k_constant,
            df_model,
            df_resid,
            llf,
            aic,
            bic,
            predicted,
            cov_params,
            converged,
        }
    }

    /// Confidence interval for each coefficient at level `1 − alpha` (normal).
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        conf_int(&self.params, &self.bse, alpha)
    }
}

// =========================================================================== //
//  Binary zero-hurdle model (internal building block)                         //
// =========================================================================== //

/// The binary "zero hurdle" model: `P(Y=0) = e^{−μ}`, `P(Y>0) = 1 − e^{−μ}`,
/// with `μ = exp(Xβ)`. This is the zero part of the Poisson hurdle.
#[derive(Clone, Debug)]
struct ZeroHurdlePoisson {
    /// `1` if the original count was positive, else `0`.
    positive: Array1<f64>,
    exog: Array2<f64>,
    maxiter: usize,
    gtol: f64,
}

impl ZeroHurdlePoisson {
    fn new(endog: &Array1<f64>, exog: &Array2<f64>) -> ZeroHurdlePoisson {
        let positive = endog.mapv(|y| if y > 0.0 { 1.0 } else { 0.0 });
        ZeroHurdlePoisson {
            positive,
            exog: exog.clone(),
            maxiter: 200,
            gtol: 1e-12,
        }
    }

    fn mean(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.exog.dot(beta).mapv(f64::exp)
    }

    fn loglike(&self, beta: &Array1<f64>) -> f64 {
        let mu = self.mean(beta);
        let mut ll = 0.0;
        for i in 0..mu.len() {
            ll += if self.positive[i] > 0.5 {
                log1m_exp_neg(mu[i]) // log(1 - e^{-mu})
            } else {
                -mu[i] // log(e^{-mu})
            };
        }
        ll
    }

    /// Score: weight `wᵢ` on row `xᵢ`, `wᵢ = −μ` for zeros and
    /// `μ e^{−μ}/(1 − e^{−μ})` for positives (chain rule, `dμ/dβ = μx`).
    fn score(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mu = self.mean(beta);
        let n = mu.len();
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            if self.positive[i] > 0.5 {
                let em = (-mu[i]).exp();
                let s = -(-mu[i]).exp_m1();
                w[i] = mu[i] * em / s;
            } else {
                w[i] = -mu[i];
            }
        }
        self.exog.t().dot(&w)
    }

    /// Hessian `Σ dᵢ xᵢ xᵢᵀ`. For zeros `dᵢ = −μ`; for positives
    /// `dᵢ = (da/dμ) μ` with `a = μ e^{−μ}/(1 − e^{−μ})`.
    fn hessian(&self, beta: &Array1<f64>) -> Array2<f64> {
        let mu = self.mean(beta);
        let n = mu.len();
        let mut d = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i];
            if self.positive[i] > 0.5 {
                let em = (-m).exp();
                let s = -(-m).exp_m1();
                let da_dm = em / s - m * em / (s * s);
                d[i] = da_dm * m;
            } else {
                d[i] = -m;
            }
        }
        weighted_gram(&self.exog, &d)
    }

    fn start(&self) -> Array1<f64> {
        // Poisson Newton on the original counts is the reference's start too.
        Array1::<f64>::zeros(self.exog.ncols())
    }

    fn fit_params(&self) -> Result<(Array1<f64>, Array2<f64>, bool)> {
        let fgh = |b: &Array1<f64>| {
            let f = -self.loglike(b);
            let g = self.score(b).mapv(|v| -v);
            let h = self.hessian(b).mapv(|v| -v);
            (f, g, h)
        };
        let opt = newton_stationary(&self.start(), fgh, self.maxiter, self.gtol)?;
        let params = opt.x;
        let neg_h = self.hessian(&params).mapv(|v| -v);
        let cov = inv(&neg_h)?;
        Ok((params, cov, opt.converged))
    }
}

// =========================================================================== //
//  Poisson hurdle model                                                       //
// =========================================================================== //

/// A Poisson hurdle count model awaiting estimation.
///
/// Construct with [`HurdleCountModel::new`]; estimate with
/// [`HurdleCountModel::fit`]. Both the zero hurdle and the truncated count parts
/// use the same design `exog` (which should already include any intercept).
#[derive(Clone, Debug)]
pub struct HurdleCountModel {
    endog: Array1<f64>,
    exog: Array2<f64>,
    k_constant: usize,
}

impl HurdleCountModel {
    /// Build a Poisson hurdle model (Poisson zero model + zero-truncated Poisson
    /// count model). Returns an error on shape mismatch.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        let k_constant = detect_k_constant(&exog);
        Ok(HurdleCountModel {
            endog,
            exog,
            k_constant,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Estimate the two independent parts and assemble [`HurdleCountResults`].
    pub fn fit(&self) -> Result<HurdleCountResults> {
        // Zero part: binary hurdle over all observations.
        let zero = ZeroHurdlePoisson::new(&self.endog, &self.exog);
        let (pz, covz, convz) = zero.fit_params()?;
        let llf_zero = zero.loglike(&pz);

        // Count part: zero-truncated Poisson on the positive observations.
        let trunc = TruncatedLFPoisson::new(self.endog.clone(), self.exog.clone())?;
        let (pc, covc, convc) = trunc.fit_params()?;
        let llf_count = trunc.loglike(&pc);

        let kz = pz.len();
        let kc = pc.len();
        let ktot = kz + kc;

        // Concatenate params and block-diagonal covariance.
        let mut params = Array1::<f64>::zeros(ktot);
        for i in 0..kz {
            params[i] = pz[i];
        }
        for i in 0..kc {
            params[kz + i] = pc[i];
        }
        let mut cov = Array2::<f64>::zeros((ktot, ktot));
        for i in 0..kz {
            for j in 0..kz {
                cov[[i, j]] = covz[[i, j]];
            }
        }
        for i in 0..kc {
            for j in 0..kc {
                cov[[kz + i, kz + j]] = covc[[i, j]];
            }
        }

        let nobs = self.endog.len() as f64;
        let mut bse = Array1::<f64>::zeros(ktot);
        for i in 0..ktot {
            bse[i] = cov[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        let llf = llf_zero + llf_count;
        // df_model sums the two parts' df_model (each k − 1 with a constant).
        let df_model = (kz as f64 - self.k_constant as f64) + (kc as f64 - self.k_constant as f64);
        // Two constants among the ktot params (one per part).
        let df_resid = nobs - (df_model + 2.0);

        let kp = ktot as f64;
        let aic = -2.0 * llf + 2.0 * kp;
        let bic = -2.0 * llf + kp * nobs.ln();

        Ok(HurdleCountResults {
            params,
            bse,
            tvalues,
            pvalues,
            params_zero: pz,
            params_count: pc,
            nobs,
            k_constant: self.k_constant,
            df_model,
            df_resid,
            llf,
            llf_zero,
            llf_count,
            aic,
            bic,
            cov_params: cov,
            converged: convz && convc,
        })
    }
}

/// The fitted result of a Poisson hurdle model.
#[derive(Clone, Debug)]
pub struct HurdleCountResults {
    /// Estimated parameters `[β_zero, β_count]`.
    pub params: Array1<f64>,
    /// Standard errors `√diag(cov)` (block-diagonal across the two parts).
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`.
    pub pvalues: Array1<f64>,

    /// Zero-hurdle coefficients only.
    pub params_zero: Array1<f64>,
    /// Truncated-count coefficients only.
    pub params_count: Array1<f64>,

    /// Number of observations.
    pub nobs: f64,
    /// Whether a constant is present in the design (0/1).
    pub k_constant: usize,
    /// Combined model degrees of freedom (sum over the two parts).
    pub df_model: f64,
    /// Residual degrees of freedom.
    pub df_resid: f64,

    /// Maximized total log-likelihood `llf_zero + llf_count`.
    pub llf: f64,
    /// Log-likelihood of the zero-hurdle part.
    pub llf_zero: f64,
    /// Log-likelihood of the truncated-count part.
    pub llf_count: f64,

    /// Akaike information criterion `−2 llf + 2 k_params`.
    pub aic: f64,
    /// Bayesian information criterion `−2 llf + k_params ln n`.
    pub bic: f64,

    /// Block-diagonal parameter covariance.
    pub cov_params: Array2<f64>,

    /// Whether both parts converged.
    pub converged: bool,
}

impl HurdleCountResults {
    /// Confidence interval for each parameter at level `1 − alpha` (normal).
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        conf_int(&self.params, &self.bse, alpha)
    }
}

// --------------------------------------------------------------------------- //
//  Shared helpers                                                             //
// --------------------------------------------------------------------------- //

/// `H = Σ dᵢ xᵢ xᵢᵀ` (the curvature contribution of per-obs weights `d`).
fn weighted_gram(x: &Array2<f64>, d: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut h = Array2::<f64>::zeros((p, p));
    for a in 0..p {
        for b in a..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[[i, a]] * d[i] * x[[i, b]];
            }
            h[[a, b]] = s;
            h[[b, a]] = s;
        }
    }
    h
}

/// Plain Poisson Newton (log link) on `(y, x)` for a starting point.
fn poisson_newton(y: &Array1<f64>, x: &Array2<f64>, iters: usize) -> Array1<f64> {
    let n = y.len();
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    for _ in 0..iters {
        let mu = x.dot(&beta).mapv(f64::exp);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            w[i] = y[i] - mu[i];
        }
        let g = x.t().dot(&w);
        let h = weighted_gram(x, &mu);
        let step = match solow_linalg::solve(&h, &g) {
            Ok(s) => s,
            Err(_) => break,
        };
        beta = &beta + &step;
        if step.dot(&step).sqrt() < 1e-12 {
            break;
        }
    }
    beta
}

fn conf_int(params: &Array1<f64>, bse: &Array1<f64>, alpha: f64) -> Array2<f64> {
    let q = norm_ppf(1.0 - alpha / 2.0);
    let k = params.len();
    let mut out = Array2::<f64>::zeros((k, 2));
    for i in 0..k {
        out[[i, 0]] = params[i] - q * bse[i];
        out[[i, 1]] = params[i] + q * bse[i];
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use solow_optimize::approx_fprime;

    fn count_design() -> (Array1<f64>, Array2<f64>) {
        let xcol = array![
            0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4, 1.1, -0.9, 0.8, -0.3
        ];
        let mut x = Array2::<f64>::ones((16, 2));
        x.column_mut(1).assign(&xcol);
        let y =
            array![0.0, 2.0, 3.0, 8.0, 0.0, 5.0, 1.0, 4.0, 0.0, 3.0, 0.0, 1.0, 6.0, 0.0, 4.0, 2.0];
        (y, x)
    }

    #[test]
    fn truncated_score_zero_at_optimum() {
        let (y, x) = count_design();
        let m = TruncatedLFPoisson::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let s = m.score(&res.params);
        assert!(s.dot(&s).sqrt() < 1e-9, "score {}", s.dot(&s).sqrt());
    }

    #[test]
    fn truncated_score_matches_fd() {
        let (y, x) = count_design();
        let m = TruncatedLFPoisson::new(y, x).unwrap();
        let b = array![0.3, 0.2];
        let s = m.score(&b);
        let fd = approx_fprime(&b, |q| m.loglike(q));
        for i in 0..b.len() {
            assert_abs_diff_eq!(s[i], fd[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn truncated_hessian_matches_fd() {
        let (y, x) = count_design();
        let m = TruncatedLFPoisson::new(y, x).unwrap();
        let b = array![0.3, 0.2];
        let h = m.hessian(&b);
        let eps = 1e-6;
        for j in 0..b.len() {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[j] += eps;
            bm[j] -= eps;
            let gp = m.score(&bp);
            let gm = m.score(&bm);
            for i in 0..b.len() {
                let fd = (gp[i] - gm[i]) / (2.0 * eps);
                assert_abs_diff_eq!(h[[i, j]], fd, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn truncated_drops_zeros() {
        let (y, x) = count_design();
        let nz = y.iter().filter(|&&v| v > 0.0).count();
        let m = TruncatedLFPoisson::new(y, x).unwrap();
        assert_eq!(m.nobs(), nz);
    }

    #[test]
    fn zero_hurdle_score_matches_fd() {
        let (y, x) = count_design();
        let m = ZeroHurdlePoisson::new(&y, &x);
        let b = array![0.1, 0.3];
        let s = m.score(&b);
        let fd = approx_fprime(&b, |q| m.loglike(q));
        for i in 0..b.len() {
            assert_abs_diff_eq!(s[i], fd[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn zero_hurdle_hessian_matches_fd() {
        let (y, x) = count_design();
        let m = ZeroHurdlePoisson::new(&y, &x);
        let b = array![0.1, 0.3];
        let h = m.hessian(&b);
        let eps = 1e-6;
        for j in 0..b.len() {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[j] += eps;
            bm[j] -= eps;
            let gp = m.score(&bp);
            let gm = m.score(&bm);
            for i in 0..b.len() {
                let fd = (gp[i] - gm[i]) / (2.0 * eps);
                assert_abs_diff_eq!(h[[i, j]], fd, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn hurdle_loglike_separates() {
        let (y, x) = count_design();
        let res = HurdleCountModel::new(y, x).unwrap().fit().unwrap();
        assert!(res.converged);
        assert_abs_diff_eq!(res.llf, res.llf_zero + res.llf_count, epsilon = 1e-12);
        assert_eq!(res.params.len(), 4);
    }
}
