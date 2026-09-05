//! Random-intercept linear mixed-effects model fit by (RE)ML.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::{norm_cdf, norm_ppf};
use solow_linalg::{det, inv, solve};
use solow_optimize::{approx_hess, minimize_bfgs};

/// Estimation criterion for [`MixedLm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemlMethod {
    /// Restricted maximum likelihood (the default; corrects for the degrees of
    /// freedom consumed by the fixed effects).
    Reml,
    /// Ordinary maximum likelihood.
    Ml,
}

/// A random-intercept mixed model awaiting estimation.
///
/// The design is `y = X·β + b_g + ε`, where `b_g ~ N(0, ψ·σ²)` is a single
/// random intercept shared by all observations in group `g`, and
/// `ε ~ N(0, σ²)`.
#[derive(Clone, Debug)]
pub struct MixedLm {
    endog: Array1<f64>,
    exog: Array2<f64>,
    /// Row indices for each distinct group, in first-appearance order.
    groups: Vec<Vec<usize>>,
    method: RemlMethod,
}

impl MixedLm {
    /// Build a random-intercept model from response `endog`, fixed-effects
    /// design `exog` (rows are observations; include your own intercept column),
    /// and an integer group label per observation.
    ///
    /// Estimation defaults to REML. Use [`MixedLm::method`] to switch to ML.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>, group_labels: &[i64]) -> Result<Self> {
        let n = endog.len();
        if exog.nrows() != n {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if group_labels.len() != n {
            return Err(Error::Shape("group_labels length != endog length".into()));
        }
        // Group rows, preserving first-appearance order of labels.
        let mut order: Vec<i64> = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, &g) in group_labels.iter().enumerate() {
            match order.iter().position(|&x| x == g) {
                Some(k) => groups[k].push(i),
                None => {
                    order.push(g);
                    groups.push(vec![i]);
                }
            }
        }
        Ok(MixedLm {
            endog,
            exog,
            groups,
            method: RemlMethod::Reml,
        })
    }

    /// Select the estimation criterion (REML or ML). Returns `self` for chaining.
    pub fn method(mut self, method: RemlMethod) -> Self {
        self.method = method;
        self
    }

    fn n_obs(&self) -> usize {
        self.endog.len()
    }
    fn k_fe(&self) -> usize {
        self.exog.ncols()
    }
    fn reml(&self) -> bool {
        self.method == RemlMethod::Reml
    }

    /// Effective sample-size factor: `N` for ML, `N − p` for REML.
    fn fac(&self) -> f64 {
        let n = self.n_obs() as f64;
        if self.reml() {
            n - self.k_fe() as f64
        } else {
            n
        }
    }

    /// Accumulate the GLS normal-equation pieces at covariance ratio `psi`.
    ///
    /// Returns `(xtvix, xtviy)` where `xtvix = Σ Xᵍᵀ Vᵍ⁻¹ Xᵍ` and
    /// `xtviy = Σ Xᵍᵀ Vᵍ⁻¹ yᵍ`, with `Vᵍ⁻¹ = I − ψ/(1+nᵍψ)·11ᵀ`.
    fn gls_pieces(&self, psi: f64) -> (Array2<f64>, Array1<f64>) {
        let p = self.k_fe();
        let mut xtvix = Array2::<f64>::zeros((p, p));
        let mut xtviy = Array1::<f64>::zeros(p);
        for rows in &self.groups {
            let ng = rows.len() as f64;
            let c = psi / (1.0 + ng * psi);
            // Column sums of X over the group, and Σ y.
            let mut xs = Array1::<f64>::zeros(p);
            let mut ys = 0.0;
            for &i in rows {
                ys += self.endog[i];
                for j in 0..p {
                    xs[j] += self.exog[[i, j]];
                }
            }
            // Xᵀ Vinv X = Xᵀ X − c (Xᵀ1)(Xᵀ1)ᵀ ; Xᵀ Vinv y likewise.
            for &i in rows {
                let yi = self.endog[i];
                for a in 0..p {
                    let xia = self.exog[[i, a]];
                    xtviy[a] += xia * yi;
                    for b in 0..p {
                        xtvix[[a, b]] += xia * self.exog[[i, b]];
                    }
                }
            }
            for a in 0..p {
                xtviy[a] -= c * xs[a] * ys;
                for b in 0..p {
                    xtvix[[a, b]] -= c * xs[a] * xs[b];
                }
            }
        }
        (xtvix, xtviy)
    }

    /// GLS fixed-effects estimate at `psi`.
    fn fe_params(&self, psi: f64) -> Result<Array1<f64>> {
        let (xtvix, xtviy) = self.gls_pieces(psi);
        solve(&xtvix, &xtviy)
    }

    /// Quadratic form `Σ rᵍᵀ Vᵍ⁻¹ rᵍ` for residuals `r = y − Xβ` at `psi`.
    fn quad_form(&self, psi: f64, beta: &Array1<f64>) -> f64 {
        let p = self.k_fe();
        let mut qf = 0.0;
        for rows in &self.groups {
            let ng = rows.len() as f64;
            let c = psi / (1.0 + ng * psi);
            let mut rsum = 0.0;
            let mut rss = 0.0;
            for &i in rows {
                let mut fitted = 0.0;
                for j in 0..p {
                    fitted += self.exog[[i, j]] * beta[j];
                }
                let r = self.endog[i] - fitted;
                rsum += r;
                rss += r * r;
            }
            qf += rss - c * rsum * rsum;
        }
        qf
    }

    /// Profile (RE)ML log-likelihood at covariance ratio `psi` (with `β`
    /// profiled by GLS and `σ²` profiled in closed form).
    fn profile_loglike(&self, psi: f64) -> Result<f64> {
        let p = self.k_fe();
        let beta = self.fe_params(psi)?;
        let qf = self.quad_form(psi, &beta);

        // Σ logdet(Vᵍ) = Σ log(1 + nᵍ ψ).
        let mut logdet_v = 0.0;
        for rows in &self.groups {
            logdet_v += (1.0 + rows.len() as f64 * psi).ln();
        }

        let fac = self.fac();
        let two_pi = std::f64::consts::TAU;
        let mut like = -logdet_v / 2.0;
        like -= fac * qf.ln() / 2.0;
        like -= fac * two_pi.ln() / 2.0;
        like += fac * fac.ln() / 2.0;
        like -= fac / 2.0;

        if self.reml() {
            let (xtvix, _) = self.gls_pieces(psi);
            let d = det(&xtvix)?;
            like -= d.abs().ln() / 2.0;
        }
        let _ = p;
        Ok(like)
    }

    /// Same accumulation as [`Self::profile_loglike`] but as an infallible
    /// function of a packed `[β, ψ]` parameter (with `σ²` profiled out), for
    /// numerical Hessian evaluation. `β` is *not* re-profiled here.
    fn joint_loglike(&self, packed: &Array1<f64>) -> f64 {
        let p = self.k_fe();
        let beta = packed.slice(ndarray::s![..p]).to_owned();
        let psi = packed[p];

        let qf = self.quad_form(psi, &beta);
        let mut logdet_v = 0.0;
        for rows in &self.groups {
            logdet_v += (1.0 + rows.len() as f64 * psi).ln();
        }
        let fac = self.fac();
        let two_pi = std::f64::consts::TAU;
        let mut like = -logdet_v / 2.0;
        like -= fac * qf.ln() / 2.0;
        like -= fac * two_pi.ln() / 2.0;
        like += fac * fac.ln() / 2.0;
        like -= fac / 2.0;
        if self.reml() {
            let (xtvix, _) = self.gls_pieces(psi);
            if let Ok(d) = det(&xtvix) {
                like -= d.abs().ln() / 2.0;
            }
        }
        like
    }

    /// Fit the model, maximizing the profile (RE)ML log-likelihood over `ψ`.
    pub fn fit(&self) -> Result<MixedLmResults> {
        // Optimize over θ = ln(ψ) (keeps ψ > 0 without constraints).
        // The objective is the negative profile log-likelihood.
        let neg = |theta: &Array1<f64>| -> f64 {
            let psi = theta[0].exp();
            match self.profile_loglike(psi) {
                Ok(v) => -v,
                Err(_) => f64::INFINITY,
            }
        };
        let grad = |theta: &Array1<f64>| -> Array1<f64> {
            // Central difference on the 1-D objective.
            let base = 1e-7;
            let h = base * (1.0 + theta[0].abs());
            let mut tp = theta.clone();
            tp[0] = theta[0] + h;
            let fp = neg(&tp);
            tp[0] = theta[0] - h;
            let fm = neg(&tp);
            Array1::from_vec(vec![(fp - fm) / (2.0 * h)])
        };

        // A reasonable starting value: ψ ≈ 1 (θ = 0).
        let start = Array1::from_vec(vec![0.0_f64]);
        let res = minimize_bfgs(&start, neg, grad, 500, 1e-11)?;
        // Polish with a few Newton steps on the 1-D profile to tighten further.
        let theta = newton_polish(&res.x[0], |t| {
            let psi = t.exp();
            self.profile_loglike(psi).map(|v| -v)
        });
        let psi = theta.exp();

        let beta = self.fe_params(psi)?;
        let qf = self.quad_form(psi, &beta);
        let scale = qf / self.fac();
        let cov_re = psi * scale; // reported in the response's variance units
        let llf = self.profile_loglike(psi)?;

        // Fixed-effects standard errors from the negative Hessian of the joint
        // (non-profiled) log-likelihood w.r.t. [β, ψ].
        let p = self.k_fe();
        let mut packed = Array1::<f64>::zeros(p + 1);
        for j in 0..p {
            packed[j] = beta[j];
        }
        packed[p] = psi;
        let hess = approx_hess(&packed, |x| self.joint_loglike(x));
        let neg_hess = hess.mapv(|v| -v);
        let pcov = inv(&neg_hess)?;
        let mut bse_fe = Array1::<f64>::zeros(p);
        for j in 0..p {
            bse_fe[j] = pcov[[j, j]].max(0.0).sqrt();
        }

        Ok(MixedLmResults {
            fe_params: beta,
            cov_re,
            scale,
            llf,
            bse_fe,
            psi,
            reml: self.reml(),
        })
    }
}

/// A few damped Newton iterations on a smooth 1-D objective `f(θ)` (to be
/// minimized), starting from `theta0`. Returns the refined `θ`.
fn newton_polish<F>(theta0: &f64, f: F) -> f64
where
    F: Fn(f64) -> Result<f64>,
{
    let mut t = *theta0;
    for _ in 0..50 {
        let h = 1e-5 * (1.0 + t.abs());
        let (fm, f0, fp) = match (f(t - h), f(t), f(t + h)) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => break,
        };
        let g = (fp - fm) / (2.0 * h);
        let hh = (fp - 2.0 * f0 + fm) / (h * h);
        if !hh.is_finite() || hh.abs() < 1e-14 {
            break;
        }
        let step = g / hh;
        if !step.is_finite() {
            break;
        }
        t -= step;
        if step.abs() < 1e-13 {
            break;
        }
    }
    t
}

/// Estimated quantities from a fitted [`MixedLm`].
#[derive(Clone, Debug)]
pub struct MixedLmResults {
    /// Fixed-effects coefficients `β` (GLS estimates).
    pub fe_params: Array1<f64>,
    /// Random-effects variance `Ψ` (the random-intercept variance, in the
    /// response's variance units, i.e. `ψ·σ²`).
    pub cov_re: f64,
    /// Residual variance `σ²`.
    pub scale: f64,
    /// Maximized profile (RE)ML log-likelihood.
    pub llf: f64,
    /// Standard errors of the fixed-effects coefficients.
    pub bse_fe: Array1<f64>,
    /// Covariance ratio `ψ = Ψ/σ²` (the unscaled random-intercept variance).
    pub psi: f64,
    /// Whether REML (vs ML) was used.
    pub reml: bool,
}

impl MixedLmResults {
    /// Wald `z`-statistics for the fixed effects (`β / se`).
    pub fn tvalues(&self) -> Array1<f64> {
        let mut t = Array1::<f64>::zeros(self.fe_params.len());
        for j in 0..t.len() {
            t[j] = self.fe_params[j] / self.bse_fe[j];
        }
        t
    }

    /// Two-sided normal-approximation p-values for the fixed effects.
    pub fn pvalues(&self) -> Array1<f64> {
        self.tvalues().mapv(|z| {
            // 2 * (1 - Phi(|z|)) via the symmetric tail.
            2.0 * (1.0 - norm_cdf(z.abs()))
        })
    }

    /// Two-sided Wald confidence intervals for the fixed effects at the given
    /// significance level `alpha` (rows are `[lower, upper]`).
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        let p = self.fe_params.len();
        let q = norm_ppf(1.0 - alpha / 2.0);
        let mut ci = Array2::<f64>::zeros((p, 2));
        for j in 0..p {
            ci[[j, 0]] = self.fe_params[j] - q * self.bse_fe[j];
            ci[[j, 1]] = self.fe_params[j] + q * self.bse_fe[j];
        }
        ci
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// A balanced random-intercept design with a strong group effect: the
    /// profile log-likelihood must be maximized at the recovered ψ, and the
    /// closed-form scale identity `scale = qf/fac` must hold.
    #[test]
    fn profile_is_maximized_at_solution() {
        // Small synthetic balanced data.
        let groups: Vec<i64> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
        let n = groups.len();
        let y = Array1::from_vec(vec![
            2.0, 2.4, 1.8, 5.1, 4.8, 5.4, -1.0, -0.6, -1.3, 3.2, 3.6, 2.9,
        ]);
        let mut x = Array2::<f64>::ones((n, 2));
        let x1 = [
            0.1, 0.5, -0.2, 0.3, -0.4, 0.6, 0.0, 0.2, -0.1, 0.4, -0.3, 0.5,
        ];
        for i in 0..n {
            x[[i, 1]] = x1[i];
        }
        let m = MixedLm::new(y, x, &groups).unwrap();
        let r = m.fit().unwrap();

        // ψ recovered, scale identity holds.
        let qf = m.quad_form(r.psi, &r.fe_params);
        assert_abs_diff_eq!(r.scale, qf / m.fac(), epsilon = 1e-10);
        assert_abs_diff_eq!(r.cov_re, r.psi * r.scale, epsilon = 1e-12);

        // The profile objective is stationary at ψ (derivative ≈ 0).
        let f = |t: f64| -m.profile_loglike(t.exp()).unwrap();
        let theta = r.psi.ln();
        let h = 1e-6;
        let g = (f(theta + h) - f(theta - h)) / (2.0 * h);
        assert!(g.abs() < 1e-5, "profile gradient {g} not ~0");
    }

    #[test]
    fn ml_scale_uses_full_sample_size() {
        // For ML the scale divides by N, not N - p.
        let groups: Vec<i64> = vec![0, 0, 1, 1, 2, 2];
        let y = Array1::from_vec(vec![1.0, 1.5, 3.0, 2.6, -0.5, 0.1]);
        let x = Array2::<f64>::ones((6, 1));
        let m = MixedLm::new(y, x, &groups).unwrap().method(RemlMethod::Ml);
        let r = m.fit().unwrap();
        let qf = m.quad_form(r.psi, &r.fe_params);
        assert_abs_diff_eq!(r.scale, qf / 6.0, epsilon = 1e-10);
        assert!(!r.reml);
    }
}
