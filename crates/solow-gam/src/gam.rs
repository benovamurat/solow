//! Generalized additive model with a penalized B-spline smooth term, fit by
//! penalized iteratively reweighted least squares (P-IRLS) at a fixed
//! smoothing parameter `alpha`.

use crate::bspline::BSplines;
use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_glm::{Family, Link};
use solow_linalg::{inv, solve};

/// A penalized additive model: an intercept plus one B-spline smooth term,
/// estimated by P-IRLS at a fixed smoothing parameter.
///
/// The design is `[1, B(x)]` where `B(x)` is the [`BSplines`] basis (constant
/// column dropped). The roughness penalty `alpha * S` is applied only to the
/// spline coefficients, where `S` is the curvature penalty of the basis. This
/// matches the reference `GLMGam` with a single smooth component, an explicit
/// constant linear term, and the given explicit `alpha`.
#[derive(Clone, Debug)]
pub struct GlmGam {
    endog: Array1<f64>,
    smoother: BSplines,
    family: Family,
    link: Link,
    alpha: f64,
    maxiter: usize,
    tol: f64,
}

impl GlmGam {
    /// Build a GAM for `endog` with a smooth term over `x`.
    ///
    /// `df` and `degree` configure the B-spline basis; `alpha` is the fixed
    /// smoothing parameter; the family's canonical link is used.
    pub fn new(
        endog: Array1<f64>,
        x: &Array1<f64>,
        df: usize,
        degree: usize,
        alpha: f64,
        family: Family,
    ) -> Result<Self> {
        let link = family.default_link();
        Self::with_link(endog, x, df, degree, alpha, family, link)
    }

    /// Build a GAM with an explicit link function.
    pub fn with_link(
        endog: Array1<f64>,
        x: &Array1<f64>,
        df: usize,
        degree: usize,
        alpha: f64,
        family: Family,
        link: Link,
    ) -> Result<Self> {
        if endog.len() != x.len() {
            return Err(Error::Shape("endog length != x length".into()));
        }
        if alpha < 0.0 {
            return Err(Error::Shape("alpha must be non-negative".into()));
        }
        let smoother = BSplines::new(x, df, degree)?;
        Ok(GlmGam {
            endog,
            smoother,
            family,
            link,
            alpha,
            maxiter: 5000,
            // Absolute tolerance on the change in deviance between iterations.
            tol: 1e-13,
        })
    }

    /// The full design matrix `[1, B(x)]` (intercept plus spline columns).
    fn design(&self) -> Array2<f64> {
        let basis = self.smoother.basis();
        let (n, k) = basis.dim();
        let mut x = Array2::<f64>::zeros((n, k + 1));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 0..k {
                x[[i, j + 1]] = basis[[i, j]];
            }
        }
        x
    }

    /// The block penalty `2 * S` embedded into the full parameter space.
    ///
    /// The reference augments the penalty by a factor of two inside P-IRLS, so
    /// the normal equations use `X'WX + 2 alpha S`. The intercept column is
    /// unpenalized (its block is zero).
    fn penalty_2s(&self) -> Array2<f64> {
        let cov = self.smoother.cov_der2();
        let k = cov.ncols();
        let p = k + 1;
        let mut s2 = Array2::<f64>::zeros((p, p));
        for a in 0..k {
            for b in 0..k {
                s2[[a + 1, b + 1]] = 2.0 * self.alpha * cov[[a, b]];
            }
        }
        s2
    }

    /// Fit the model by penalized IRLS and return the results.
    pub fn fit(&self) -> Result<GamResults> {
        let exog = self.design();
        let (n, p) = exog.dim();
        let s2 = self.penalty_2s();

        let y = &self.endog;
        let ybar = y.sum() / n as f64;

        // Initialise from the family's starting mean.
        let mut mu = y.mapv(|yi| self.family.starting_mu(yi, ybar));
        let mut eta = mu.mapv(|m| self.link.link(m));
        // `endog`/`mu` are owned, standard-layout arrays; surface a clean error
        // rather than panicking in the impossible non-contiguous case.
        let y_s = y
            .as_slice()
            .ok_or_else(|| Error::Value("endog must be contiguous".into()))?;
        let mut dev = self.family.deviance(
            y_s,
            mu.as_slice()
                .ok_or_else(|| Error::Value("mu must be contiguous".into()))?,
        );

        let mut params = Array1::<f64>::zeros(p);
        let mut converged = false;
        let mut n_iter = 0;

        for it in 0..self.maxiter {
            n_iter = it + 1;

            // Working weights w = 1 / (g'(mu)^2 V(mu)) and adjusted response
            // z = eta + g'(mu) (y - mu).
            let mut w = Array1::<f64>::zeros(n);
            let mut z = Array1::<f64>::zeros(n);
            for i in 0..n {
                let gp = self.link.deriv(mu[i]);
                let var = self.family.variance(mu[i]);
                w[i] = 1.0 / (gp * gp * var);
                z[i] = eta[i] + (y[i] - mu[i]) * gp;
            }

            // Penalized weighted least squares: (X'WX + 2 alpha S) b = X'W z.
            let (xtwx, xtwz) = weighted_normal_eqs(&exog, &w, &z);
            let mut lhs = xtwx;
            lhs += &s2;
            params = solve(&lhs, &xtwz)?;

            eta = exog.dot(&params);
            mu = eta.mapv(|e| self.link.inverse(e));

            let dev_new = self.family.deviance(
                y_s,
                mu.as_slice()
                    .ok_or_else(|| Error::Value("mu must be contiguous".into()))?,
            );
            if (dev_new - dev).abs() <= self.tol {
                dev = dev_new;
                converged = true;
                break;
            }
            dev = dev_new;
        }

        // Effective degrees of freedom per column: the diagonal of the hat
        // matrix decomposed onto exog columns. With wexog = sqrt(w) X and
        // ncp = (X'WX + 2 alpha S)^-1, the scale cancels for canonical links:
        //     edf_j = sum_i wexog[i] (ncp wexog[i]')  projected to column j.
        let mut w_final = Array1::<f64>::zeros(n);
        for i in 0..n {
            let gp = self.link.deriv(mu[i]);
            let var = self.family.variance(mu[i]);
            w_final[i] = 1.0 / (gp * gp * var);
        }
        let (xtwx_final, _) = weighted_normal_eqs(&exog, &w_final, y);
        let mut a = xtwx_final;
        a += &s2;
        let ncp = inv(&a)?;

        // wexog = sqrt(w) * X.
        let mut wexog = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let s = w_final[i].sqrt();
            for j in 0..p {
                wexog[[i, j]] = exog[[i, j]] * s;
            }
        }
        // edf_j = sum_i wexog[i, :] . (ncp . wexog[i, :]).
        let tmp = ncp.dot(&wexog.t()); // (p, n)
        let mut edf = Array1::<f64>::zeros(p);
        for j in 0..p {
            let mut acc = 0.0;
            for i in 0..n {
                acc += wexog[[i, j]] * tmp[[j, i]];
            }
            edf[j] = acc;
        }
        let edf_total: f64 = edf.sum();

        // Scale: 1 for Poisson/Binomial families, Pearson chi-square / df_resid
        // otherwise, where df_resid = nobs - edf_total.
        let df_resid = n as f64 - edf_total;
        let scale = if self.family.fixed_scale() {
            1.0
        } else {
            let mut chi2 = 0.0;
            for i in 0..n {
                let r = y[i] - mu[i];
                chi2 += r * r / self.family.variance(mu[i]);
            }
            chi2 / df_resid
        };

        // Penalized deviance: deviance + params' (2 alpha S) params (the
        // objective minimised by the augmented P-IRLS).
        let penalty_quad = params.dot(&s2.dot(&params));
        let penalized_deviance = dev + penalty_quad;

        let fittedvalues = mu.clone();

        Ok(GamResults {
            params,
            fittedvalues,
            edf,
            edf_total,
            scale,
            deviance: dev,
            penalized_deviance,
            df_resid,
            converged,
            n_iter,
            dim_basis: self.smoother.dim_basis(),
        })
    }

    /// The B-spline smoother used by this model.
    pub fn smoother(&self) -> &BSplines {
        &self.smoother
    }
}

/// Build `(X'WX, X'Wz)` for diagonal weights `w`.
fn weighted_normal_eqs(
    exog: &Array2<f64>,
    w: &Array1<f64>,
    z: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let (n, p) = exog.dim();
    let mut wexog = Array2::<f64>::zeros((n, p));
    let mut wz = Array1::<f64>::zeros(n);
    for i in 0..n {
        wz[i] = w[i] * z[i];
        for j in 0..p {
            wexog[[i, j]] = w[i] * exog[[i, j]];
        }
    }
    let xtwx = exog.t().dot(&wexog);
    let xtwz = exog.t().dot(&wz);
    (xtwx, xtwz)
}

/// The fitted result of a [`GlmGam`].
#[derive(Clone, Debug)]
pub struct GamResults {
    /// Estimated parameters: `[intercept, spline coefficients...]`.
    pub params: Array1<f64>,
    /// Fitted mean response `mu` at each observation.
    pub fittedvalues: Array1<f64>,
    /// Effective degrees of freedom for each design column.
    pub edf: Array1<f64>,
    /// Total effective degrees of freedom (trace of the hat matrix).
    pub edf_total: f64,
    /// Dispersion/scale estimate (1 for Poisson/Binomial).
    pub scale: f64,
    /// (Unpenalized) model deviance.
    pub deviance: f64,
    /// Penalized deviance `deviance + params' (2 alpha S) params`.
    pub penalized_deviance: f64,
    /// Residual degrees of freedom `nobs - edf_total`.
    pub df_resid: f64,
    /// Whether the P-IRLS iteration converged.
    pub converged: bool,
    /// Number of P-IRLS iterations performed.
    pub n_iter: usize,
    /// Number of spline basis columns.
    pub dim_basis: usize,
}

impl GamResults {
    /// The spline coefficients (excluding the intercept).
    pub fn spline_params(&self) -> Array1<f64> {
        self.params.slice(s![1..]).to_owned()
    }

    /// The estimated intercept.
    pub fn intercept(&self) -> f64 {
        self.params[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn gaussian_unpenalized_is_close_to_basis_fit() {
        // With alpha = 0 the GAM is an ordinary least squares fit onto the
        // spline basis plus intercept; residuals should be small for a smooth
        // signal.
        let n = 60;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|xi| (2.0 * std::f64::consts::PI * xi).sin());
        let res = GlmGam::new(y.clone(), &x, 10, 3, 0.0, Family::Gaussian)
            .unwrap()
            .fit()
            .unwrap();
        assert!(res.converged);
        // The unpenalized spline fits a smooth sine very closely.
        let resid: f64 = (&res.fittedvalues - &y).mapv(|e| e * e).sum();
        assert!(resid < 1e-2, "residual SS too large: {resid}");
        // edf should be near the number of columns for alpha = 0.
        assert!(res.edf_total > 9.0 && res.edf_total <= 10.0 + 1e-6);
    }

    #[test]
    fn larger_alpha_reduces_edf() {
        let n = 80;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|xi| (2.0 * std::f64::consts::PI * xi).sin());
        let lo = GlmGam::new(y.clone(), &x, 10, 3, 0.1, Family::Gaussian)
            .unwrap()
            .fit()
            .unwrap();
        let hi = GlmGam::new(y.clone(), &x, 10, 3, 100.0, Family::Gaussian)
            .unwrap()
            .fit()
            .unwrap();
        assert!(
            hi.edf_total < lo.edf_total,
            "more smoothing should lower edf"
        );
    }

    #[test]
    fn poisson_fit_converges_positive_mean() {
        let n = 50;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|xi| {
            (1.0 + (2.0 * std::f64::consts::PI * xi).sin())
                .exp()
                .round()
        });
        let res = GlmGam::new(y, &x, 8, 3, 1.0, Family::Poisson)
            .unwrap()
            .fit()
            .unwrap();
        assert!(res.converged);
        assert!(res.fittedvalues.iter().all(|&m| m > 0.0));
        assert_eq!(res.scale, 1.0);
    }
}
