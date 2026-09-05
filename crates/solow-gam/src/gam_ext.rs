//! Penalized additive model fit supporting **non-canonical** link functions.
//!
//! When the link is the family's canonical link, the observed and expected
//! information matrices coincide and the effective degrees of freedom (edf) can
//! be read off the expected-information hat matrix (as in [`crate::GlmGam`]).
//! For a non-canonical link they differ, and the reference computes the edf
//! from the **observed** information. This module reproduces that
//! observed-information edf exactly while sharing the P-IRLS estimation logic.
//!
//! The smooth term is supplied as a precomputed basis matrix plus its penalty
//! matrix, so the same fitter serves both the B-spline ([`crate::BSplines`])
//! and cyclic cubic spline ([`crate::CyclicCubicSplines`]) bases.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_glm::{Family, Link};
use solow_linalg::{pinv, svd};

/// A penalized additive model with an explicit (possibly non-canonical) link.
///
/// The design is `[1, basis]`; the penalty `alpha * penalty` is applied to the
/// smooth coefficients only (the intercept is unpenalized). The reference
/// augments the penalty by a factor of two inside P-IRLS, so the normal
/// equations use `X'WX + 2 alpha S`.
#[derive(Clone, Debug)]
pub struct GlmGamExt {
    endog: Array1<f64>,
    /// Smooth basis columns (without the intercept), `n_obs` x `k`.
    basis: Array2<f64>,
    /// Smooth penalty matrix `S`, `k` x `k`.
    penalty: Array2<f64>,
    family: Family,
    link: Link,
    alpha: f64,
    maxiter: usize,
    tol: f64,
}

impl GlmGamExt {
    /// Build the model from a precomputed smooth basis and penalty.
    ///
    /// `basis` is the `n_obs` x `k` smooth design (no intercept column);
    /// `penalty` is the `k` x `k` wiggliness penalty `S`; `alpha` is the fixed
    /// smoothing parameter; `link` may be non-canonical for `family`.
    pub fn new(
        endog: Array1<f64>,
        basis: Array2<f64>,
        penalty: Array2<f64>,
        alpha: f64,
        family: Family,
        link: Link,
    ) -> Result<Self> {
        let (n, k) = basis.dim();
        if endog.len() != n {
            return Err(Error::Shape("endog length != basis rows".into()));
        }
        if penalty.dim() != (k, k) {
            return Err(Error::Shape("penalty must be k x k matching basis".into()));
        }
        if alpha < 0.0 {
            return Err(Error::Shape("alpha must be non-negative".into()));
        }
        Ok(GlmGamExt {
            endog,
            basis,
            penalty,
            family,
            link,
            alpha,
            maxiter: 5000,
            tol: 1e-13,
        })
    }

    /// The full design matrix `[1, basis]`.
    fn design(&self) -> Array2<f64> {
        let (n, k) = self.basis.dim();
        let mut x = Array2::<f64>::zeros((n, k + 1));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 0..k {
                x[[i, j + 1]] = self.basis[[i, j]];
            }
        }
        x
    }

    /// The block penalty `2 alpha S` embedded into the full parameter space
    /// (intercept block is zero). The factor of two matches the reference's
    /// augmented P-IRLS.
    fn penalty_2s(&self) -> Array2<f64> {
        let k = self.penalty.ncols();
        let p = k + 1;
        let mut s2 = Array2::<f64>::zeros((p, p));
        for a in 0..k {
            for b in 0..k {
                s2[[a + 1, b + 1]] = 2.0 * self.alpha * self.penalty[[a, b]];
            }
        }
        s2
    }

    /// Fit the model by penalized IRLS and return the results.
    ///
    /// Each P-IRLS step solves the augmented weighted least-squares problem
    /// `min_b || sqrt(W1) (Z1 - X1 b) ||^2` where `X1 = [X; R]`, `Z1 = [z; 0]`,
    /// `W1 = [w; 1]` and `R' R = 2 alpha S` is the matrix square root of the
    /// embedded penalty. The solution is taken with the Moore-Penrose
    /// pseudo-inverse so that a rank-deficient design (e.g. a cyclic spline
    /// basis collinear with the intercept) yields the same minimum-norm
    /// parameters as the reference.
    pub fn fit(&self) -> Result<GamExtResults> {
        let exog = self.design();
        let (n, p) = exog.dim();
        let s2 = self.penalty_2s();
        // Penalty square-root rows R (r x p) with R' R = 2 alpha S.
        let rs = matrix_sqrt(&s2)?;

        let y = &self.endog;
        let ybar = y.sum() / n as f64;

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
        let mut ncp = Array2::<f64>::eye(p);
        let mut converged = false;
        let mut n_iter = 0;

        for it in 0..self.maxiter {
            n_iter = it + 1;
            let mut w = Array1::<f64>::zeros(n);
            let mut z = Array1::<f64>::zeros(n);
            for i in 0..n {
                let gp = self.link.deriv(mu[i]);
                let var = self.family.variance(mu[i]);
                w[i] = 1.0 / (gp * gp * var);
                z[i] = eta[i] + (y[i] - mu[i]) * gp;
            }

            let (new_params, new_ncp) = penalized_wls(&exog, &rs, &w, &z)?;
            params = new_params;
            ncp = new_ncp;

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
        // `ncp` is the normalized covariance of the *augmented* WLS at the final
        // step: pinv(W1^{1/2} X1) pinv(W1^{1/2} X1)', equivalently the
        // pseudo-inverse of `X'W X + 2 alpha S`. This is exactly the reference's
        // `normalized_cov_params` (at scale 1).

        // Observed-information weights for the hat matrix (the reference uses
        // `hessian_factor(observed=True)`):
        //   eim_i   = 1 / (g'(mu)^2 V(mu))
        //   score_i = (y - mu) / (g'(mu) V(mu))
        //   tmp_i   = V(mu) g''(mu) + V'(mu) g'(mu)
        //   oim_i   = eim_i * (1 + score_i * tmp_i)
        // The edf is scale-invariant, so we compute it at scale 1.
        let mut oim = Array1::<f64>::zeros(n);
        for i in 0..n {
            let gp = self.link.deriv(mu[i]);
            let gpp = link_deriv2(self.link, mu[i]);
            let var = self.family.variance(mu[i]);
            let vp = variance_deriv(self.family, mu[i]);
            let eim = 1.0 / (gp * gp * var);
            let score = (y[i] - mu[i]) / (gp * var);
            let tmp = var * gpp + vp * gp;
            oim[i] = eim * (1.0 + score * tmp);
        }

        // wexog = sqrt(oim) * X (the observed-information weighted design).
        let mut wexog = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let s = oim[i].sqrt();
            for j in 0..p {
                wexog[[i, j]] = exog[[i, j]] * s;
            }
        }
        // edf_j = sum_i wexog[i, j] * (ncp wexog[i, :]')_j  (axis = 0 of the
        // reference's hat-matrix-diag), i.e. project onto each column.
        let tmp_mat = ncp.dot(&wexog.t()); // (p, n)
        let mut edf = Array1::<f64>::zeros(p);
        for j in 0..p {
            let mut acc = 0.0;
            for i in 0..n {
                acc += wexog[[i, j]] * tmp_mat[[j, i]];
            }
            edf[j] = acc;
        }
        let edf_total: f64 = edf.sum();

        // Scale: 1 for fixed-scale families; otherwise Pearson chi-square over
        // df_resid = nobs - edf_total (matching the reference's final scale).
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

        let penalty_quad = params.dot(&s2.dot(&params));
        let penalized_deviance = dev + penalty_quad;
        let fittedvalues = mu.clone();

        Ok(GamExtResults {
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
            dim_basis: self.basis.ncols(),
        })
    }
}

/// Second derivative of the link `g''(mu)`, matching the reference link
/// classes. Only the links exercised here are implemented analytically; others
/// fall back to a central finite difference of `Link::deriv`.
fn link_deriv2(link: Link, mu: f64) -> f64 {
    use solow_distributions::{norm_pdf, norm_ppf};
    match link {
        Link::Identity => 0.0,
        Link::Log => -1.0 / (mu * mu),
        Link::Logit => {
            let v = mu * (1.0 - mu);
            (2.0 * mu - 1.0) / (v * v)
        }
        Link::Probit => {
            // g''(p) = linpred / pdf(linpred)^2 with linpred = Phi^-1(p).
            let lp = norm_ppf(mu.clamp(1e-12, 1.0 - 1e-12));
            let pdf = norm_pdf(lp);
            lp / (pdf * pdf)
        }
        Link::Sqrt => -0.25 / mu.powf(1.5),
        Link::InversePower => 2.0 / (mu * mu * mu),
        Link::InverseSquared => 6.0 / (mu * mu * mu * mu),
        Link::CLogLog => {
            // g(mu) = ln(-ln(1-mu)). Central difference fallback (not exercised
            // by the verified cases) keeps behaviour finite.
            let h = 1e-6;
            (link.deriv(mu + h) - link.deriv(mu - h)) / (2.0 * h)
        }
    }
}

/// Derivative of the variance function `V'(mu)`, matching the reference
/// variance-function classes.
fn variance_deriv(family: Family, mu: f64) -> f64 {
    match family {
        Family::Gaussian => 0.0,
        Family::Poisson => 1.0,
        Family::Binomial => 1.0 - 2.0 * mu,
        Family::Gamma => 2.0 * mu,
        Family::InverseGaussian => 3.0 * mu * mu,
        Family::NegativeBinomial { alpha } => 1.0 + 2.0 * alpha * mu,
    }
}

/// Symmetric matrix square root `R` such that `R' R = mat` for a symmetric PSD
/// `mat`, dropping singular directions below `threshold` (reference
/// `tools.linalg.matrix_sqrt`, `full=False`). `R` has `rank` rows and
/// `mat.ncols()` columns.
fn matrix_sqrt(mat: &Array2<f64>) -> Result<Array2<f64>> {
    const THRESHOLD: f64 = 1e-15;
    let (_u, s, vt) = svd(mat)?;
    // Keep directions with singular value above the threshold.
    let p = mat.ncols();
    let mut rows: Vec<f64> = Vec::new();
    let mut r = 0usize;
    for (i, &si) in s.iter().enumerate() {
        if si > THRESHOLD {
            let sq = si.sqrt();
            for j in 0..p {
                rows.push(sq * vt[[i, j]]);
            }
            r += 1;
        }
    }
    Array2::from_shape_vec((r, p), rows).map_err(|_| Error::Shape("matrix_sqrt: shape".into()))
}

/// Penalized weighted least squares by augmented design and pseudo-inverse,
/// matching the reference `penalized_wls`.
///
/// Solves `min_b || sqrt(W1) (Z1 - X1 b) ||^2` with the augmented system
/// `X1 = [X; R]`, `Z1 = [z; 0]`, `W1 = [w; 1]`, returning the minimum-norm
/// parameters and `normalized_cov_params = pinv(W1^{1/2} X1) pinv(...)'`.
fn penalized_wls(
    exog: &Array2<f64>,
    rs: &Array2<f64>,
    w: &Array1<f64>,
    z: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let (n, p) = exog.dim();
    let r = rs.nrows();
    let m = n + r;

    // Augmented weighted design wexog1 = sqrt(W1) * X1 and response wz1.
    let mut wexog1 = Array2::<f64>::zeros((m, p));
    let mut wz1 = Array1::<f64>::zeros(m);
    for i in 0..n {
        let s = w[i].sqrt();
        wz1[i] = z[i] * s;
        for j in 0..p {
            wexog1[[i, j]] = exog[[i, j]] * s;
        }
    }
    // Penalty rows carry weight 1 and augmented response 0.
    for i in 0..r {
        for j in 0..p {
            wexog1[[n + i, j]] = rs[[i, j]];
        }
    }

    let (pinv_w, _sv) = pinv(&wexog1)?; // p x m
    let params = pinv_w.dot(&wz1);
    let ncp = pinv_w.dot(&pinv_w.t());
    Ok((params, ncp))
}

/// The fitted result of a [`GlmGamExt`].
#[derive(Clone, Debug)]
pub struct GamExtResults {
    /// Estimated parameters: `[intercept, smooth coefficients...]`.
    pub params: Array1<f64>,
    /// Fitted mean response `mu`.
    pub fittedvalues: Array1<f64>,
    /// Effective degrees of freedom per design column (observed information).
    pub edf: Array1<f64>,
    /// Total effective degrees of freedom.
    pub edf_total: f64,
    /// Dispersion/scale estimate (1 for fixed-scale families).
    pub scale: f64,
    /// (Unpenalized) model deviance.
    pub deviance: f64,
    /// Penalized deviance `deviance + params' (2 alpha S) params`.
    pub penalized_deviance: f64,
    /// Residual degrees of freedom `nobs - edf_total`.
    pub df_resid: f64,
    /// Whether P-IRLS converged.
    pub converged: bool,
    /// Number of P-IRLS iterations.
    pub n_iter: usize,
    /// Number of smooth basis columns.
    pub dim_basis: usize,
}

impl GamExtResults {
    /// The smooth coefficients (excluding the intercept).
    pub fn smooth_params(&self) -> Array1<f64> {
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
    use crate::BSplines;
    use ndarray::Array1;

    fn make_basis(x: &Array1<f64>, df: usize, degree: usize) -> (Array2<f64>, Array2<f64>) {
        let bs = BSplines::new(x, df, degree).unwrap();
        (bs.basis().clone(), bs.cov_der2().clone())
    }

    #[test]
    fn canonical_log_matches_expected_edf_for_poisson() {
        // For a canonical link (Poisson + log) the observed-information edf
        // must equal the expected-information edf computed by GlmGam.
        let n = 60;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|xi| {
            (1.0 + (2.0 * std::f64::consts::PI * xi).sin())
                .exp()
                .round()
        });
        let (basis, pen) = make_basis(&x, 8, 3);

        let ext = GlmGamExt::new(y.clone(), basis, pen, 1.0, Family::Poisson, Link::Log)
            .unwrap()
            .fit()
            .unwrap();
        let canon = crate::GlmGam::new(y, &x, 8, 3, 1.0, Family::Poisson)
            .unwrap()
            .fit()
            .unwrap();
        assert!(ext.converged && canon.converged);
        assert!(
            (ext.edf_total - canon.edf_total).abs() < 1e-9,
            "canonical edf mismatch: ext={} canon={}",
            ext.edf_total,
            canon.edf_total
        );
    }

    #[test]
    fn gaussian_log_converges_positive() {
        let n = 70;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|xi| (0.5 + 0.7 * (2.0 * std::f64::consts::PI * xi).sin()).exp());
        let (basis, pen) = make_basis(&x, 8, 3);
        let res = GlmGamExt::new(y, basis, pen, 0.5, Family::Gaussian, Link::Log)
            .unwrap()
            .fit()
            .unwrap();
        assert!(res.converged);
        assert!(res.fittedvalues.iter().all(|&m| m > 0.0));
        assert!(res.edf_total > 0.0 && res.edf_total < 9.0);
    }
}
