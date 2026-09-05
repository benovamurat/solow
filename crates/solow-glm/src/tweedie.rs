//! The Tweedie family (compound Poisson-gamma) with variance power
//! `p ∈ (1, 2)`, variance function `V(μ) = μ^p`, and a log link, fit by
//! iteratively reweighted least squares (IRLS).
//!
//! This extends the GLM/IRLS machinery to a power-variance family that the
//! built-in [`Family`](crate::Family) enum does not cover. It reuses the
//! crate's [`Link::Log`](crate::Link) for the link/inverse/derivative and
//! produces the very same [`GlmResults`](crate::GlmResults) type, so callers
//! see an identical interface.
//!
//! The log-likelihood is the *exact* maximised Tweedie log-likelihood,
//! evaluated through the Dunn & Smyth (2004) series for the compound
//! Poisson-gamma density (the same series the reference package uses via
//! `wright_bessel`), not an extended-quasi-likelihood approximation.
//!
//! ```
//! use ndarray::{array, Array1};
//! use solow_glm::TweedieGlm;
//!
//! let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
//! let y: Array1<f64> = array![0.0, 1.2, 2.5, 4.1, 9.0];
//! let res = TweedieGlm::new(y, x, 1.5).unwrap().fit().unwrap();
//! assert!(res.converged);
//! ```

use crate::glm::GlmResults;
use crate::links::Link;
use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{lgamma, norm_sf};
use solow_linalg::pinv;

/// A Tweedie GLM (log link, power variance `V(μ) = μ^p`) awaiting estimation.
#[derive(Clone, Debug)]
pub struct TweedieGlm {
    endog: Array1<f64>,
    exog: Array2<f64>,
    var_power: f64,
    link: Link,
    k_constant: usize,
    maxiter: usize,
    tol: f64,
}

impl TweedieGlm {
    /// A Tweedie GLM with the canonical log link and the given variance power
    /// `p`. The compound Poisson-gamma regime requires `p ∈ (1, 2)`.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>, var_power: f64) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if !(var_power > 1.0 && var_power < 2.0) {
            return Err(Error::Value(
                "Tweedie var_power must lie strictly in (1, 2)".into(),
            ));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        let k_constant = detect_k_constant(&exog);
        Ok(TweedieGlm {
            endog,
            exog,
            var_power,
            link: Link::Log,
            k_constant,
            maxiter: 400,
            // Absolute tolerance on the change in deviance; converge to the MLE.
            tol: 1e-13,
        })
    }

    /// Variance function `V(μ) = μ^p`.
    fn variance(&self, mu: f64) -> f64 {
        mu.powf(self.var_power)
    }

    /// Per-observation starting mean for IRLS (matches the GLM default).
    fn starting_mu(&self, y: f64, ybar: f64) -> f64 {
        (y + ybar) / 2.0
    }

    /// Total deviance `Σ d(yᵢ, μᵢ)`.
    fn deviance(&self, y: &[f64], mu: &[f64]) -> f64 {
        y.iter()
            .zip(mu)
            .map(|(&yi, &mi)| tweedie_unit_deviance(yi, mi, self.var_power))
            .sum()
    }

    fn weighted_lstsq(
        &self,
        z: &Array1<f64>,
        w: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>, usize)> {
        let (n, p) = self.exog.dim();
        let mut wexog = Array2::<f64>::zeros((n, p));
        let mut wz = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = w[i].sqrt();
            wz[i] = z[i] * s;
            for j in 0..p {
                wexog[[i, j]] = self.exog[[i, j]] * s;
            }
        }
        let (pinv_w, sv) = pinv(&wexog)?;
        let params = pinv_w.dot(&wz);
        let ncp = pinv_w.dot(&pinv_w.t());
        let smax = sv.iter().cloned().fold(0.0_f64, f64::max);
        let tol = smax * (sv.len() as f64) * f64::EPSILON;
        let rank = sv.iter().filter(|&&s| s > tol).count();
        Ok((params, ncp, rank))
    }

    /// Estimate the model by IRLS, returning the standard [`GlmResults`].
    pub fn fit(&self) -> Result<GlmResults> {
        let n = self.endog.len();
        let p = self.exog.ncols();
        let ybar = self.endog.sum() / n as f64;

        let mut mu = self.endog.mapv(|y| self.starting_mu(y, ybar));
        let mut eta = mu.mapv(|m| self.link.link(m));
        // `endog`/`mu` are owned, standard-layout arrays; surface a clean error
        // rather than panicking in the impossible non-contiguous case.
        let endog_s = self
            .endog
            .as_slice()
            .ok_or_else(|| Error::Value("endog must be contiguous".into()))?;
        let mut dev = self.deviance(
            endog_s,
            mu.as_slice()
                .ok_or_else(|| Error::Value("mu must be contiguous".into()))?,
        );

        let mut params = Array1::<f64>::zeros(p);
        let mut ncp = Array2::<f64>::eye(p);
        let mut rank = p;
        let mut converged = false;
        let mut n_iter = 0;

        for it in 0..self.maxiter {
            n_iter = it + 1;
            // Working weights and adjusted response.
            let mut w = Array1::<f64>::zeros(n);
            let mut z = Array1::<f64>::zeros(n);
            for i in 0..n {
                let gp = self.link.deriv(mu[i]);
                let var = self.variance(mu[i]);
                w[i] = 1.0 / (gp * gp * var);
                z[i] = eta[i] + (self.endog[i] - mu[i]) * gp;
            }
            let (new_params, new_ncp, new_rank) = self.weighted_lstsq(&z, &w)?;
            params = new_params;
            ncp = new_ncp;
            rank = new_rank;
            eta = self.exog.dot(&params);
            mu = eta.mapv(|e| self.link.inverse(e));
            let dev_new = self.deviance(
                endog_s,
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

        Ok(self.assemble(params, ncp, rank, mu, eta, dev, converged, n_iter))
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Build the public [`GlmResults`] from the converged IRLS state.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        &self,
        params: Array1<f64>,
        normalized_cov_params: Array2<f64>,
        rank: usize,
        mu: Array1<f64>,
        eta: Array1<f64>,
        deviance: f64,
        converged: bool,
        n_iter: usize,
    ) -> GlmResults {
        let n = self.endog.len();
        let nobs = n as f64;
        let k_constant = self.k_constant;
        let df_model = rank as f64 - k_constant as f64;
        let df_resid = nobs - rank as f64;

        let resid_response = &self.endog - &mu;

        // Pearson chi-squared and residuals; Tweedie scale is free (estimated).
        let mut pearson_chi2 = 0.0;
        let mut resid_pearson = Array1::<f64>::zeros(n);
        let mut resid_deviance = Array1::<f64>::zeros(n);
        for i in 0..n {
            let v = self.variance(mu[i]);
            let pr = (self.endog[i] - mu[i]) / v.sqrt();
            resid_pearson[i] = pr;
            pearson_chi2 += pr * pr;
            let ud = tweedie_unit_deviance(self.endog[i], mu[i], self.var_power).max(0.0);
            resid_deviance[i] = (self.endog[i] - mu[i]).signum() * ud.sqrt();
        }

        let scale = pearson_chi2 / df_resid;

        let cov_params = &normalized_cov_params * scale;
        let k = params.len();
        let mut bse = Array1::<f64>::zeros(k);
        for i in 0..k {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        // Exact maximised Tweedie log-likelihood at the estimated dispersion.
        let llf: f64 = (0..n)
            .map(|i| tweedie_loglike_obs(self.endog[i], mu[i], self.var_power, scale))
            .sum();

        // Null (intercept-only) deviance: fitted mean is ȳ regardless of link.
        let ybar = self.endog.sum() / nobs;
        let mu_null = Array1::from_elem(n, ybar);
        // SAFETY: owned contiguous arrays (`endog` by value, `mu_null` from `from_elem`).
        let null_deviance = self.deviance(
            self.endog.as_slice().unwrap_or(&[]),
            mu_null.as_slice().unwrap_or(&[]),
        );

        // Null log-likelihood at the *fitted* dispersion (the reference's
        // convention), backing the Cox–Snell pseudo-R² in the summary header.
        let llf_null: f64 = (0..n)
            .map(|i| tweedie_loglike_obs(self.endog[i], ybar, self.var_power, scale))
            .sum();

        let k_params = df_model + 1.0;
        let aic = -2.0 * llf + 2.0 * k_params;
        let bic_deviance = deviance - df_resid * nobs.ln();
        let bic_llf = -2.0 * llf + k_params * nobs.ln();

        GlmResults {
            params,
            bse,
            tvalues,
            pvalues,
            nobs,
            rank,
            k_constant,
            df_model,
            df_resid,
            scale,
            deviance,
            pearson_chi2,
            null_deviance,
            llf,
            aic,
            bic_deviance,
            bic_llf,
            bic: bic_deviance,
            fittedvalues: mu,
            linear_predictor: eta,
            resid_response,
            resid_pearson,
            resid_deviance,
            normalized_cov_params,
            cov_params,
            n_iter,
            converged,
            family_name: "Tweedie",
            link_name: self.link.name(),
            llf_null,
        }
    }
}

/// Tweedie unit deviance `d(y, μ)` for general power `p ∈ (1, 2)`.
///
/// `d = 2·[ y^(2−p)/((1−p)(2−p)) − y·μ^(1−p)/(1−p) + μ^(2−p)/(2−p) ]`,
/// with the convention `0^(2−p) = 0` (valid since `2 − p > 0`), so a zero
/// response contributes the finite `2·μ^(2−p)/(2−p)`.
fn tweedie_unit_deviance(y: f64, mu: f64, p: f64) -> f64 {
    let y2p = if y > 0.0 { y.powf(2.0 - p) } else { 0.0 };
    let term = y2p / ((1.0 - p) * (2.0 - p)) - y * mu.powf(1.0 - p) / (1.0 - p)
        + mu.powf(2.0 - p) / (2.0 - p);
    2.0 * term
}

/// Per-observation Tweedie log-likelihood `ℓ(y; μ, φ)` for `p ∈ (1, 2)`,
/// evaluated through the Dunn & Smyth (2004) series for the compound
/// Poisson-gamma density.
///
/// Writing `θ = μ^(1−p)/(1−p)`, `κ = μ^(2−p)/(2−p)`, the exponential-dispersion
/// kernel is `(y·θ − κ)/φ`. For `y > 0` the normalising constant `c(y, φ)`
/// is `(1/y)·W(−α, 0; x)` where `α = (2−p)/(1−p)`,
/// `x = ((p−1)φ/y)^α / ((2−p)φ)`, and `W` is the Wright Bessel function
/// `W(a, b; x) = Σ_{j≥0} x^j / (j!·Γ(aj + b))`. With `a = −α = (2−p)/(p−1) > 0`
/// and `b = 0`, the `j = 0` term vanishes (`1/Γ(0) = 0`), and each `j ≥ 1`
/// term is summed in log-space for numerical stability.
fn tweedie_loglike_obs(y: f64, mu: f64, p: f64, phi: f64) -> f64 {
    let theta = mu.powf(1.0 - p) / (1.0 - p);
    let kappa = mu.powf(2.0 - p) / (2.0 - p);
    let base = (y * theta - kappa) / phi;
    if y <= 0.0 {
        return base;
    }
    base + log_density_constant(y, p, phi) - y.ln()
}

/// `ln W(−α, 0; x)` for the compound Poisson-gamma density, computed by the
/// Dunn-Smyth log-space series with adaptive windowing around the dominant
/// term. Returns the log of the Wright-Bessel sum (excluding the leading
/// `−ln y`, which the caller adds).
fn log_density_constant(y: f64, p: f64, phi: f64) -> f64 {
    // a = -alpha = (2-p)/(p-1) > 0 for 1 < p < 2.
    let a = (2.0 - p) / (p - 1.0);
    let alpha = (2.0 - p) / (1.0 - p); // < 0
                                       // ln x = alpha*(ln((p-1)*phi) - ln y) - ln((2-p)*phi).
    let log_x = alpha * (((p - 1.0) * phi).ln() - y.ln()) - ((2.0 - p) * phi).ln();

    // log of term j (>= 1): j*ln x - lnΓ(j+1) - lnΓ(a*j).
    let log_term = |j: f64| j * log_x - lgamma(j + 1.0) - lgamma(a * j);

    // Locate the dominant term by climbing from j = 1.
    let mut j_peak = 1.0_f64;
    let mut best = log_term(1.0);
    let mut j = 2.0_f64;
    loop {
        let v = log_term(j);
        if v > best {
            best = v;
            j_peak = j;
            j += 1.0;
        } else {
            break;
        }
        if j > 1e8 {
            break;
        }
    }

    // Accumulate terms outward from the peak until they fall far below it.
    // A 60-nat (~ exp(-60) ≈ 1e-26 relative) drop is well past f64 precision.
    const DROP: f64 = 60.0;
    let mut terms: Vec<f64> = vec![best];

    // Rightward.
    let mut jr = j_peak + 1.0;
    let mut prev = best;
    loop {
        let v = log_term(jr);
        terms.push(v);
        if best - v > DROP && v < prev {
            break;
        }
        prev = v;
        jr += 1.0;
        if jr > j_peak + 5e5 {
            break;
        }
    }

    // Leftward.
    let mut jl = j_peak - 1.0;
    while jl >= 1.0 {
        let v = log_term(jl);
        terms.push(v);
        if best - v > DROP {
            break;
        }
        jl -= 1.0;
    }

    // log-sum-exp.
    let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = terms.iter().map(|t| (t - m).exp()).sum();
    m + s.ln()
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

    #[test]
    fn var_power_must_be_in_open_unit_interval() {
        let x = Array2::<f64>::ones((4, 1));
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(TweedieGlm::new(y.clone(), x.clone(), 1.0).is_err());
        assert!(TweedieGlm::new(y.clone(), x.clone(), 2.0).is_err());
        assert!(TweedieGlm::new(y.clone(), x.clone(), 0.5).is_err());
        assert!(TweedieGlm::new(y, x, 1.5).is_ok());
    }

    #[test]
    fn unit_deviance_is_zero_at_saturation() {
        // d(y, y) = 0 for y > 0.
        for &y in &[0.3, 1.0, 2.7, 10.0] {
            let d = tweedie_unit_deviance(y, y, 1.5);
            assert!(d.abs() < 1e-12, "d({y},{y}) = {d}");
        }
    }

    #[test]
    fn unit_deviance_finite_at_zero_response() {
        // y = 0 contributes 2*mu^(2-p)/(2-p), finite and positive.
        let d = tweedie_unit_deviance(0.0, 1.5, 1.5);
        let expected = 2.0 * 1.5_f64.powf(0.5) / 0.5;
        assert!((d - expected).abs() < 1e-12, "d(0,1.5)={d}");
    }

    #[test]
    fn loglike_series_matches_reference_values() {
        // Reference values from the authoritative package's Tweedie.loglike_obs
        // (full Dunn-Smyth series via wright_bessel), p = 1.5.
        let cases = [
            (2.3_f64, 1.8_f64, 0.7_f64, -1.4765212538_f64),
            (0.0, 1.2, 0.7, -3.1298431857),
            (5.1, 3.0, 0.5, -2.4552614752),
            (0.4, 0.9, 0.6, -0.4306441317),
        ];
        for (y, mu, phi, want) in cases {
            let got = tweedie_loglike_obs(y, mu, 1.5, phi);
            assert!(
                (got - want).abs() < 1e-7,
                "ll(y={y}, mu={mu}, phi={phi}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn simple_fit_converges() {
        let x = ndarray::array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0]
        ];
        let y = Array1::from_vec(vec![0.0, 1.2, 2.5, 4.1, 6.0, 9.0]);
        let res = TweedieGlm::new(y, x, 1.5).unwrap().fit().unwrap();
        assert!(res.converged);
        assert!(res.llf.is_finite());
        assert!(res.scale > 0.0);
    }
}
