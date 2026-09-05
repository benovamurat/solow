//! The generalized linear model, estimated by iteratively reweighted least
//! squares (IRLS).

use crate::family::Family;
use crate::links::Link;
use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{norm_ppf, norm_sf};
use solow_linalg::{lstsq_qr, pinv};

/// A generalized linear model awaiting estimation.
#[derive(Clone, Debug)]
pub struct Glm {
    endog: Array1<f64>,
    exog: Array2<f64>,
    family: Family,
    link: Link,
    k_constant: usize,
    maxiter: usize,
    tol: f64,
}

impl Glm {
    /// A GLM with the family's canonical/default link.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>, family: Family) -> Result<Self> {
        let link = family.default_link();
        Self::with_link(endog, exog, family, link)
    }

    /// A GLM with an explicit link.
    pub fn with_link(
        endog: Array1<f64>,
        exog: Array2<f64>,
        family: Family,
        link: Link,
    ) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        let k_constant = detect_k_constant(&exog);
        Ok(Glm {
            endog,
            exog,
            family,
            link,
            k_constant,
            maxiter: 300,
            // Absolute tolerance on the change in deviance; converge to the MLE.
            tol: 1e-12,
        })
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
        // Fast path: full-rank weighted design via Householder QR (no per-iteration
        // SVD). Same weighted least-squares solution as the pseudoinverse; falls back
        // to the SVD pseudoinverse for rank-deficient designs.
        if let Some((params, ncp)) = lstsq_qr(&wexog, &wz)? {
            return Ok((params, ncp, p));
        }
        let (pinv_w, sv) = pinv(&wexog)?;
        let params = pinv_w.dot(&wz);
        let ncp = pinv_w.dot(&pinv_w.t());
        let smax = sv.iter().cloned().fold(0.0_f64, f64::max);
        let tol = smax * (sv.len() as f64) * f64::EPSILON;
        let rank = sv.iter().filter(|&&s| s > tol).count();
        Ok((params, ncp, rank))
    }

    /// Estimate the model by IRLS.
    pub fn fit(&self) -> Result<GlmResults> {
        let n = self.endog.len();
        let p = self.exog.ncols();
        let ybar = self.endog.sum() / n as f64;

        let mut mu = self.endog.mapv(|y| self.family.starting_mu(y, ybar));
        let mut eta = mu.mapv(|m| self.link.link(m));
        // `endog` and `mu` are owned, standard-layout arrays, so this contiguous
        // view is always available; surface a clean error rather than panicking
        // in the impossible non-contiguous case.
        let endog_s = self
            .endog
            .as_slice()
            .ok_or_else(|| Error::Value("endog must be contiguous".into()))?;
        let mut dev = self.family.deviance(
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
                let var = self.family.variance(mu[i]);
                w[i] = 1.0 / (gp * gp * var);
                z[i] = eta[i] + (self.endog[i] - mu[i]) * gp;
            }
            let (new_params, new_ncp, new_rank) = self.weighted_lstsq(&z, &w)?;
            params = new_params;
            ncp = new_ncp;
            rank = new_rank;
            eta = self.exog.dot(&params);
            mu = eta.mapv(|e| self.link.inverse(e));
            let dev_new = self.family.deviance(
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

        Ok(GlmResults::new(
            self, params, ncp, rank, mu, eta, dev, converged, n_iter,
        ))
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
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

/// The fitted result of a [`Glm`].
#[derive(Clone, Debug)]
pub struct GlmResults {
    /// Estimated coefficients.
    pub params: Array1<f64>,
    /// Standard errors.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values from the normal distribution.
    pub pvalues: Array1<f64>,

    /// Number of observations.
    pub nobs: f64,
    /// Rank of the design.
    pub rank: usize,
    /// Whether a constant is present (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom (`rank − k_constant`).
    pub df_model: f64,
    /// Residual degrees of freedom (`nobs − rank`).
    pub df_resid: f64,

    /// Dispersion estimate (1 for fixed-scale families).
    pub scale: f64,
    /// Model deviance.
    pub deviance: f64,
    /// Pearson chi-squared statistic.
    pub pearson_chi2: f64,
    /// Deviance of the intercept-only model.
    pub null_deviance: f64,

    /// Log-likelihood.
    pub llf: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Deviance-based BIC (`deviance − df_resid · ln nobs`).
    pub bic_deviance: f64,
    /// Likelihood-based BIC (`−2 llf + (df_model+1) · ln nobs`).
    pub bic_llf: f64,
    /// The conventional `bic` (deviance-based, for compatibility).
    pub bic: f64,

    /// Fitted means `μ`.
    pub fittedvalues: Array1<f64>,
    /// Linear predictor `η`.
    pub linear_predictor: Array1<f64>,
    /// Response residuals `y − μ`.
    pub resid_response: Array1<f64>,
    /// Pearson residuals `(y − μ)/√V(μ)`.
    pub resid_pearson: Array1<f64>,
    /// Deviance residuals `sign(y − μ)·√d(y, μ)`.
    pub resid_deviance: Array1<f64>,

    /// `cov_params / scale`.
    pub normalized_cov_params: Array2<f64>,
    /// Coefficient covariance `scale · normalized_cov_params`.
    pub cov_params: Array2<f64>,

    /// Number of IRLS iterations.
    pub n_iter: usize,
    /// Whether IRLS converged.
    pub converged: bool,

    /// Display name of the error family (e.g. `"Poisson"`), for the summary's
    /// `Model Family:` field.
    pub family_name: &'static str,
    /// Display name of the link (e.g. `"Log"`), for the summary's
    /// `Link Function:` field.
    pub link_name: &'static str,
    /// Intercept-only (null) log-likelihood, evaluated at the fitted scale.
    /// Drives the Cox–Snell pseudo-R² reported in the summary header.
    pub llf_null: f64,
}

impl GlmResults {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &Glm,
        params: Array1<f64>,
        normalized_cov_params: Array2<f64>,
        rank: usize,
        mu: Array1<f64>,
        eta: Array1<f64>,
        deviance: f64,
        converged: bool,
        n_iter: usize,
    ) -> GlmResults {
        let n = model.endog.len();
        let nobs = n as f64;
        let k_constant = model.k_constant;
        let df_model = rank as f64 - k_constant as f64;
        let df_resid = nobs - rank as f64;
        let family = model.family;

        let resid_response = &model.endog - &mu;

        // Pearson chi-squared and residuals.
        let mut pearson_chi2 = 0.0;
        let mut resid_pearson = Array1::<f64>::zeros(n);
        let mut resid_deviance = Array1::<f64>::zeros(n);
        for i in 0..n {
            let v = family.variance(mu[i]);
            let pr = (model.endog[i] - mu[i]) / v.sqrt();
            resid_pearson[i] = pr;
            pearson_chi2 += pr * pr;
            let ud = family.unit_deviance(model.endog[i], mu[i]).max(0.0);
            resid_deviance[i] = (model.endog[i] - mu[i]).signum() * ud.sqrt();
        }

        let scale = if family.fixed_scale() {
            1.0
        } else {
            pearson_chi2 / df_resid
        };

        let cov_params = &normalized_cov_params * scale;
        let k = params.len();
        let mut bse = Array1::<f64>::zeros(k);
        for i in 0..k {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        // The Gaussian+identity log-likelihood uses the ML variance (SSR/nobs)
        // rather than the Pearson dispersion, matching the reference (and OLS).
        let llf_scale = if family == Family::Gaussian && model.link == Link::Identity {
            resid_response.iter().map(|r| r * r).sum::<f64>() / nobs
        } else {
            scale
        };
        // SAFETY: owned contiguous arrays (`endog` by value, `mu` from `mapv`),
        // so `as_slice()` is always `Some`; fallback to an empty slice never fires.
        let endog_s = model.endog.as_slice().unwrap_or(&[]);
        let llf = family.loglike(endog_s, mu.as_slice().unwrap_or(&[]), llf_scale);

        // Null (intercept-only) deviance: fitted mean is ȳ regardless of link.
        let ybar = model.endog.sum() / nobs;
        let mu_null = Array1::from_elem(n, ybar);
        // SAFETY: owned contiguous arrays (see above).
        let null_deviance = family.deviance(endog_s, mu_null.as_slice().unwrap_or(&[]));

        // Null log-likelihood. Unlike `llf`, the reference evaluates this at the
        // fitted dispersion `scale` for every family (including Gaussian+identity,
        // where `llf` itself uses the concentrated `SSR/nobs`). This is the
        // baseline for the Cox–Snell pseudo-R² shown in the summary header.
        let llf_null = family.loglike(endog_s, mu_null.as_slice().unwrap_or(&[]), scale);

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
            family_name: family.name(),
            link_name: model.link.name(),
            llf_null,
        }
    }

    /// McFadden's pseudo-R² based on deviance: `1 − deviance / null_deviance`.
    pub fn pseudo_rsquared(&self) -> f64 {
        1.0 - self.deviance / self.null_deviance
    }

    /// Cox–Snell (likelihood-ratio) pseudo-R²: `1 − exp((llf_null − llf)·2/nobs)`.
    ///
    /// This is the variant the reference reports as `Pseudo R-squ. (CS)` in a GLM
    /// summary; it is defined for both discrete and continuous responses.
    pub fn pseudo_rsquared_cs(&self) -> f64 {
        1.0 - ((self.llf_null - self.llf) * (2.0 / self.nobs)).exp()
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

    /// A full results table in the canonical reference GLM layout: the
    /// "Generalized Linear Model Regression Results" header block followed by the
    /// coefficient table (with `z`, `P>|z|`, and the 95 % confidence interval).
    ///
    /// `names`, if given, labels the coefficients; otherwise `x0, x1, …` are used.
    /// The output matches the reference field-for-field (the only volatile fields
    /// are the `Date:`/`Time:` stamps, which carry the current UTC clock).
    pub fn summary(&self, names: Option<&[&str]>) -> String {
        self.summary_titled("y", names)
    }

    /// As [`summary`](Self::summary), with the dependent-variable label of your
    /// choosing.
    pub fn summary_titled(&self, dep: &str, names: Option<&[&str]>) -> String {
        use std::fmt::Write as _;
        const W: usize = 78;
        let bar = "=".repeat(W);
        let dash = "-".repeat(W);
        let k = self.params.len();
        let ci = self.conf_int(0.05);
        let (date, time) = utc_now_strings();

        let mut s = String::new();
        // Title (centered).
        let title = "Generalized Linear Model Regression Results";
        let pad = (W.saturating_sub(title.len())) / 2;
        let _ = writeln!(s, "{}{}", " ".repeat(pad), title);
        let _ = writeln!(s, "{bar}");

        // Header: two label/value pairs per line, in the reference's column widths.
        let row = |l1: &str, v1: String, l2: &str, v2: String, s: &mut String| {
            let _ = writeln!(s, "{:<20}{:>17}   {:<21}{:>17}", l1, v1, l2, v2);
        };
        row(
            "Dep. Variable:",
            dep.to_string(),
            "No. Observations:",
            format!("{:.0}", self.nobs),
            &mut s,
        );
        row(
            "Model:",
            "GLM".into(),
            "Df Residuals:",
            format!("{:.0}", self.df_resid),
            &mut s,
        );
        row(
            "Model Family:",
            self.family_name.to_string(),
            "Df Model:",
            format!("{:.0}", self.df_model),
            &mut s,
        );
        row(
            "Link Function:",
            self.link_name.to_string(),
            "Scale:",
            fmt_g(self.scale, 5),
            &mut s,
        );
        row(
            "Method:",
            "IRLS".into(),
            "Log-Likelihood:",
            fmt_g(self.llf, 5),
            &mut s,
        );
        row("Date:", date, "Deviance:", fmt_g(self.deviance, 5), &mut s);
        row(
            "Time:",
            time,
            "Pearson chi2:",
            fmt_g(self.pearson_chi2, 3),
            &mut s,
        );
        row(
            "No. Iterations:",
            format!("{}", self.n_iter),
            "Pseudo R-squ. (CS):",
            fmt_g(self.pseudo_rsquared_cs(), 4),
            &mut s,
        );
        row(
            "Covariance Type:",
            "nonrobust".into(),
            "",
            String::new(),
            &mut s,
        );
        let _ = writeln!(s, "{bar}");

        // Coefficient table (z-inference). The column widths reproduce the
        // reference's parameter table exactly (data-column right edges at
        // 21/32/43/54/66/78).
        let _ = writeln!(
            s,
            "{:<11}{:>10}{:>11}{:>11}{:>11}{:>12}{:>12}",
            "", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
        );
        let _ = writeln!(s, "{dash}");
        for i in 0..k {
            let name = names
                .and_then(|n| n.get(i).copied())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("x{i}"));
            let _ = writeln!(
                s,
                "{:<11}{:>10.4}{:>11.3}{:>11.3}{:>11.3}{:>12.3}{:>12.3}",
                name,
                self.params[i],
                self.bse[i],
                self.tvalues[i],
                self.pvalues[i],
                ci[[i, 0]],
                ci[[i, 1]]
            );
        }
        let _ = write!(s, "{bar}");
        s
    }
}

// ---------------------------------------------------------------------------
// Summary formatting helpers. These mirror the reference's printf semantics
// (`%#.Ng` and the `%a, %d %b %Y` / `%H:%M:%S` header stamps); the same small
// helpers back the OLS summary in `solow-regression`.
// ---------------------------------------------------------------------------

/// Format with `sig` significant figures keeping trailing zeros, matching the
/// reference's `%#.Ng`: scientific (with a signed, ≥2-digit exponent) when the
/// exponent is `< -4` or `>= sig`, fixed otherwise.
fn fmt_g(v: f64, sig: usize) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return format!("{:.*}", sig.saturating_sub(1), 0.0);
    }
    // Determine the decimal exponent *after* rounding to `sig` significant
    // figures: rounding can carry across a power-of-10 boundary (e.g.
    // 0.9999707 → 1.000, 9.9996 → 10.00), and C's `%g` picks the precision
    // from the rounded magnitude, not the raw one.
    let raw_exp = v.abs().log10().floor() as i32;
    let rounded = {
        let scale = 10f64.powi(sig as i32 - 1 - raw_exp);
        (v.abs() * scale).round() / scale
    };
    let exp = rounded.log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        // Rust prints e.g. "8.344e-6"; the reference uses a signed, ≥2-digit
        // exponent ("8.344e-06"), so normalize the exponent field.
        let raw = format!("{:.*e}", sig.saturating_sub(1), v);
        if let Some(epos) = raw.find('e') {
            let (mant, exp_part) = raw.split_at(epos);
            let digits = &exp_part[1..];
            let (sign, mag) = if let Some(rest) = digits.strip_prefix('-') {
                ("-", rest)
            } else {
                ("+", digits.strip_prefix('+').unwrap_or(digits))
            };
            format!("{mant}e{sign}{mag:0>2}")
        } else {
            raw
        }
    } else {
        let dec = (sig as i32 - 1 - exp).max(0) as usize;
        format!("{:.*}", dec, v)
    }
}

/// Current UTC date/time as `("Thu, 18 Jun 2026", "03:21:29")`, matching the
/// reference header look. Uses only `std::time` via the civil-from-days algorithm.
fn utc_now_strings() -> (String, String) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (h, mi, sc) = (tod / 3600, (tod % 3600) / 60, tod % 60);

    // Howard Hinnant's civil_from_days (1970-01-01 == day 0).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let wdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    // 1970-01-01 was a Thursday (index 3).
    let wd = (days.rem_euclid(7) + 3).rem_euclid(7) as usize;
    let date = format!(
        "{}, {:02} {} {}",
        wdays[wd],
        d,
        months[(m as usize - 1).min(11)],
        year
    );
    let time = format!("{h:02}:{mi:02}:{sc:02}");
    (date, time)
}

#[cfg(test)]
mod summary_tests {
    use super::*;
    use crate::Family;
    use ndarray::array;

    #[test]
    fn fmt_g_rounds_across_power_of_ten_boundary() {
        // Plain magnitudes keep the reference's `%#.Ng` form.
        assert_eq!(fmt_g(1.0, 5), "1.0000");
        assert_eq!(fmt_g(0.027, 5), "0.027000");
        // Rounding that carries to the next power of ten must take the precision
        // of the *rounded* magnitude (matching C `%g`), not the raw one. This is
        // the `Pseudo R-squ. (CS)` case where 0.9999971 must render as "1.000".
        assert_eq!(fmt_g(0.99999707, 4), "1.000");
        assert_eq!(fmt_g(9.9996, 4), "10.00");
    }

    #[test]
    fn glm_summary_has_every_block() {
        // Poisson GLM with an intercept; the summary must carry every block of the
        // reference's "Generalized Linear Model Regression Results" layout.
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
            [1.0, 8.0],
            [1.0, 9.0],
        ];
        let y = array![1.0, 2.0, 4.0, 7.0, 12.0, 11.0, 15.0, 20.0, 25.0, 30.0];
        let res = Glm::new(y, x, Family::Poisson).unwrap().fit().unwrap();
        let s = res.summary(Some(&["const", "x1"]));

        // Title + header labels.
        assert!(s.contains("Generalized Linear Model Regression Results"));
        assert!(s.contains("Dep. Variable:"));
        assert!(s.contains("Model:") && s.contains("GLM"));
        assert!(s.contains("Model Family:") && s.contains("Poisson"));
        assert!(s.contains("Link Function:") && s.contains("Log"));
        assert!(s.contains("Method:") && s.contains("IRLS"));
        assert!(s.contains("Date:") && s.contains("Time:"));
        assert!(s.contains("No. Observations:"));
        assert!(s.contains("Df Residuals:") && s.contains("Df Model:"));
        assert!(s.contains("Scale:"));
        assert!(s.contains("Log-Likelihood:"));
        assert!(s.contains("Deviance:"));
        assert!(s.contains("Pearson chi2:"));
        assert!(s.contains("Pseudo R-squ. (CS):"));
        assert!(s.contains("Covariance Type:") && s.contains("nonrobust"));
        // Coefficient table: z / P>|z| (no t), the CI columns, and labels.
        assert!(s.contains("coef") && s.contains("std err"));
        assert!(s.contains("P>|z|") && !s.contains("P>|t|"));
        assert!(s.contains("[0.025") && s.contains("0.975]"));
        assert!(s.contains("const") && s.contains("x1"));
        // GLM has no normality / autocorrelation block.
        assert!(!s.contains("Omnibus") && !s.contains("Durbin-Watson"));
    }
}
