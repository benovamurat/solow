//! Rubin's rules for combining multiple-imputation estimates.
//!
//! Given the parameter estimates and their covariance matrices from `m`
//! separately analysed imputed data sets, [`combine`] pools them into a single
//! estimate together with the within-, between- and total-imputation
//! covariance matrices, the fraction of missing information, and the
//! Barnard–Rubin degrees of freedom. Everything here is fully deterministic:
//! the per-imputation inputs are taken as given, so the result depends only on
//! closed-form linear algebra and never on a random draw.
//!
//! The pooled point estimate and total covariance reproduce the reference
//! `MICE.combine` exactly (`params`, `cov_within`, `cov_between`,
//! `cov_total`, and `fmi`). The Barnard–Rubin degrees of freedom follow the
//! canonical small-sample correction of Barnard & Rubin (1999), which refines
//! the original Rubin (1987) degrees of freedom toward `dfcom` (the
//! complete-data residual degrees of freedom) as the fraction of missing
//! information shrinks.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

/// Pooled multiple-imputation estimate produced by Rubin's combining rules.
///
/// All vector quantities are indexed by parameter; all matrices are
/// `p x p` covariance matrices over the `p` parameters.
#[derive(Debug, Clone)]
pub struct CombinedEstimate {
    /// Pooled point estimate, the mean of the per-imputation parameter vectors.
    pub params: Array1<f64>,
    /// Within-imputation covariance: the mean of the per-imputation covariances.
    pub cov_within: Array2<f64>,
    /// Between-imputation covariance: the sample covariance of the
    /// per-imputation parameter vectors (with `m - 1` in the denominator).
    pub cov_between: Array2<f64>,
    /// Total (pooled) covariance, `cov_within + (1 + 1/m) * cov_between`.
    pub cov_total: Array2<f64>,
    /// Standard errors, the square roots of the diagonal of [`Self::cov_total`].
    pub bse: Array1<f64>,
    /// Relative increase in variance due to nonresponse, per parameter.
    pub relative_increase: Array1<f64>,
    /// Fraction of missing information, per parameter.
    pub fmi: Array1<f64>,
    /// Barnard–Rubin pooled degrees of freedom, per parameter.
    pub df: Array1<f64>,
    /// Number of imputations combined.
    pub m: usize,
}

impl CombinedEstimate {
    /// Wald `t`-statistics, `params / bse`.
    pub fn tvalues(&self) -> Array1<f64> {
        &self.params / &self.bse
    }

    /// Two-sided p-values from Student's `t` with the Barnard–Rubin degrees of
    /// freedom of each parameter.
    pub fn pvalues(&self) -> Array1<f64> {
        let t = self.tvalues();
        Array1::from_iter(
            t.iter()
                .zip(self.df.iter())
                .map(|(&tv, &df)| 2.0 * solow_distributions::t_sf(tv.abs(), df)),
        )
    }

    /// Two-sided confidence intervals at confidence level `1 - alpha`.
    ///
    /// Returns a `p x 2` matrix whose rows are `[lower, upper]`, using the
    /// Student-`t` critical value with the Barnard–Rubin degrees of freedom.
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        let p = self.params.len();
        let mut out = Array2::<f64>::zeros((p, 2));
        for i in 0..p {
            let q = solow_distributions::t_ppf(1.0 - alpha / 2.0, self.df[i]);
            out[[i, 0]] = self.params[i] - q * self.bse[i];
            out[[i, 1]] = self.params[i] + q * self.bse[i];
        }
        out
    }
}

/// Combine per-imputation estimates with Rubin's rules.
///
/// `params_list` holds the `m` parameter vectors (each of length `p`) and
/// `cov_list` the corresponding `m` covariance matrices (each `p x p`).
/// `dfcom` is the complete-data residual degrees of freedom used by the
/// Barnard–Rubin small-sample correction; pass `f64::INFINITY` to recover the
/// large-sample (Rubin 1987) degrees of freedom.
///
/// At least two imputations are required (the between-imputation variance is
/// otherwise undefined).
pub fn combine(
    params_list: &[Array1<f64>],
    cov_list: &[Array2<f64>],
    dfcom: f64,
) -> Result<CombinedEstimate> {
    let m = params_list.len();
    if m < 2 {
        return Err(Error::Value(
            "combining requires at least two imputations".into(),
        ));
    }
    if cov_list.len() != m {
        return Err(Error::Shape(format!(
            "params_list has {m} entries but cov_list has {}",
            cov_list.len()
        )));
    }
    let p = params_list[0].len();
    if p == 0 {
        return Err(Error::Value("parameter vectors are empty".into()));
    }
    for (k, par) in params_list.iter().enumerate() {
        if par.len() != p {
            return Err(Error::Shape(format!(
                "params_list[{k}] has length {} (expected {p})",
                par.len()
            )));
        }
    }
    for (k, cov) in cov_list.iter().enumerate() {
        if cov.dim() != (p, p) {
            return Err(Error::Shape(format!(
                "cov_list[{k}] has shape {:?} (expected {p}x{p})",
                cov.dim()
            )));
        }
    }

    let mf = m as f64;

    // Pooled point estimate: mean of the per-imputation parameter vectors.
    let mut params = Array1::<f64>::zeros(p);
    for par in params_list {
        params += par;
    }
    params /= mf;

    // Within-imputation covariance: mean of the per-imputation covariances.
    let mut cov_within = Array2::<f64>::zeros((p, p));
    for cov in cov_list {
        cov_within += cov;
    }
    cov_within /= mf;

    // Between-imputation covariance: sample covariance of the parameter
    // vectors, with the unbiased `m - 1` denominator (matching `np.cov`).
    let mut cov_between = Array2::<f64>::zeros((p, p));
    for par in params_list {
        let d = par - &params;
        for i in 0..p {
            for j in 0..p {
                cov_between[[i, j]] += d[i] * d[j];
            }
        }
    }
    cov_between /= mf - 1.0;

    // Total covariance.
    let f = 1.0 + 1.0 / mf;
    let cov_total = &cov_within + &(f * &cov_between);

    // Per-parameter scalar quantities derived from the diagonals.
    let mut bse = Array1::<f64>::zeros(p);
    let mut relative_increase = Array1::<f64>::zeros(p);
    let mut fmi = Array1::<f64>::zeros(p);
    let mut df = Array1::<f64>::zeros(p);
    let dfm = mf - 1.0;
    for i in 0..p {
        let ubar = cov_within[[i, i]];
        let b = cov_between[[i, i]];
        let t = cov_total[[i, i]];
        bse[i] = t.sqrt();
        relative_increase[i] = f * b / ubar;
        let lambda = f * b / t;
        fmi[i] = lambda;
        df[i] = barnard_rubin_df(m, lambda, dfm, dfcom);
    }

    Ok(CombinedEstimate {
        params,
        cov_within,
        cov_between,
        cov_total,
        bse,
        relative_increase,
        fmi,
        df,
        m,
    })
}

/// Barnard–Rubin pooled degrees of freedom for one parameter.
///
/// `lambda` is the fraction of missing information, `dfm = m - 1`, and `dfcom`
/// is the complete-data residual degrees of freedom. With `dfcom` infinite the
/// observed-data adjustment vanishes and the result collapses to the original
/// Rubin (1987) value `dfm / lambda^2`.
fn barnard_rubin_df(_m: usize, lambda: f64, dfm: f64, dfcom: f64) -> f64 {
    // Guard the degenerate case of no between-imputation variability, which
    // would otherwise divide by zero; the limit is the complete-data df.
    if lambda <= 0.0 {
        return if dfcom.is_finite() {
            dfcom
        } else {
            f64::INFINITY
        };
    }
    let nu_old = dfm / (lambda * lambda);
    if !dfcom.is_finite() {
        return nu_old;
    }
    let nu_obs = (dfcom + 1.0) / (dfcom + 3.0) * dfcom * (1.0 - lambda);
    nu_old * nu_obs / (nu_old + nu_obs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pooled_estimate_is_the_mean() {
        let p1 = array![1.0, 3.0];
        let p2 = array![3.0, 5.0];
        let c1 = array![[0.1, 0.0], [0.0, 0.2]];
        let c2 = array![[0.3, 0.0], [0.0, 0.4]];
        let res = combine(&[p1, p2], &[c1, c2], f64::INFINITY).unwrap();
        assert!((res.params[0] - 2.0).abs() < 1e-12);
        assert!((res.params[1] - 4.0).abs() < 1e-12);
        // Within is the mean of the covariances.
        assert!((res.cov_within[[0, 0]] - 0.2).abs() < 1e-12);
        assert!((res.cov_within[[1, 1]] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn identical_imputations_give_zero_between_and_full_df() {
        let p = array![1.0, -2.0];
        let c = array![[0.05, 0.01], [0.01, 0.08]];
        let res = combine(
            &[p.clone(), p.clone(), p.clone()],
            &[c.clone(), c.clone(), c.clone()],
            50.0,
        )
        .unwrap();
        // No between-imputation variance: total equals within, fmi is zero.
        for i in 0..2 {
            for j in 0..2 {
                assert!((res.cov_between[[i, j]]).abs() < 1e-12);
                assert!((res.cov_total[[i, j]] - c[[i, j]]).abs() < 1e-12);
            }
            assert!(res.fmi[i].abs() < 1e-12);
            // Barnard-Rubin df collapses to dfcom when fmi -> 0.
            assert!((res.df[i] - 50.0).abs() < 1e-9);
        }
    }

    #[test]
    fn infinite_dfcom_recovers_rubin_1987_df() {
        let p1 = array![1.0];
        let p2 = array![1.4];
        let p3 = array![0.7];
        let c = array![[0.02]];
        let res = combine(
            &[p1, p2, p3],
            &[c.clone(), c.clone(), c.clone()],
            f64::INFINITY,
        )
        .unwrap();
        let m = 3.0;
        let f = 1.0 + 1.0 / m;
        let lambda = res.fmi[0];
        let expected = (m - 1.0) / (lambda * lambda);
        assert!((res.df[0] - expected).abs() < 1e-9);
        // Sanity: relative increase consistent with fmi.
        let r = res.relative_increase[0];
        assert!((lambda - r / (1.0 + r)).abs() < 1e-12);
        let _ = f;
    }

    #[test]
    fn rejects_single_imputation_and_mismatched_shapes() {
        let p = array![1.0];
        let c = array![[0.1]];
        assert!(combine(std::slice::from_ref(&p), std::slice::from_ref(&c), 10.0).is_err());
        let p2 = array![1.0, 2.0];
        assert!(combine(&[p, p2], &[c.clone(), c], 10.0).is_err());
    }
}
