//! Ordinary-least-squares residual diagnostics: serial-correlation,
//! functional-form, and conditional-heteroscedasticity tests, plus nested-model
//! comparison statistics.
//!
//! Each test builds a small auxiliary OLS fit (via [`solow_regression`]) on the
//! supplied residuals or design and reports a Lagrange-multiplier (`nobs · R²`,
//! chi-squared) statistic alongside the matching F statistic from the parameter
//! restriction. The implementations mirror the reference
//! `…stats.diagnostic` (`acorr_breusch_godfrey`, `linear_reset`, `het_arch`,
//! `acorr_lm`) and the `compare_lr_test` / `compare_f_test` methods of the
//! linear-regression results object.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::{chi2_sf, f_sf};
use solow_linalg::inv;
use solow_regression::{LinearModel, LinearResults};

/// Build the matrix of lagged columns of `x` with `trim="both"` semantics.
///
/// For a series of length `n` and `k` lags this returns an `(n − k) × k` matrix
/// whose row `t` (for `t = 0 .. n−k`) holds `[x[k+t−1], x[k+t−2], …, x[k+t−k]]`,
/// i.e. column `j` is lag `j + 1`. Mirrors `lagmat(x, k, trim="both")`.
fn lagmat_both(x: &[f64], k: usize) -> Array2<f64> {
    let n = x.len();
    let rows = n - k;
    let mut out = Array2::<f64>::zeros((rows, k));
    for t in 0..rows {
        // The "current" index in the original series is `k + t`.
        let base = k + t;
        for j in 0..k {
            out[[t, j]] = x[base - 1 - j];
        }
    }
    out
}

/// Horizontally stack a leading column of ones in front of `m`.
fn with_const(m: &Array2<f64>) -> Array2<f64> {
    let (r, c) = m.dim();
    let mut out = Array2::<f64>::zeros((r, c + 1));
    for i in 0..r {
        out[[i, 0]] = 1.0;
        for j in 0..c {
            out[[i, j + 1]] = m[[i, j]];
        }
    }
    out
}

/// Default highest lag: `min(10, nobs / 5)`, matching the reference rule.
fn default_nlags(nobs: usize) -> usize {
    (nobs / 5).min(10)
}

/// Wald test of the linear restriction `R · params = 0` for a fitted OLS model.
///
/// With `use_f = false` returns the chi-squared form
/// `(Rβ)' [R · cov_params · R']⁻¹ (Rβ)` and its `chi²(J)` survival p-value;
/// with `use_f = true` returns that quadratic form divided by `J` together with
/// its `F(J, df_resid)` survival p-value. `r_mat` is `J × k`. Mirrors the
/// reference `wald_test(r_mat, use_f=…, scalar=True)`.
fn wald_test(res: &LinearResults, r_mat: &Array2<f64>, use_f: bool) -> Result<(f64, f64)> {
    let j = r_mat.nrows();
    let rb = r_mat.dot(&res.params); // J
    let rv = r_mat.dot(&res.cov_params); // J × k
    let rvr = rv.dot(&r_mat.t()); // J × J
    let rvr_inv = inv(&rvr)?;
    let mid = rvr_inv.dot(&rb); // J
    let quad = rb.dot(&mid);
    if use_f {
        let stat = quad / j as f64;
        Ok((stat, f_sf(stat, j as f64, res.df_resid)))
    } else {
        Ok((quad, chi2_sf(quad, j as f64)))
    }
}

/// Generic Lagrange-multiplier test for autocorrelation (Engle's `acorr_lm`).
///
/// Regresses `resid[k..]` on a constant and `k = nlags` of its own lags and
/// reports `lm = (nobs − ddof) · R²` with a `chi²(k)` p-value, plus the
/// auxiliary regression's overall F statistic and p-value. When `nlags` is
/// `None` the reference default `min(10, nobs / 5)` is used. Returns
/// `(lm, lm_pvalue, fvalue, f_pvalue)`. Mirrors the reference `acorr_lm` with
/// `cov_type="nonrobust"`.
pub fn acorr_lm(
    resid: &Array1<f64>,
    nlags: Option<usize>,
    ddof: usize,
) -> Result<(f64, f64, f64, f64)> {
    let n = resid.len();
    let k = nlags.unwrap_or_else(|| default_nlags(n));
    if k == 0 {
        return Err(Error::Value("nlags must be >= 1".into()));
    }
    if k >= n {
        return Err(Error::Value("nlags too large for series length".into()));
    }
    let r: Vec<f64> = resid.to_vec();
    let lags = lagmat_both(&r, k);
    let nobs = lags.nrows();
    let design = with_const(&lags);
    let yshort = Array1::from_iter(r[n - nobs..].iter().copied());
    let res = LinearModel::ols(yshort, design)?.fit()?;
    let lm = (nobs as f64 - ddof as f64) * res.rsquared;
    let lm_pvalue = chi2_sf(lm, k as f64);
    Ok((lm, lm_pvalue, res.fvalue, res.f_pvalue))
}

/// Engle's ARCH Lagrange-multiplier test (`het_arch`).
///
/// Equivalent to [`acorr_lm`] applied to the *squared* residuals: it tests for
/// autoregressive conditional heteroscedasticity. Returns
/// `(lm, lm_pvalue, fvalue, f_pvalue)`. Mirrors the reference `het_arch`.
pub fn het_arch(
    resid: &Array1<f64>,
    nlags: Option<usize>,
    ddof: usize,
) -> Result<(f64, f64, f64, f64)> {
    let sq = resid.mapv(|v| v * v);
    acorr_lm(&sq, nlags, ddof)
}

/// Breusch–Godfrey Lagrange-multiplier test for residual autocorrelation.
///
/// Takes the OLS residuals together with the *original* model design `exog` and
/// runs the auxiliary regression of the residuals on `exog` augmented with a
/// constant and `nlags` lags of the residuals (lags before the sample start are
/// zero-padded, matching the reference). Reports `lm = nobs · R²` with a
/// `chi²(nlags)` p-value and the F statistic of the joint restriction that all
/// `nlags` lag-coefficients are zero, with its `F(nlags, df_resid)` p-value.
/// When `nlags` is `None` the reference default `min(10, nobs / 5)` is used.
/// Returns `(lm, lm_pvalue, fvalue, f_pvalue)`. Mirrors the reference
/// `acorr_breusch_godfrey`.
pub fn acorr_breusch_godfrey(
    resid: &Array1<f64>,
    exog: &Array2<f64>,
    nlags: Option<usize>,
) -> Result<(f64, f64, f64, f64)> {
    let n = resid.len();
    if exog.nrows() != n {
        return Err(Error::Shape("exog rows must equal residual length".into()));
    }
    let k = nlags.unwrap_or_else(|| default_nlags(n));
    if k == 0 {
        return Err(Error::Value("nlags must be >= 1".into()));
    }
    // Prepend `k` zeros to the residual series, then take both-trimmed lags so
    // the auxiliary sample has exactly `n` rows.
    let mut padded = vec![0.0; k];
    padded.extend(resid.iter().copied());
    let lags = lagmat_both(&padded, k); // n × k
    let nobs = lags.nrows();
    debug_assert_eq!(nobs, n);
    let lag_const = with_const(&lags); // n × (k+1)

    // exog | 1 | lag1 .. lagk
    let k_old = exog.ncols();
    let k_vars = k_old + lag_const.ncols();
    let mut design = Array2::<f64>::zeros((nobs, k_vars));
    for i in 0..nobs {
        for j in 0..k_old {
            design[[i, j]] = exog[[i, j]];
        }
        for j in 0..lag_const.ncols() {
            design[[i, k_old + j]] = lag_const[[i, j]];
        }
    }
    // xshort = last `nobs` of the padded series = the original residuals.
    let yshort = resid.clone();
    let res = LinearModel::ols(yshort, design)?.fit()?;
    let lm = nobs as f64 * res.rsquared;
    let lm_pvalue = chi2_sf(lm, k as f64);

    // F test: the final `k` coefficients (the residual lags) are jointly zero.
    let mut r_mat = Array2::<f64>::zeros((k, k_vars));
    for i in 0..k {
        r_mat[[i, k_vars - k + i]] = 1.0;
    }
    let (fvalue, f_pvalue) = wald_test(&res, &r_mat, true)?;
    Ok((lm, lm_pvalue, fvalue, f_pvalue))
}

/// Augmentation scheme for [`linear_reset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetAug {
    /// Augment with powers of the fitted values `Xβ̂` (the default).
    Fitted,
    /// Augment with powers of the non-constant, non-binary columns of `exog`.
    Exog,
}

/// Ramsey's RESET test for neglected nonlinearity.
///
/// Refits the model with the design augmented by powers `2, …, power` of either
/// the original fitted values (`ResetAug::Fitted`) or the non-constant,
/// non-binary `exog` columns (`ResetAug::Exog`), then performs a Wald test of
/// the null that all added coefficients are zero. With `use_f = false` the
/// chi-squared form is returned; with `use_f = true` the F form. `power` must be
/// `≥ 2` and `exog` must contain at least one non-constant column. Returns
/// `(statistic, pvalue)`. Mirrors the reference `linear_reset`.
pub fn linear_reset(
    endog: &Array1<f64>,
    exog: &Array2<f64>,
    power: usize,
    test_type: ResetAug,
    use_f: bool,
) -> Result<(f64, f64)> {
    if power < 2 {
        return Err(Error::Value("power must be >= 2".into()));
    }
    let (n, k) = exog.dim();
    if endog.len() != n {
        return Err(Error::Shape("endog length must equal exog rows".into()));
    }

    // Columns to be raised to powers.
    let aug_base: Array2<f64> = match test_type {
        ResetAug::Fitted => {
            let res = LinearModel::ols(endog.clone(), exog.clone())?.fit()?;
            // n × 1 column of fitted values.
            let mut a = Array2::<f64>::zeros((n, 1));
            for i in 0..n {
                a[[i, 0]] = res.fittedvalues[i];
            }
            a
        }
        ResetAug::Exog => {
            // Drop constant and binary columns (a column is binary if every
            // entry equals the column max or the column min).
            let mut keep: Vec<usize> = Vec::new();
            for j in 0..k {
                let col = exog.column(j);
                let mut mx = f64::NEG_INFINITY;
                let mut mn = f64::INFINITY;
                for &v in col.iter() {
                    mx = mx.max(v);
                    mn = mn.min(v);
                }
                let binary = col.iter().all(|&v| v == mx || v == mn);
                if !binary {
                    keep.push(j);
                }
            }
            if keep.is_empty() {
                return Err(Error::Value(
                    "model contains only constant or binary data".into(),
                ));
            }
            let mut a = Array2::<f64>::zeros((n, keep.len()));
            for (cc, &j) in keep.iter().enumerate() {
                for i in 0..n {
                    a[[i, cc]] = exog[[i, j]];
                }
            }
            a
        }
    };
    let base_cols = aug_base.ncols();
    let powers: Vec<usize> = (2..=power).collect();
    let nrestr = base_cols * powers.len();

    // Augmented design: exog | base^2 | base^3 | … | base^power.
    let k_full = k + nrestr;
    let mut design = Array2::<f64>::zeros((n, k_full));
    for i in 0..n {
        for j in 0..k {
            design[[i, j]] = exog[[i, j]];
        }
    }
    let mut col = k;
    for &p in &powers {
        for bc in 0..base_cols {
            for i in 0..n {
                design[[i, col]] = aug_base[[i, bc]].powi(p as i32);
            }
            col += 1;
        }
    }

    let res = LinearModel::ols(endog.clone(), design)?.fit()?;
    // Restriction: the last `nrestr` coefficients are jointly zero.
    let mut r_mat = Array2::<f64>::zeros((nrestr, k_full));
    for i in 0..nrestr {
        r_mat[[i, k_full - nrestr + i]] = 1.0;
    }
    wald_test(&res, &r_mat, use_f)
}

/// Likelihood-ratio comparison of a restricted (nested) OLS fit against the
/// full fit.
///
/// Returns `(lr_stat, p_value, df_diff)` where
/// `lr_stat = −2 (llf_restricted − llf_full)` is chi-squared distributed with
/// `df_diff = df_resid_restricted − df_resid_full` degrees of freedom. The
/// restricted model must be nested in `full`. Mirrors the reference
/// `compare_lr_test`.
pub fn compare_lr_test(full: &LinearResults, restricted: &LinearResults) -> (f64, f64, f64) {
    let lrdf = restricted.df_resid - full.df_resid;
    let lrstat = -2.0 * (restricted.llf - full.llf);
    let lr_pvalue = chi2_sf(lrstat, lrdf);
    (lrstat, lr_pvalue, lrdf)
}

/// F-test comparison of a restricted (nested) OLS fit against the full fit.
///
/// Returns `(f_value, p_value, df_diff)` where
/// `f_value = (ssr_restricted − ssr_full) / df_diff / ssr_full · df_resid_full`
/// is `F(df_diff, df_resid_full)` distributed and
/// `df_diff = df_resid_restricted − df_resid_full`. The restricted model must be
/// nested in `full`. Mirrors the reference `compare_f_test`.
pub fn compare_f_test(full: &LinearResults, restricted: &LinearResults) -> (f64, f64, f64) {
    let df_full = full.df_resid;
    let df_diff = restricted.df_resid - df_full;
    let f_value = (restricted.ssr - full.ssr) / df_diff / full.ssr * df_full;
    let p_value = f_sf(f_value, df_diff, df_full);
    (f_value, p_value, df_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn small_ols() -> (Array1<f64>, Array2<f64>) {
        let x = array![
            [1.0, 0.2, -0.5],
            [1.0, -0.1, 0.3],
            [1.0, 0.4, 0.1],
            [1.0, -0.3, -0.2],
            [1.0, 0.5, 0.6],
            [1.0, -0.2, -0.4],
            [1.0, 0.1, 0.2],
            [1.0, 0.3, -0.1],
            [1.0, -0.4, 0.5],
            [1.0, 0.0, -0.3],
            [1.0, 0.25, 0.15],
            [1.0, -0.35, 0.05],
        ];
        let y = array![0.9, 1.1, 1.4, 0.7, 1.8, 0.6, 1.2, 1.0, 0.8, 1.05, 1.3, 0.95];
        (y, x)
    }

    #[test]
    fn lagmat_both_shape_and_values() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let m = lagmat_both(&x, 2);
        assert_eq!(m.dim(), (3, 2));
        // row 0 corresponds to current index 2: [x1, x0] = [2, 1]
        assert_eq!(m[[0, 0]], 2.0);
        assert_eq!(m[[0, 1]], 1.0);
        assert_eq!(m[[2, 0]], 4.0);
        assert_eq!(m[[2, 1]], 3.0);
    }

    #[test]
    fn acorr_lm_runs_and_bounds() {
        let (y, x) = small_ols();
        let res = LinearModel::ols(y, x).unwrap().fit().unwrap();
        let (lm, p, f, fp) = acorr_lm(&res.resid, Some(2), 0).unwrap();
        assert!(lm >= 0.0);
        assert!((0.0..=1.0).contains(&p));
        assert!(f >= 0.0);
        assert!((0.0..=1.0).contains(&fp));
    }

    #[test]
    fn het_arch_equals_acorr_lm_of_squares() {
        let (y, x) = small_ols();
        let res = LinearModel::ols(y, x).unwrap().fit().unwrap();
        let a = het_arch(&res.resid, Some(2), 0).unwrap();
        let b = acorr_lm(&res.resid.mapv(|v| v * v), Some(2), 0).unwrap();
        assert!((a.0 - b.0).abs() < 1e-12);
        assert!((a.2 - b.2).abs() < 1e-12);
    }

    #[test]
    fn bg_and_reset_run() {
        let (y, x) = small_ols();
        let res = LinearModel::ols(y.clone(), x.clone())
            .unwrap()
            .fit()
            .unwrap();
        let (lm, p, _f, fp) = acorr_breusch_godfrey(&res.resid, &x, Some(2)).unwrap();
        assert!(lm >= 0.0 && (0.0..=1.0).contains(&p) && (0.0..=1.0).contains(&fp));
        let (stat, pv) = linear_reset(&y, &x, 3, ResetAug::Fitted, false).unwrap();
        assert!(stat >= 0.0 && (0.0..=1.0).contains(&pv));
    }

    #[test]
    fn compare_tests_nested() {
        let (y, x) = small_ols();
        let full = LinearModel::ols(y.clone(), x.clone())
            .unwrap()
            .fit()
            .unwrap();
        // Restricted: drop last column.
        let xr = x.slice(ndarray::s![.., 0..2]).to_owned();
        let restr = LinearModel::ols(y, xr).unwrap().fit().unwrap();
        let (lr, lp, ldf) = compare_lr_test(&full, &restr);
        let (f, fp, fdf) = compare_f_test(&full, &restr);
        assert_eq!(ldf, 1.0);
        assert_eq!(fdf, 1.0);
        assert!(lr >= 0.0 && (0.0..=1.0).contains(&lp));
        assert!(f >= 0.0 && (0.0..=1.0).contains(&fp));
    }
}
