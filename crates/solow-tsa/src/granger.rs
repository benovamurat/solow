//! Granger causality tests for a bivariate series.
//!
//! Mirrors the reference `grangercausalitytests(x, maxlag)` with `addconst =
//! true`. For each lag the function fits a *restricted* model (the first
//! series regressed on its own lags plus a constant) and an *unrestricted*
//! ("joint") model (additionally including lags of the second series), then
//! reports the three sum-of-squared-residual based tests.

use ndarray::Array2;
use solow_core::error::{Error, Result};
use solow_distributions::{chi2_sf, f_sf};
use solow_regression::LinearModel;

use crate::tsatools::{lagmat, Original, Trim};

/// The three Granger-causality test statistics produced for a single lag.
#[derive(Debug, Clone)]
pub struct GrangerLagResult {
    /// The lag order this result corresponds to.
    pub lag: usize,
    /// `ssr`-based F test: `(F, pvalue, df_num, df_den)`.
    pub ssr_ftest: (f64, f64, usize, usize),
    /// `ssr`-based chi-squared test: `(chi2, pvalue, df)`.
    pub ssr_chi2test: (f64, f64, usize),
    /// Likelihood-ratio test: `(chi2, pvalue, df)`.
    pub lrtest: (f64, f64, usize),
}

/// Granger causality tests for lags `1..=maxlag`.
///
/// `data` is an `n × 2` matrix whose first column is the response (`y`) whose
/// Granger-causation by the second column (`x`) is tested. Returns one
/// [`GrangerLagResult`] per lag.
pub fn grangercausalitytests(data: &Array2<f64>, maxlag: usize) -> Result<Vec<GrangerLagResult>> {
    let (nobs, ncols) = data.dim();
    if ncols != 2 {
        return Err(Error::Shape("data must have exactly two columns".into()));
    }
    if maxlag == 0 {
        return Err(Error::Value("maxlag must be a positive integer".into()));
    }
    if !data.iter().all(|v| v.is_finite()) {
        return Err(Error::Value("x contains NaN or inf values.".into()));
    }
    // addconst == true so the +1 is for the constant.
    if nobs <= 3 * maxlag + 1 {
        return Err(Error::Value("Insufficient observations.".into()));
    }

    let mut out = Vec::with_capacity(maxlag);
    for mlg in 1..=maxlag {
        out.push(granger_one_lag(data, mlg)?);
    }
    Ok(out)
}

/// Build `lagmat2ds(x, mxlg, trim="both", dropex=1)` for a two-column input.
///
/// Column 0 is the contemporaneous `y`; columns `1..=mxlg` are lags 1..mxlg of
/// `y`; columns `mxlg+1..=2*mxlg` are lags 1..mxlg of the second series.
fn lagmat2ds(data: &Array2<f64>, mxlg: usize) -> Result<Array2<f64>> {
    let y = data.column(0).to_owned();
    let x = data.column(1).to_owned();
    let ycol = y.view().insert_axis(ndarray::Axis(1)).to_owned();
    let xcol = x.view().insert_axis(ndarray::Axis(1)).to_owned();
    // lagmat(., maxlag, trim="both", original="in") gives columns
    // [lag0, lag1, ..., lagmaxlag] after trimming both ends.
    let (ylags, _) = lagmat(&ycol, mxlg, Trim::Both, Original::In)?;
    let (xlags, _) = lagmat(&xcol, mxlg, Trim::Both, Original::In)?;
    let nrows = ylags.nrows();
    // Take y columns 0..=mxlg, then x columns 1..=mxlg (dropex = 1).
    let total = (mxlg + 1) + mxlg;
    let mut dta = Array2::<f64>::zeros((nrows, total));
    for i in 0..nrows {
        for j in 0..=mxlg {
            dta[[i, j]] = ylags[[i, j]];
        }
        for j in 1..=mxlg {
            dta[[i, mxlg + j]] = xlags[[i, j]];
        }
    }
    Ok(dta)
}

fn add_constant_append(x: &Array2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut out = Array2::<f64>::zeros((n, k + 1));
    for i in 0..n {
        for j in 0..k {
            out[[i, j]] = x[[i, j]];
        }
        out[[i, k]] = 1.0;
    }
    out
}

fn granger_one_lag(data: &Array2<f64>, mxlg: usize) -> Result<GrangerLagResult> {
    let dta = lagmat2ds(data, mxlg)?;
    let nrows = dta.nrows();
    let y = dta.column(0).to_owned();

    // Restricted ("down"): own lags only -> columns 1..=mxlg, plus constant.
    let mut own = Array2::<f64>::zeros((nrows, mxlg));
    for i in 0..nrows {
        for j in 0..mxlg {
            own[[i, j]] = dta[[i, 1 + j]];
        }
    }
    let dtaown = add_constant_append(&own);

    // Unrestricted ("joint"): columns 1.. (own lags + other-series lags) + const.
    let kjoint = dta.ncols() - 1;
    let mut joint = Array2::<f64>::zeros((nrows, kjoint));
    for i in 0..nrows {
        for j in 0..kjoint {
            joint[[i, j]] = dta[[i, 1 + j]];
        }
    }
    let dtajoint = add_constant_append(&joint);

    let res_down = LinearModel::ols(y.clone(), dtaown)?.fit()?;
    let res_joint = LinearModel::ols(y, dtajoint)?.fit()?;

    let ssr_down = res_down.ssr;
    let ssr_joint = res_joint.ssr;
    let df_resid_joint = res_joint.df_resid;
    let nobs_down = res_down.nobs;

    // ssr based F test.
    let fgc1 = (ssr_down - ssr_joint) / ssr_joint / mxlg as f64 * df_resid_joint;
    let f_p = f_sf(fgc1, mxlg as f64, df_resid_joint);
    let ssr_ftest = (fgc1, f_p, mxlg, df_resid_joint as usize);

    // ssr based chi2 test.
    let fgc2 = nobs_down * (ssr_down - ssr_joint) / ssr_joint;
    let chi2_p = chi2_sf(fgc2, mxlg as f64);
    let ssr_chi2test = (fgc2, chi2_p, mxlg);

    // Likelihood-ratio test.
    let lr = -2.0 * (res_down.llf - res_joint.llf);
    let lr_p = chi2_sf(lr, mxlg as f64);
    let lrtest = (lr, lr_p, mxlg);

    Ok(GrangerLagResult {
        lag: mxlg,
        ssr_ftest,
        ssr_chi2test,
        lrtest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn toy() -> Array2<f64> {
        // y depends on lagged x.
        let n = 60;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut s = 1.0_f64;
        for xt in x.iter_mut() {
            s = (s * 1103515245.0 + 12345.0) % 2147483648.0;
            *xt = (s / 2147483648.0) - 0.5;
        }
        for t in 2..n {
            y[t] = 0.4 * y[t - 1] + 0.5 * x[t - 1] - 0.3 * x[t - 2];
        }
        for t in 0..n {
            data[[t, 0]] = y[t];
            data[[t, 1]] = x[t];
        }
        data
    }

    #[test]
    fn produces_one_result_per_lag() {
        let d = toy();
        let r = grangercausalitytests(&d, 3).unwrap();
        assert_eq!(r.len(), 3);
        for (i, lag) in r.iter().enumerate() {
            assert_eq!(lag.lag, i + 1);
            // p-values are valid probabilities.
            assert!((0.0..=1.0).contains(&lag.ssr_ftest.1));
            assert!((0.0..=1.0).contains(&lag.ssr_chi2test.1));
            assert!((0.0..=1.0).contains(&lag.lrtest.1));
            // df_num equals the lag.
            assert_eq!(lag.ssr_ftest.2, i + 1);
        }
    }

    #[test]
    fn rejects_wrong_columns() {
        let d = Array2::<f64>::zeros((20, 3));
        assert!(grangercausalitytests(&d, 2).is_err());
    }

    #[test]
    fn rejects_too_many_lags() {
        let d = Array2::<f64>::zeros((7, 2));
        assert!(grangercausalitytests(&d, 3).is_err());
    }
}
