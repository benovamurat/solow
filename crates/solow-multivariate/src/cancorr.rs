//! Canonical correlation analysis (CanCorr).
//!
//! Mirrors the reference's `multivariate.cancorr.CanCorr`. Given two sets of
//! variables `endog` (`nobs × k_endog`) and `exog` (`nobs × k_exog`), the
//! canonical correlations are the singular values of `Uxᵀ Uy`, where `Ux` and
//! `Uy` are the left singular vectors of the column-centred `exog` and `endog`.
//! There are `k = min(k_endog, k_exog)` of them, each clamped to `[0, 1]`.
//!
//! [`CanCorr::corr_test`] reports, for each canonical correlation, the
//! approximate Wilks'-lambda F-test of the hypothesis that this and all smaller
//! canonical correlations are zero.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::f_sf;
use solow_linalg::svd;

/// A fitted canonical correlation analysis.
#[derive(Clone, Debug)]
pub struct CanCorr {
    /// Canonical correlations, descending, length `min(k_endog, k_exog)`.
    pub cancorr: Array1<f64>,
    k_endog: usize,
    k_exog: usize,
    nobs: usize,
}

impl CanCorr {
    /// Fit a canonical correlation analysis of `endog` against `exog`.
    ///
    /// Returns an error if either set of variables is collinear (a singular
    /// value below tolerance).
    pub fn new(endog: &Array2<f64>, exog: &Array2<f64>) -> Result<Self> {
        let (nobs, k_endog) = endog.dim();
        let (nobs_x, k_exog) = exog.dim();
        if nobs != nobs_x {
            return Err(Error::Shape(
                "endog and exog must have the same number of rows".into(),
            ));
        }
        let tolerance = 1e-8;

        let x = demean(exog);
        let y = demean(endog);

        let (ux, sx, _vx) = svd(&x)?;
        if sx.iter().filter(|&&v| v > tolerance).count() < sx.len() {
            return Err(Error::Singular("exog is collinear".into()));
        }
        let (uy, sy, _vy) = svd(&y)?;
        if sy.iter().filter(|&&v| v > tolerance).count() < sy.len() {
            return Err(Error::Singular("endog is collinear".into()));
        }

        // SVD of Uxᵀ Uy; the singular values are the canonical correlations.
        let m = ux.t().dot(&uy);
        let (_u, s, _v) = svd(&m)?;
        let k = k_endog.min(k_exog);
        let cancorr = Array1::from_iter((0..k).map(|i| s[i].clamp(0.0, 1.0)));

        Ok(CanCorr {
            cancorr,
            k_endog,
            k_exog,
            nobs,
        })
    }

    /// Approximate Wilks'-lambda F-test for each canonical correlation.
    ///
    /// Returns one [`CanCorrTestRow`] per canonical correlation, ordered from
    /// the first (largest) to the last, matching the reference table.
    pub fn corr_test(&self) -> Vec<CanCorrTestRow> {
        let k_yvar = self.k_endog as f64;
        let k_xvar = self.k_exog as f64;
        let nobs = self.nobs as f64;
        let eigenvals: Vec<f64> = self.cancorr.iter().map(|&c| c * c).collect();
        let n = eigenvals.len();

        // Accumulate the Wilks product from the smallest canonical correlation
        // upward (matching the reference's reverse loop), then re-order.
        let mut rows_rev: Vec<CanCorrTestRow> = Vec::with_capacity(n);
        let mut prod = 1.0;
        for i in (0..n).rev() {
            prod *= 1.0 - eigenvals[i];
            let p = k_yvar - i as f64;
            let q = k_xvar - i as f64;
            let r = (nobs - k_yvar - 1.0) - (p - q + 1.0) / 2.0;
            let u = (p * q - 2.0) / 4.0;
            let df1 = p * q;
            let t = if p * p + q * q - 5.0 > 0.0 {
                (((p * q).powi(2) - 4.0) / (p * p + q * q - 5.0)).sqrt()
            } else {
                1.0
            };
            let df2 = r * t - 2.0 * u;
            let lmd = prod.powf(1.0 / t);
            let f = (1.0 - lmd) / lmd * df2 / df1;
            let pr_f = f_sf(f, df1, df2);
            rows_rev.push(CanCorrTestRow {
                index: i,
                cancorr: self.cancorr[i],
                wilks_lambda: prod,
                num_df: df1,
                den_df: df2,
                f_value: f,
                pr_f,
            });
        }
        // rows_rev is ordered i = n-1, n-2, ..., 0. The reference returns them
        // in ascending index order (0, 1, ..., n-1).
        rows_rev.reverse();
        rows_rev
    }
}

/// Column-centred copy of `m` (subtract each column mean).
fn demean(m: &Array2<f64>) -> Array2<f64> {
    let (nobs, nvar) = m.dim();
    let mut out = m.clone();
    for j in 0..nvar {
        let mean = m.column(j).sum() / nobs as f64;
        for i in 0..nobs {
            out[[i, j]] -= mean;
        }
    }
    out
}

/// One row of the canonical-correlation approximate F-test table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanCorrTestRow {
    /// Index of the canonical correlation (0 = largest).
    pub index: usize,
    /// The canonical correlation.
    pub cancorr: f64,
    /// Wilks' lambda for this and all smaller canonical correlations.
    pub wilks_lambda: f64,
    /// Numerator degrees of freedom of the F-approximation.
    pub num_df: f64,
    /// Denominator degrees of freedom of the F-approximation.
    pub den_df: f64,
    /// Approximate F-value.
    pub f_value: f64,
    /// Upper-tail p-value.
    pub pr_f: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cancorr_in_unit_interval_descending() {
        let exog = array![
            [1.0, 0.2],
            [2.0, -0.1],
            [3.0, 0.4],
            [4.0, 0.9],
            [2.5, -0.3],
            [1.7, 0.6],
        ];
        let endog = array![
            [0.9, 1.1],
            [1.8, 0.7],
            [3.2, 0.2],
            [3.9, -0.4],
            [2.4, 0.5],
            [1.6, 0.8],
        ];
        let cc = CanCorr::new(&endog, &exog).unwrap();
        assert_eq!(cc.cancorr.len(), 2);
        for &c in cc.cancorr.iter() {
            assert!((0.0..=1.0).contains(&c));
        }
        assert!(cc.cancorr[0] >= cc.cancorr[1]);
        let rows = cc.corr_test();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].index, 0);
        assert_eq!(rows[1].index, 1);
    }
}
