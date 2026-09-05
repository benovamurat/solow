//! Shrinkage covariance estimators — [`ShrunkCovariance`], [`LedoitWolf`], [`Oas`].

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::empirical::EmpiricalCovariance;

/// Fit `(1 − ρ) · S + ρ · (tr(S)/d) · I` at a caller-supplied ρ.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct ShrunkCovariance {
    /// Underlying sample-covariance fit.
    pub base: EmpiricalCovariance,
    /// Shrunk covariance matrix.
    pub covariance: Array2<f64>,
    /// Shrinkage intensity used.
    pub shrinkage: f64,
}

impl ShrunkCovariance {
    /// Fit at a given ρ ∈ [0, 1].
    pub fn fit(x: ArrayView2<'_, f64>, shrinkage: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&shrinkage) {
            return Err(Error::Value(format!(
                "ShrunkCovariance::fit: shrinkage must be in [0, 1] (got {shrinkage})"
            )));
        }
        let base = EmpiricalCovariance::fit(x)?;
        let cov = shrunk_to_identity(&base.covariance, shrinkage);
        Ok(Self {
            base,
            covariance: cov,
            shrinkage,
        })
    }
}

/// Ledoit-Wolf (2004) automatic-shrinkage estimator.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LedoitWolf {
    /// Underlying sample-covariance fit.
    pub base: EmpiricalCovariance,
    /// Shrunk covariance matrix.
    pub covariance: Array2<f64>,
    /// Automatic shrinkage ρ̂.
    pub shrinkage: f64,
}

impl LedoitWolf {
    /// Fit with automatic shrinkage.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        let base = EmpiricalCovariance::fit(x)?;
        let n = x.nrows();
        let d = x.ncols();
        // Sample estimates.
        let s = &base.covariance;
        // Target μ · I with μ = tr(S) / d.
        let mu: f64 = (0..d).map(|i| s[[i, i]]).sum::<f64>() / d as f64;
        // ‖S − μ·I‖² (squared Frobenius).
        let mut d_norm2 = 0.0_f64;
        for i in 0..d {
            for j in 0..d {
                let target = if i == j { mu } else { 0.0 };
                let dd = s[[i, j]] - target;
                d_norm2 += dd * dd;
            }
        }
        // b̄² — the average squared deviation of the per-sample outer
        // products from the pooled S.
        let mut b_sum = 0.0_f64;
        let mean = &base.location;
        for k in 0..n {
            let mut outer = Array2::<f64>::zeros((d, d));
            for i in 0..d {
                for j in 0..d {
                    outer[[i, j]] = (x[[k, i]] - mean[i]) * (x[[k, j]] - mean[j]);
                }
            }
            let mut term = 0.0_f64;
            for i in 0..d {
                for j in 0..d {
                    let e = outer[[i, j]] - s[[i, j]];
                    term += e * e;
                }
            }
            b_sum += term;
        }
        let b_bar2 = b_sum / (n as f64).powi(2);
        // Shrinkage.
        let shrinkage = (b_bar2 / d_norm2.max(1e-300)).clamp(0.0, 1.0);
        let cov = shrunk_to_identity(s, shrinkage);
        Ok(Self {
            base,
            covariance: cov,
            shrinkage,
        })
    }
}

/// OAS shrinkage (Chen-Wiesel-Eldar-Hero 2010).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Oas {
    /// Underlying sample-covariance fit.
    pub base: EmpiricalCovariance,
    /// Shrunk covariance matrix.
    pub covariance: Array2<f64>,
    /// Automatic shrinkage ρ̂.
    pub shrinkage: f64,
}

impl Oas {
    /// Fit with OAS shrinkage.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        let base = EmpiricalCovariance::fit(x)?;
        let n = x.nrows() as f64;
        let d = x.ncols() as f64;
        let s = &base.covariance;
        let mu: f64 = (0..x.ncols()).map(|i| s[[i, i]]).sum::<f64>() / d;
        let mut tr_s2 = 0.0_f64;
        for i in 0..x.ncols() {
            for j in 0..x.ncols() {
                tr_s2 += s[[i, j]] * s[[i, j]];
            }
        }
        let tr_s = mu * d;
        let numer = (1.0 - 2.0 / d) * tr_s2 + tr_s * tr_s;
        let denom = (n + 1.0 - 2.0 / d) * (tr_s2 - tr_s * tr_s / d);
        let shrinkage = (numer / denom.max(1e-300)).clamp(0.0, 1.0);
        let cov = shrunk_to_identity(s, shrinkage);
        Ok(Self {
            base,
            covariance: cov,
            shrinkage,
        })
    }
}

fn shrunk_to_identity(s: &Array2<f64>, rho: f64) -> Array2<f64> {
    let d = s.nrows();
    let mu: f64 = (0..d).map(|i| s[[i, i]]).sum::<f64>() / d as f64;
    let mut out = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let target = if i == j { mu } else { 0.0 };
            out[[i, j]] = (1.0 - rho) * s[[i, j]] + rho * target;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn shrunk_at_rho_zero_reduces_to_sample() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let s = ShrunkCovariance::fit(x.view(), 0.0).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (s.covariance[[i, j]] - s.base.covariance[[i, j]]).abs() < 1e-12
                );
            }
        }
    }

    #[test]
    fn ledoit_wolf_returns_a_valid_shrinkage() {
        let x = array![
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0],
            [9.0, 10.0, 11.0]
        ];
        let lw = LedoitWolf::fit(x.view()).unwrap();
        assert!((0.0..=1.0).contains(&lw.shrinkage));
    }

    #[test]
    fn oas_returns_a_valid_shrinkage() {
        let x = array![
            [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]
        ];
        let oas = Oas::fit(x.view()).unwrap();
        assert!((0.0..=1.0).contains(&oas.shrinkage));
    }
}
