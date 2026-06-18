//! Robust (sandwich) covariance estimators for the linear model.
//!
//! After an OLS fit, the textbook coefficient covariance `scale · (XᵀX)⁻¹`
//! relies on spherical errors. When that assumption fails — heteroskedastic,
//! autocorrelated, or clustered errors — a *sandwich* estimator
//!
//! ```text
//!     V = (XᵀX)⁻¹ · S · (XᵀX)⁻¹
//! ```
//!
//! replaces the inner "meat" `S = Xᵀ diag(u²) X` (for the simplest case) with a
//! variant tailored to the error structure. This module provides:
//!
//! * **HC0–HC3** — heteroskedasticity-consistent covariances. HC0 is the raw
//!   White estimator; HC1 applies the `n/(n−k)` degrees-of-freedom scaling;
//!   HC2 and HC3 reweight each squared residual by the leverage `h_ii`.
//! * **HAC** (Newey–West) — heteroskedasticity- and autocorrelation-consistent,
//!   built from the score series `xᵢ uᵢ` with a Bartlett kernel over `maxlags`
//!   lags, optionally with the `n/(n−k)` small-sample correction.
//! * **cluster** — one-way clustered covariance summing the group score totals,
//!   optionally with the reference `G/(G−1) · (n−1)/(n−k)` correction.
//!
//! The free function [`robust_cov`] takes the design `X`, residuals `u`, the
//! normalized covariance `(XᵀX)⁻¹`, and `df_resid`, and is what the convenience
//! methods on `LinearResults` delegate to. All formulas reproduce the reference
//! `get_robustcov_results` to closed-form precision.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

/// The kind of robust covariance to compute.
#[derive(Clone, Debug, PartialEq)]
pub enum CovType {
    /// White heteroskedasticity-consistent covariance, no correction.
    Hc0,
    /// HC0 scaled by `n / (n − k)`.
    Hc1,
    /// HC2: each squared residual divided by `1 − h_ii`.
    Hc2,
    /// HC3: each residual divided by `1 − h_ii`, then squared.
    Hc3,
    /// Newey–West HAC with a Bartlett kernel.
    Hac {
        /// Highest lag included in the kernel window (the zero lag is implicit).
        maxlags: usize,
        /// Apply the `n / (n − k)` small-sample correction.
        use_correction: bool,
    },
    /// One-way clustered covariance.
    Cluster {
        /// Integer group id per observation (length `nobs`).
        groups: Vec<i64>,
        /// Apply the `G/(G−1) · (n−1)/(n−k)` small-sample correction.
        use_correction: bool,
    },
}

/// Leverages `h_ii = diag(X (XᵀX)⁻¹ Xᵀ)` from the design and normalized covariance.
fn leverages(x: &Array2<f64>, normalized_cov_params: &Array2<f64>) -> Array1<f64> {
    // h_ii = xᵢ · (XᵀX)⁻¹ · xᵢ, computed row-wise to avoid the full hat matrix.
    let xc = x.dot(normalized_cov_params); // n × k
    let n = x.nrows();
    let k = x.ncols();
    let mut h = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..k {
            acc += xc[[i, j]] * x[[i, j]];
        }
        h[i] = acc;
    }
    h
}

/// Sandwich `(XᵀX)⁻¹ · (Xᵀ diag(scale) X) · (XᵀX)⁻¹`.
///
/// This mirrors the reference `_HCCM`, where `pinv_wexog = (XᵀX)⁻¹ Xᵀ` gives
/// `pinv · diag(scale) · pinvᵀ`. We form the inner `Xᵀ diag(scale) X` explicitly.
fn hccm_diag(
    x: &Array2<f64>,
    scale: &Array1<f64>,
    normalized_cov_params: &Array2<f64>,
) -> Array2<f64> {
    let k = x.ncols();
    let n = x.nrows();
    // meat = Xᵀ diag(scale) X
    let mut meat = Array2::<f64>::zeros((k, k));
    for i in 0..n {
        let s = scale[i];
        if s == 0.0 {
            continue;
        }
        let row = x.row(i);
        for a in 0..k {
            let xa = row[a] * s;
            for b in 0..k {
                meat[[a, b]] += xa * row[b];
            }
        }
    }
    sandwich(normalized_cov_params, &meat)
}

/// `bread · meat · bread` (bread is symmetric here, `(XᵀX)⁻¹`).
fn sandwich(bread: &Array2<f64>, meat: &Array2<f64>) -> Array2<f64> {
    bread.dot(meat).dot(&bread.t())
}

/// Bartlett kernel weights `w_l = 1 − l/(L+1)` for `l = 0..=maxlags`.
fn bartlett_weights(maxlags: usize) -> Vec<f64> {
    (0..=maxlags)
        .map(|l| 1.0 - (l as f64) / ((maxlags + 1) as f64))
        .collect()
}

/// Score series `xᵢ uᵢ` (`n × k`): each design row scaled by its residual.
fn scores(x: &Array2<f64>, resid: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let mut xu = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let u = resid[i];
        for j in 0..k {
            xu[[i, j]] = x[[i, j]] * u;
        }
    }
    xu
}

/// HAC "meat" `S = Γ₀ + Σ_{l=1}^{L} w_l (Γ_l + Γ_lᵀ)` with `Γ_l = Σ_i xu_i xu_{i−l}ᵀ`.
fn s_hac(xu: &Array2<f64>, maxlags: usize) -> Array2<f64> {
    let n = xu.nrows();
    let k = xu.ncols();
    let weights = bartlett_weights(maxlags);
    // Γ₀ = xuᵀ xu (weights[0] == 1).
    let mut s = xu.t().dot(xu);
    for (lag, &w) in weights.iter().enumerate().take(maxlags + 1).skip(1) {
        if lag >= n {
            break;
        }
        // gamma_l = Σ_i xu_{i} xu_{i-lag}ᵀ  == xu[lag:]ᵀ · xu[:-lag]
        let mut gamma = Array2::<f64>::zeros((k, k));
        for i in lag..n {
            let hi = xu.row(i);
            let lo = xu.row(i - lag);
            for a in 0..k {
                let v = hi[a];
                for b in 0..k {
                    gamma[[a, b]] += v * lo[b];
                }
            }
        }
        for a in 0..k {
            for b in 0..k {
                s[[a, b]] += w * (gamma[[a, b]] + gamma[[b, a]]);
            }
        }
    }
    s
}

/// Cluster "meat" `S = Σ_g g_g g_gᵀ` where `g_g = Σ_{i∈g} xu_i` is the group score sum.
fn s_cluster(xu: &Array2<f64>, groups: &[i64]) -> (Array2<f64>, usize) {
    let k = xu.ncols();
    // Stable enumeration of distinct group ids (order does not affect the sum).
    let mut uniq: Vec<i64> = groups.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let n_groups = uniq.len();
    let index = |g: i64| uniq.binary_search(&g).unwrap();

    // Group sums of scores: n_groups × k.
    let mut gsum = Array2::<f64>::zeros((n_groups, k));
    for (i, &g) in groups.iter().enumerate() {
        let gi = index(g);
        for j in 0..k {
            gsum[[gi, j]] += xu[[i, j]];
        }
    }
    // S = gsumᵀ gsum.
    let s = gsum.t().dot(&gsum);
    (s, n_groups)
}

/// Compute a robust coefficient covariance matrix.
///
/// * `x` — the design matrix `X` (`n × k`). For OLS this is `exog`; for WLS/GLS
///   it should be the *whitened* design `wexog`.
/// * `resid` — the residuals `u` (`n`). For OLS this is `resid`; for WLS/GLS the
///   whitened residuals `wresid`.
/// * `normalized_cov_params` — `(XᵀX)⁻¹` (the model's `normalized_cov_params`).
/// * `df_resid` — residual degrees of freedom `n − k` used by HC1/HAC scaling.
///
/// Returns the `k × k` sandwich covariance. The standard errors are the square
/// roots of the diagonal (see [`bse_from_cov`]).
pub fn robust_cov(
    x: &Array2<f64>,
    resid: &Array1<f64>,
    normalized_cov_params: &Array2<f64>,
    df_resid: f64,
    cov_type: &CovType,
) -> Result<Array2<f64>> {
    let n = x.nrows();
    let k = x.ncols();
    if resid.len() != n {
        return Err(Error::Shape(format!(
            "resid length {} != design rows {}",
            resid.len(),
            n
        )));
    }
    if normalized_cov_params.dim() != (k, k) {
        return Err(Error::Shape(format!(
            "normalized_cov_params must be {k}×{k}"
        )));
    }
    let nobs = n as f64;

    match cov_type {
        CovType::Hc0 | CovType::Hc1 => {
            let mut scale: Array1<f64> = resid.iter().map(|&u| u * u).collect();
            if matches!(cov_type, CovType::Hc1) {
                let f = nobs / df_resid;
                scale.mapv_inplace(|v| v * f);
            }
            Ok(hccm_diag(x, &scale, normalized_cov_params))
        }
        CovType::Hc2 => {
            let h = leverages(x, normalized_cov_params);
            let scale: Array1<f64> = resid
                .iter()
                .zip(h.iter())
                .map(|(&u, &hi)| u * u / (1.0 - hi))
                .collect();
            Ok(hccm_diag(x, &scale, normalized_cov_params))
        }
        CovType::Hc3 => {
            let h = leverages(x, normalized_cov_params);
            let scale: Array1<f64> = resid
                .iter()
                .zip(h.iter())
                .map(|(&u, &hi)| {
                    let t = u / (1.0 - hi);
                    t * t
                })
                .collect();
            Ok(hccm_diag(x, &scale, normalized_cov_params))
        }
        CovType::Hac {
            maxlags,
            use_correction,
        } => {
            let xu = scores(x, resid);
            let s = s_hac(&xu, *maxlags);
            let mut cov = sandwich(normalized_cov_params, &s);
            if *use_correction {
                let f = nobs / df_resid;
                cov.mapv_inplace(|v| v * f);
            }
            Ok(cov)
        }
        CovType::Cluster {
            groups,
            use_correction,
        } => {
            if groups.len() != n {
                return Err(Error::Shape(format!(
                    "groups length {} != design rows {}",
                    groups.len(),
                    n
                )));
            }
            let xu = scores(x, resid);
            let (s, n_groups) = s_cluster(&xu, groups);
            let mut cov = sandwich(normalized_cov_params, &s);
            if *use_correction {
                let g = n_groups as f64;
                let f = (g / (g - 1.0)) * ((nobs - 1.0) / (nobs - k as f64));
                cov.mapv_inplace(|v| v * f);
            }
            Ok(cov)
        }
    }
}

/// Standard errors `√diag(cov)` from a covariance matrix.
pub fn bse_from_cov(cov: &Array2<f64>) -> Array1<f64> {
    let k = cov.nrows();
    let mut bse = Array1::<f64>::zeros(k);
    for i in 0..k {
        bse[i] = cov[[i, i]].sqrt();
    }
    bse
}
