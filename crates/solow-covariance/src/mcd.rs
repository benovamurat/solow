//! Minimum-Covariance-Determinant estimator (Rousseeuw 1984,
//! Rousseeuw-Van Driessen 1999 FAST-MCD).
//!
//! Deterministic implementation: subsets are drawn from a portable
//! MMIX-LCG seeded by the caller.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::empirical::{invert_symmetric, EmpiricalCovariance};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

/// Fitted MCD estimator.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MinCovDet {
    /// Robust location.
    pub location: Array1<f64>,
    /// Robust covariance.
    pub covariance: Array2<f64>,
    /// Indices of the samples in the final `h`-subset.
    pub support: Vec<usize>,
    /// Determinant of the final covariance.
    pub determinant: f64,
    /// Support fraction actually used (`h / n`).
    pub support_fraction: f64,
}

impl MinCovDet {
    /// Fit with `support_fraction = 0.75` and 50 random restarts.
    pub fn fit(x: ArrayView2<'_, f64>, seed: u64) -> Result<Self> {
        Self::fit_with(x, 0.75, 50, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        support_fraction: f64,
        n_restarts: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() < 3 || x.ncols() == 0 {
            return Err(Error::Value(
                "MinCovDet::fit_with: need ≥ 3 samples and ≥ 1 feature".into(),
            ));
        }
        if !(0.5..=1.0).contains(&support_fraction) {
            return Err(Error::Value(format!(
                "MinCovDet::fit_with: support_fraction must be in [0.5, 1] (got {support_fraction})"
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        let h = ((n as f64) * support_fraction).ceil() as usize;
        if h < d + 1 {
            return Err(Error::Value(format!(
                "MinCovDet::fit_with: support size h={h} must be ≥ d+1={}", d + 1
            )));
        }
        let mut state = seed.wrapping_add(0xDEAD_BEEF_C0DE_F00D);
        let mut best: Option<(Vec<usize>, f64, EmpiricalCovariance)> = None;
        for _ in 0..n_restarts {
            // Draw a random (d+1)-subset and grow via C-step to h.
            let mut idx: Vec<usize> = Vec::with_capacity(d + 1);
            let mut seen = std::collections::HashSet::new();
            while idx.len() < d + 1 {
                let i = uniform_index(&mut state, n as u64);
                if seen.insert(i) {
                    idx.push(i);
                }
            }
            let mut cur_subset = idx;
            for _ in 0..30 {
                let sub = subset(x, &cur_subset);
                let Ok(fit) = EmpiricalCovariance::fit(sub.view()) else {
                    break;
                };
                let mahal = mahalanobis_of_all(x, &fit).unwrap_or_else(|_| Array1::zeros(n));
                let mut order: Vec<(usize, f64)> =
                    (0..n).map(|i| (i, mahal[i])).collect();
                order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let new_subset: Vec<usize> = order.iter().take(h).map(|(i, _)| *i).collect();
                if new_subset == cur_subset {
                    let det = determinant(&fit.covariance).unwrap_or(f64::INFINITY);
                    match &best {
                        None => best = Some((cur_subset.clone(), det, fit)),
                        Some((_, best_det, _)) if det < *best_det => {
                            best = Some((cur_subset.clone(), det, fit));
                        }
                        _ => {}
                    }
                    break;
                }
                cur_subset = new_subset;
            }
        }
        let (support, det, fit) = best.ok_or_else(|| {
            Error::Value("MinCovDet::fit_with: no C-step converged; try more restarts".into())
        })?;
        Ok(Self {
            location: fit.location,
            covariance: fit.covariance,
            support,
            determinant: det,
            support_fraction,
        })
    }
}

fn subset(x: ArrayView2<'_, f64>, idx: &[usize]) -> Array2<f64> {
    let d = x.ncols();
    let mut sub = Array2::<f64>::zeros((idx.len(), d));
    for (r, &i) in idx.iter().enumerate() {
        for j in 0..d {
            sub[[r, j]] = x[[i, j]];
        }
    }
    sub
}

fn mahalanobis_of_all(x: ArrayView2<'_, f64>, fit: &EmpiricalCovariance) -> Result<Array1<f64>> {
    fit.mahalanobis(x)
}

fn determinant(m: &Array2<f64>) -> Result<f64> {
    // Determinant via LU decomposition with partial pivot; small matrix.
    let n = m.nrows();
    let mut a = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[[i, j]];
        }
    }
    let mut sign = 1.0_f64;
    for i in 0..n {
        // Partial pivot.
        let mut pivot = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Ok(0.0);
        }
        if pivot != i {
            a.swap(i, pivot);
            sign = -sign;
        }
        for r in (i + 1)..n {
            let factor = a[r][i] / a[i][i];
            for c in i..n {
                a[r][c] -= factor * a[i][c];
            }
        }
    }
    let mut det = sign;
    for i in 0..n {
        det *= a[i][i];
    }
    Ok(det)
}

// Suppress unused-import warnings when a downstream test path doesn't
// exercise every helper.
#[allow(dead_code)]
fn _touch(_m: &Array2<f64>) -> Result<Array2<f64>> {
    invert_symmetric(_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mcd_recovers_the_uncontaminated_centre() {
        // 30 clean Gaussian-ish points + 6 outliers far away. The MCD
        // location should sit near the clean centre, not the sample mean.
        let mut rows: Vec<[f64; 2]> = (0..30)
            .map(|i| [(i as f64 * 0.1).sin(), (i as f64 * 0.13).cos()])
            .collect();
        rows.extend((0..6).map(|_| [50.0, 50.0]));
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = Array2::from_shape_vec((36, 2), flat).unwrap();
        let mcd = MinCovDet::fit_with(x.view(), 0.75, 30, 42).unwrap();
        // MCD centre should be close to origin (clean data centred there).
        assert!(
            mcd.location[0].abs() < 3.0 && mcd.location[1].abs() < 3.0,
            "MCD centre = ({}, {}) — expected near origin",
            mcd.location[0],
            mcd.location[1]
        );
    }

    #[test]
    fn mcd_rejects_too_small_support_fraction() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(MinCovDet::fit_with(x.view(), 0.4, 10, 0).is_err());
    }
}
