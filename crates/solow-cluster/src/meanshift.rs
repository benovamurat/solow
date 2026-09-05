//! Mean-shift clustering (Comaniciu-Meer 2002).
//!
//! Non-parametric mode-seeking on the Gaussian-kernel density
//! estimate: each sample climbs the local density gradient until it
//! converges to a mode. Modes are then merged by
//! `bandwidth`-radius connectivity to give the final cluster centres.
//!
//! # Complexity
//!
//! Per-iteration `O(n²)` because every sample sees a `bandwidth`-
//! radius neighbourhood; suitable for `n` up to a few thousand.
//!
//! # References
//!
//! * Comaniciu, D., & Meer, P. (2002). *Mean shift: A robust approach
//!   toward feature space analysis.* IEEE PAMI 24(5), 603-619.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted mean-shift result.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MeanShift {
    /// Discovered cluster centres.
    pub cluster_centers: Array2<f64>,
    /// Per-sample cluster label (index into `cluster_centers`).
    pub labels: Array1<usize>,
    /// Bandwidth used at fit time.
    pub bandwidth: f64,
}

impl MeanShift {
    /// Fit mean-shift with the given Gaussian bandwidth.
    ///
    /// Each sample is iterated toward its local density mode up to
    /// `max_iter` times, terminating on a per-sample shift below `tol`.
    /// After convergence, modes within `bandwidth` of each other are
    /// merged; every sample is assigned the label of the closest
    /// surviving mode.
    pub fn fit(x: ArrayView2<'_, f64>, bandwidth: f64) -> Result<Self> {
        Self::fit_with(x, bandwidth, 300, 1e-3)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        bandwidth: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "MeanShift::fit_with: x must be non-empty".into(),
            ));
        }
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return Err(Error::Value(format!(
                "MeanShift::fit_with: bandwidth must be finite and > 0 (got {bandwidth})"
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        // Climb each sample.
        let bw2 = bandwidth * bandwidth;
        let mut peaks = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                peaks[[i, j]] = x[[i, j]];
            }
            for _ in 0..max_iter {
                let mut num = vec![0.0_f64; d];
                let mut den = 0.0_f64;
                for r in 0..n {
                    let mut dist2 = 0.0_f64;
                    for j in 0..d {
                        let diff = peaks[[i, j]] - x[[r, j]];
                        dist2 += diff * diff;
                    }
                    // Gaussian kernel weight.
                    let w = (-dist2 / (2.0 * bw2)).exp();
                    for j in 0..d {
                        num[j] += w * x[[r, j]];
                    }
                    den += w;
                }
                if den <= 0.0 {
                    break;
                }
                let mut max_shift = 0.0_f64;
                for j in 0..d {
                    let new_val = num[j] / den;
                    let shift = (new_val - peaks[[i, j]]).abs();
                    if shift > max_shift {
                        max_shift = shift;
                    }
                    peaks[[i, j]] = new_val;
                }
                if max_shift < tol {
                    break;
                }
            }
        }
        // Merge peaks within `bandwidth` of an existing centre.
        let mut centers: Vec<Vec<f64>> = Vec::new();
        let mut assign = vec![0usize; n];
        for i in 0..n {
            let mut matched: Option<usize> = None;
            for (ci, c) in centers.iter().enumerate() {
                let mut dist2 = 0.0_f64;
                for j in 0..d {
                    let diff = peaks[[i, j]] - c[j];
                    dist2 += diff * diff;
                }
                if dist2 <= bw2 {
                    matched = Some(ci);
                    break;
                }
            }
            match matched {
                Some(ci) => assign[i] = ci,
                None => {
                    let new_center: Vec<f64> = (0..d).map(|j| peaks[[i, j]]).collect();
                    assign[i] = centers.len();
                    centers.push(new_center);
                }
            }
        }
        let mut cluster_centers = Array2::<f64>::zeros((centers.len(), d));
        for (ci, c) in centers.iter().enumerate() {
            for j in 0..d {
                cluster_centers[[ci, j]] = c[j];
            }
        }
        Ok(Self {
            cluster_centers,
            labels: Array1::from(assign),
            bandwidth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mean_shift_finds_two_dense_blobs() {
        // Two well-separated blobs — mean-shift with a bandwidth smaller
        // than the inter-cluster distance should recover exactly two modes.
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05]
        ];
        let ms = MeanShift::fit(x.view(), 1.5).unwrap();
        assert_eq!(ms.cluster_centers.nrows(), 2);
        let (a, b) = (ms.labels[0], ms.labels[5]);
        assert_ne!(a, b);
        for i in 0..5 {
            assert_eq!(ms.labels[i], a);
        }
        for i in 5..10 {
            assert_eq!(ms.labels[i], b);
        }
    }
}
