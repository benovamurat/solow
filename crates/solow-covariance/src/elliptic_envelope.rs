//! EllipticEnvelope — anomaly detection via a robust Mahalanobis
//! threshold on top of a `MinCovDet` fit.

use ndarray::ArrayView2;
use solow_core::{Error, Result};

use crate::mcd::MinCovDet;

/// Fitted EllipticEnvelope.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct EllipticEnvelope {
    /// Underlying robust covariance fit.
    pub mcd: MinCovDet,
    /// Per-row squared Mahalanobis distance to the robust centre.
    pub scores: Vec<f64>,
    /// Contamination-cutoff threshold used to label outliers.
    pub threshold: f64,
    /// `+1` = inlier, `-1` = outlier, per training row.
    pub predictions: Vec<i64>,
    /// Fraction of expected outliers configured.
    pub contamination: f64,
}

impl EllipticEnvelope {
    /// Fit with the reference defaults `contamination = 0.1`, `support_fraction = None → 0.75`,
    /// `seed = 0`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 0.1, 0.75, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        contamination: f64,
        support_fraction: f64,
        seed: u64,
    ) -> Result<Self> {
        if !(0.0..0.5).contains(&contamination) {
            return Err(Error::Value(format!(
                "EllipticEnvelope: contamination must be in [0, 0.5) (got {contamination})"
            )));
        }
        let mcd = MinCovDet::fit_with(x, support_fraction, 30, seed)?;
        let scores = mcd
            .location
            .as_slice()
            .map(|_| ())
            .and_then(|_| mcd_scores(&mcd, x))
            .ok_or_else(|| Error::Value("EllipticEnvelope: Mahalanobis calculation failed".into()))?;
        // Contamination cutoff: mark the top-fraction as outliers.
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cutoff_rank = ((x.nrows() as f64) * contamination).ceil() as usize;
        let threshold = if cutoff_rank == 0 || cutoff_rank > x.nrows() {
            f64::INFINITY
        } else {
            sorted[cutoff_rank - 1]
        };
        let predictions: Vec<i64> = scores
            .iter()
            .map(|&s| if s >= threshold { -1 } else { 1 })
            .collect();
        Ok(Self {
            mcd,
            scores,
            threshold,
            predictions,
            contamination,
        })
    }
}

fn mcd_scores(mcd: &MinCovDet, x: ArrayView2<'_, f64>) -> Option<Vec<f64>> {
    let n = x.nrows();
    let d = x.ncols();
    // Rebuild a temporary EmpiricalCovariance-shaped struct so we can call
    // its mahalanobis helper — but simpler to inline here.
    // Compute inverse covariance via Gauss-Jordan.
    let inv = crate::empirical::invert_symmetric(&mcd.covariance).ok()?;
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut diff = vec![0.0_f64; d];
        for j in 0..d {
            diff[j] = x[[i, j]] - mcd.location[j];
        }
        let mut s = 0.0_f64;
        for j in 0..d {
            let mut tmp = 0.0_f64;
            for k in 0..d {
                tmp += inv[[j, k]] * diff[k];
            }
            s += diff[j] * tmp;
        }
        out[i] = s;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn elliptic_envelope_flags_a_lone_outlier_in_a_dense_cloud() {
        let mut rows: Vec<[f64; 2]> = (0..40)
            .map(|i| [(i as f64 * 0.1).sin(), (i as f64 * 0.13).cos()])
            .collect();
        rows.push([50.0, 50.0]);
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = Array2::from_shape_vec((41, 2), flat).unwrap();
        let m = EllipticEnvelope::fit_with(x.view(), 0.1, 0.75, 42).unwrap();
        assert_eq!(m.predictions[40], -1);
    }
}
