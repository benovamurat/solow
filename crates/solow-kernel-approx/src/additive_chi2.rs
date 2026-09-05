//! Vedaldi-Zisserman additive-χ² kernel feature map.
//!
//! `k_{χ²}(x, y) = Σ_i 2·xᵢ·yᵢ / (xᵢ + yᵢ)` for non-negative inputs.
//! The explicit closed-form map is the concatenation of `sqrt(xᵢ)` and,
//! for each `ω ∈ Ω`, `sqrt(2xᵢ·sech(π·ω))·{cos(ω·ln xᵢ), sin(ω·ln xᵢ)}`.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// AdditiveChi2Sampler.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveChi2Sampler {
    /// Half-window `n` in `ω = k·step` for `k ∈ {−n, …, n}`.
    pub sample_steps: usize,
    /// Spacing between consecutive samples.
    pub sample_interval: f64,
    /// Input feature count captured at fit time.
    pub n_features: usize,
}

impl AdditiveChi2Sampler {
    /// Fit with the reference defaults `sample_steps = 2`, `sample_interval = 0.5`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 2, 0.5)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        sample_steps: usize,
        sample_interval: f64,
    ) -> Result<Self> {
        if sample_interval <= 0.0 {
            return Err(Error::Value(
                "AdditiveChi2Sampler: sample_interval must be > 0".into(),
            ));
        }
        // Check non-negativity.
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                if x[[i, j]] < 0.0 {
                    return Err(Error::Value(
                        "AdditiveChi2Sampler: inputs must be non-negative".into(),
                    ));
                }
            }
        }
        Ok(Self {
            sample_steps,
            sample_interval,
            n_features: x.ncols(),
        })
    }

    /// Number of output features per input feature.
    pub fn feature_multiplier(&self) -> usize {
        1 + 2 * self.sample_steps
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(Error::Shape(format!(
                "AdditiveChi2Sampler::transform: expected {} cols, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let p = self.n_features;
        let d = self.feature_multiplier();
        let mut out = Array2::<f64>::zeros((n, p * d));
        for i in 0..n {
            for j in 0..p {
                let xij = x[[i, j]];
                let base = j * d;
                out[[i, base]] = xij.sqrt();
                for k in 1..=self.sample_steps {
                    let omega = k as f64 * self.sample_interval;
                    let factor =
                        (2.0 * xij / (std::f64::consts::PI * omega).cosh()).sqrt();
                    let (cosw, sinw) = if xij > 0.0 {
                        let l = xij.ln();
                        ((omega * l).cos(), (omega * l).sin())
                    } else {
                        (0.0, 0.0)
                    };
                    out[[i, base + 2 * k - 1]] = factor * cosw;
                    out[[i, base + 2 * k]] = factor * sinw;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn additive_chi2_widens_by_the_right_multiplier() {
        let x = array![[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]];
        let m = AdditiveChi2Sampler::fit_with(x.view(), 3, 0.25).unwrap();
        let z = m.transform(x.view()).unwrap();
        // 3 original features × (1 + 2·3) = 21 output features
        assert_eq!(z.shape(), &[2, 21]);
    }

    #[test]
    fn additive_chi2_rejects_negative_input() {
        let x = array![[1.0, -1.0]];
        assert!(AdditiveChi2Sampler::fit(x.view()).is_err());
    }
}
