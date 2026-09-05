//! BernoulliRBM — the Restricted Boltzmann Machine with binary visible
//! and binary hidden units (Hinton 2002).
//!
//! Trained by Contrastive Divergence with `k = 1` Gibbs step
//! (`persistent = false`) — the reference default recipe.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted BernoulliRBM.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliRbm {
    /// Weight matrix `(n_hidden × n_features)`.
    pub components: Array2<f64>,
    /// Hidden-unit bias.
    pub intercept_hidden: Array1<f64>,
    /// Visible-unit bias.
    pub intercept_visible: Array1<f64>,
    /// Learning rate used.
    pub learning_rate: f64,
    /// Batch size used.
    pub batch_size: usize,
    /// Number of epochs run.
    pub n_iter: usize,
    /// Seed used.
    pub seed: u64,
}

impl BernoulliRbm {
    /// Fit with defaults `n_components = 256`, `learning_rate = 0.1`,
    /// `batch_size = 10`, `n_iter = 10`, `seed = 0`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 256, 0.1, 10, 10, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        learning_rate: f64,
        batch_size: usize,
        n_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 || d == 0 || n == 0 {
            return Err(Error::Value("BernoulliRbm: empty input or n_components = 0".into()));
        }
        // Init weights ~ 𝒩(0, 0.01).
        let mut state = seed.wrapping_add(0xBEEF_D00D_F00D);
        let mut w = Array2::<f64>::zeros((n_components, d));
        for i in 0..n_components {
            for j in 0..d {
                w[[i, j]] = 0.01 * standard_normal(&mut state);
            }
        }
        let mut b_hid = Array1::<f64>::zeros(n_components);
        let mut b_vis = Array1::<f64>::zeros(d);
        // Init visible bias to log(p / (1 - p)) per feature.
        for j in 0..d {
            let mut mean = 0.0_f64;
            for i in 0..n {
                mean += x[[i, j]];
            }
            mean /= n as f64;
            let p = mean.clamp(1e-3, 1.0 - 1e-3);
            b_vis[j] = (p / (1.0 - p)).ln();
        }
        let mut epoch = 0_usize;
        for e in 0..n_iter {
            epoch = e + 1;
            let mut start = 0_usize;
            while start < n {
                let end = (start + batch_size).min(n);
                let bs = end - start;
                // Positive phase.
                let mut ph = Array2::<f64>::zeros((bs, n_components));
                for r in 0..bs {
                    for h in 0..n_components {
                        let mut s = b_hid[h];
                        for j in 0..d {
                            s += w[[h, j]] * x[[start + r, j]];
                        }
                        ph[[r, h]] = sigmoid(s);
                    }
                }
                // Sample hidden (Bernoulli).
                let mut hs = Array2::<f64>::zeros((bs, n_components));
                for r in 0..bs {
                    for h in 0..n_components {
                        hs[[r, h]] = if uniform01(&mut state) < ph[[r, h]] { 1.0 } else { 0.0 };
                    }
                }
                // Negative phase — reconstruct visible then re-compute hidden.
                let mut nv = Array2::<f64>::zeros((bs, d));
                for r in 0..bs {
                    for j in 0..d {
                        let mut s = b_vis[j];
                        for h in 0..n_components {
                            s += w[[h, j]] * hs[[r, h]];
                        }
                        nv[[r, j]] = sigmoid(s);
                    }
                }
                let mut nh = Array2::<f64>::zeros((bs, n_components));
                for r in 0..bs {
                    for h in 0..n_components {
                        let mut s = b_hid[h];
                        for j in 0..d {
                            s += w[[h, j]] * nv[[r, j]];
                        }
                        nh[[r, h]] = sigmoid(s);
                    }
                }
                // Weight gradient — (positive − negative) / bs.
                let lr = learning_rate / bs as f64;
                for h in 0..n_components {
                    for j in 0..d {
                        let mut pos = 0.0_f64;
                        let mut neg = 0.0_f64;
                        for r in 0..bs {
                            pos += ph[[r, h]] * x[[start + r, j]];
                            neg += nh[[r, h]] * nv[[r, j]];
                        }
                        w[[h, j]] += lr * (pos - neg);
                    }
                }
                for h in 0..n_components {
                    let mut pos = 0.0_f64;
                    let mut neg = 0.0_f64;
                    for r in 0..bs {
                        pos += ph[[r, h]];
                        neg += nh[[r, h]];
                    }
                    b_hid[h] += lr * (pos - neg);
                }
                for j in 0..d {
                    let mut pos = 0.0_f64;
                    let mut neg = 0.0_f64;
                    for r in 0..bs {
                        pos += x[[start + r, j]];
                        neg += nv[[r, j]];
                    }
                    b_vis[j] += lr * (pos - neg);
                }
                start = end;
            }
        }
        Ok(Self {
            components: w,
            intercept_hidden: b_hid,
            intercept_visible: b_vis,
            learning_rate,
            batch_size,
            n_iter: epoch,
            seed,
        })
    }

    /// Hidden-layer activations `p(h = 1 | v)`.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let h = self.components.nrows();
        if d != self.components.ncols() {
            return Err(Error::Shape("BernoulliRbm::transform: shape mismatch".into()));
        }
        let mut out = Array2::<f64>::zeros((n, h));
        for i in 0..n {
            for k in 0..h {
                let mut s = self.intercept_hidden[k];
                for j in 0..d {
                    s += self.components[[k, j]] * x[[i, j]];
                }
                out[[i, k]] = sigmoid(s);
            }
        }
        Ok(out)
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(1e-12);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn uniform01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let r = *state >> 11;
    (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rbm_output_has_the_right_shape() {
        let x = array![
            [1.0_f64, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]
        ];
        let rbm = BernoulliRbm::fit_with(x.view(), 3, 0.1, 2, 5, 42).unwrap();
        let z = rbm.transform(x.view()).unwrap();
        assert_eq!(z.shape(), &[3, 3]);
    }
}
