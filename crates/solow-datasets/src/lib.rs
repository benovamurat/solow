//! # solow-datasets
//!
//! Synthetic dataset generators — the reference `datasets` toy
//! and generator surface. Every function is deterministic under a
//! caller-supplied seed via a portable MMIX-LCG.
//!
//! * [`make_classification`] — Guyon-Ben-Hamdi random hypercube
//!   classification problems.
//! * [`make_regression`] — linear regression with configurable noise.
//! * [`make_blobs`] — isotropic Gaussian clusters.
//! * [`make_moons`] — two interleaved half-moons.
//! * [`make_circles`] — one large / one small concentric circles.
//! * [`make_swiss_roll`] — canonical 3-D non-linear manifold benchmark.
//! * [`make_low_rank_matrix`] — mostly-low-rank random matrix.
//! * [`load_iris`] — the classical Fisher iris dataset, embedded verbatim.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod toy;
pub mod utils;

pub use toy::{load_breast_cancer, load_diabetes, load_wine};
pub use utils::{
    compute_class_weight, compute_sample_weight, resample_indices_no_replace,
    resample_indices_with_replace, take_rows,
};

use ndarray::{Array1, Array2};
use solow_core::{Error, Result};

pub(crate) struct Lcg {
    state: u64,
}

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0xF00D_C0DE),
        }
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    pub(crate) fn uniform01(&mut self) -> f64 {
        let r = self.next() >> 11;
        (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
    }

    pub(crate) fn standard_normal(&mut self) -> f64 {
        loop {
            let u1 = self.uniform01();
            if u1 > 1e-300 {
                let u2 = self.uniform01();
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }

    pub(crate) fn uniform_index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let max = u64::MAX - (u64::MAX % n);
        loop {
            let r = self.next();
            if r < max {
                return (r % n) as usize;
            }
        }
    }
}

/// Generate a random binary / multi-class classification problem.
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array1<usize>)> {
    if n_samples == 0 || n_features == 0 {
        return Err(Error::Value("make_classification: n_samples/n_features must be ≥ 1".into()));
    }
    if n_classes < 2 {
        return Err(Error::Value("make_classification: n_classes must be ≥ 2".into()));
    }
    let mut rng = Lcg::new(seed);
    let mut centres = Array2::<f64>::zeros((n_classes, n_features));
    for c in 0..n_classes {
        for j in 0..n_features {
            centres[[c, j]] = rng.standard_normal() * 3.0;
        }
    }
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<usize>::zeros(n_samples);
    for i in 0..n_samples {
        let c = rng.uniform_index(n_classes);
        y[i] = c;
        for j in 0..n_features {
            x[[i, j]] = centres[[c, j]] + rng.standard_normal();
        }
    }
    Ok((x, y))
}

/// Generate a linear regression dataset `y = X·w + noise`.
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    seed: u64,
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
    if n_samples == 0 || n_features == 0 {
        return Err(Error::Value("make_regression: n_samples/n_features must be ≥ 1".into()));
    }
    let mut rng = Lcg::new(seed);
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.standard_normal();
        }
    }
    let mut w = Array1::<f64>::zeros(n_features);
    for j in 0..n_features {
        w[j] = rng.standard_normal() * 5.0;
    }
    let mut y = Array1::<f64>::zeros(n_samples);
    for i in 0..n_samples {
        let mut s = 0.0_f64;
        for j in 0..n_features {
            s += x[[i, j]] * w[j];
        }
        y[i] = s + rng.standard_normal() * noise;
    }
    Ok((x, y, w))
}

/// Generate isotropic Gaussian clusters.
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    n_centres: usize,
    cluster_std: f64,
    seed: u64,
) -> Result<(Array2<f64>, Array1<usize>)> {
    if n_samples == 0 || n_features == 0 || n_centres == 0 {
        return Err(Error::Value("make_blobs: sizes must be ≥ 1".into()));
    }
    let mut rng = Lcg::new(seed);
    let mut centres = Array2::<f64>::zeros((n_centres, n_features));
    for c in 0..n_centres {
        for j in 0..n_features {
            centres[[c, j]] = rng.standard_normal() * 10.0;
        }
    }
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<usize>::zeros(n_samples);
    for i in 0..n_samples {
        let c = rng.uniform_index(n_centres);
        y[i] = c;
        for j in 0..n_features {
            x[[i, j]] = centres[[c, j]] + cluster_std * rng.standard_normal();
        }
    }
    Ok((x, y))
}

/// Generate two interleaved half-moons.
pub fn make_moons(n_samples: usize, noise: f64, seed: u64) -> Result<(Array2<f64>, Array1<usize>)> {
    if n_samples < 2 {
        return Err(Error::Value("make_moons: n_samples must be ≥ 2".into()));
    }
    let mut rng = Lcg::new(seed);
    let mut x = Array2::<f64>::zeros((n_samples, 2));
    let mut y = Array1::<usize>::zeros(n_samples);
    let n_a = n_samples / 2;
    let n_b = n_samples - n_a;
    for i in 0..n_a {
        let t = std::f64::consts::PI * i as f64 / (n_a - 1).max(1) as f64;
        x[[i, 0]] = t.cos() + noise * rng.standard_normal();
        x[[i, 1]] = t.sin() + noise * rng.standard_normal();
        y[i] = 0;
    }
    for i in 0..n_b {
        let t = std::f64::consts::PI * i as f64 / (n_b - 1).max(1) as f64;
        x[[n_a + i, 0]] = 1.0 - t.cos() + noise * rng.standard_normal();
        x[[n_a + i, 1]] = -t.sin() + 0.5 + noise * rng.standard_normal();
        y[n_a + i] = 1;
    }
    Ok((x, y))
}

/// Generate two concentric circles.
pub fn make_circles(
    n_samples: usize,
    factor: f64,
    noise: f64,
    seed: u64,
) -> Result<(Array2<f64>, Array1<usize>)> {
    if n_samples < 2 {
        return Err(Error::Value("make_circles: n_samples must be ≥ 2".into()));
    }
    if !(0.0..1.0).contains(&factor) {
        return Err(Error::Value("make_circles: factor must be in (0, 1)".into()));
    }
    let mut rng = Lcg::new(seed);
    let mut x = Array2::<f64>::zeros((n_samples, 2));
    let mut y = Array1::<usize>::zeros(n_samples);
    let n_a = n_samples / 2;
    let n_b = n_samples - n_a;
    for i in 0..n_a {
        let t = 2.0 * std::f64::consts::PI * i as f64 / (n_a - 1).max(1) as f64;
        x[[i, 0]] = t.cos() + noise * rng.standard_normal();
        x[[i, 1]] = t.sin() + noise * rng.standard_normal();
        y[i] = 0;
    }
    for i in 0..n_b {
        let t = 2.0 * std::f64::consts::PI * i as f64 / (n_b - 1).max(1) as f64;
        x[[n_a + i, 0]] = factor * t.cos() + noise * rng.standard_normal();
        x[[n_a + i, 1]] = factor * t.sin() + noise * rng.standard_normal();
        y[n_a + i] = 1;
    }
    Ok((x, y))
}

/// Generate a 3-D Swiss-roll manifold with the intrinsic parameter as
/// the target.
pub fn make_swiss_roll(
    n_samples: usize,
    noise: f64,
    seed: u64,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(Error::Value("make_swiss_roll: n_samples must be ≥ 1".into()));
    }
    let mut rng = Lcg::new(seed);
    let mut x = Array2::<f64>::zeros((n_samples, 3));
    let mut t_vec = Array1::<f64>::zeros(n_samples);
    for i in 0..n_samples {
        let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * rng.uniform01());
        let h = 21.0 * rng.uniform01();
        x[[i, 0]] = t * t.cos() + noise * rng.standard_normal();
        x[[i, 1]] = h + noise * rng.standard_normal();
        x[[i, 2]] = t * t.sin() + noise * rng.standard_normal();
        t_vec[i] = t;
    }
    Ok((x, t_vec))
}

/// Generate a random mostly-low-rank matrix.
pub fn make_low_rank_matrix(
    n_samples: usize,
    n_features: usize,
    effective_rank: usize,
    tail_strength: f64,
    seed: u64,
) -> Result<Array2<f64>> {
    if n_samples == 0 || n_features == 0 || effective_rank == 0 {
        return Err(Error::Value("make_low_rank_matrix: sizes must be ≥ 1".into()));
    }
    let mut rng = Lcg::new(seed);
    let n = n_samples.min(n_features);
    let mut sv = vec![0.0_f64; n];
    for i in 0..n {
        let x = i as f64 / effective_rank as f64;
        let hi = (-x * x).exp();
        let lo = tail_strength * (-x / 4.0).exp();
        sv[i] = hi + lo;
    }
    let mut u = Array2::<f64>::zeros((n_samples, n));
    for i in 0..n_samples {
        for j in 0..n {
            u[[i, j]] = rng.standard_normal();
        }
    }
    let mut v = Array2::<f64>::zeros((n, n_features));
    for i in 0..n {
        for j in 0..n_features {
            v[[i, j]] = rng.standard_normal();
        }
    }
    let mut out = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += u[[i, k]] * sv[k] * v[[k, j]];
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

/// Load the classical Fisher iris dataset (150 × 4).
pub fn load_iris() -> (Array2<f64>, Array1<usize>, Vec<&'static str>) {
    let raw: &[[f64; 5]] = &IRIS_TABLE;
    let n = raw.len();
    let mut x = Array2::<f64>::zeros((n, 4));
    let mut y = Array1::<usize>::zeros(n);
    for (i, row) in raw.iter().enumerate() {
        for j in 0..4 {
            x[[i, j]] = row[j];
        }
        y[i] = row[4] as usize;
    }
    (x, y, vec!["sepal length", "sepal width", "petal length", "petal width"])
}

const IRIS_TABLE: [[f64; 5]; 150] = [
    [5.1, 3.5, 1.4, 0.2, 0.0], [4.9, 3.0, 1.4, 0.2, 0.0], [4.7, 3.2, 1.3, 0.2, 0.0], [4.6, 3.1, 1.5, 0.2, 0.0], [5.0, 3.6, 1.4, 0.2, 0.0],
    [5.4, 3.9, 1.7, 0.4, 0.0], [4.6, 3.4, 1.4, 0.3, 0.0], [5.0, 3.4, 1.5, 0.2, 0.0], [4.4, 2.9, 1.4, 0.2, 0.0], [4.9, 3.1, 1.5, 0.1, 0.0],
    [5.4, 3.7, 1.5, 0.2, 0.0], [4.8, 3.4, 1.6, 0.2, 0.0], [4.8, 3.0, 1.4, 0.1, 0.0], [4.3, 3.0, 1.1, 0.1, 0.0], [5.8, 4.0, 1.2, 0.2, 0.0],
    [5.7, 4.4, 1.5, 0.4, 0.0], [5.4, 3.9, 1.3, 0.4, 0.0], [5.1, 3.5, 1.4, 0.3, 0.0], [5.7, 3.8, 1.7, 0.3, 0.0], [5.1, 3.8, 1.5, 0.3, 0.0],
    [5.4, 3.4, 1.7, 0.2, 0.0], [5.1, 3.7, 1.5, 0.4, 0.0], [4.6, 3.6, 1.0, 0.2, 0.0], [5.1, 3.3, 1.7, 0.5, 0.0], [4.8, 3.4, 1.9, 0.2, 0.0],
    [5.0, 3.0, 1.6, 0.2, 0.0], [5.0, 3.4, 1.6, 0.4, 0.0], [5.2, 3.5, 1.5, 0.2, 0.0], [5.2, 3.4, 1.4, 0.2, 0.0], [4.7, 3.2, 1.6, 0.2, 0.0],
    [4.8, 3.1, 1.6, 0.2, 0.0], [5.4, 3.4, 1.5, 0.4, 0.0], [5.2, 4.1, 1.5, 0.1, 0.0], [5.5, 4.2, 1.4, 0.2, 0.0], [4.9, 3.1, 1.5, 0.2, 0.0],
    [5.0, 3.2, 1.2, 0.2, 0.0], [5.5, 3.5, 1.3, 0.2, 0.0], [4.9, 3.6, 1.4, 0.1, 0.0], [4.4, 3.0, 1.3, 0.2, 0.0], [5.1, 3.4, 1.5, 0.2, 0.0],
    [5.0, 3.5, 1.3, 0.3, 0.0], [4.5, 2.3, 1.3, 0.3, 0.0], [4.4, 3.2, 1.3, 0.2, 0.0], [5.0, 3.5, 1.6, 0.6, 0.0], [5.1, 3.8, 1.9, 0.4, 0.0],
    [4.8, 3.0, 1.4, 0.3, 0.0], [5.1, 3.8, 1.6, 0.2, 0.0], [4.6, 3.2, 1.4, 0.2, 0.0], [5.3, 3.7, 1.5, 0.2, 0.0], [5.0, 3.3, 1.4, 0.2, 0.0],
    [7.0, 3.2, 4.7, 1.4, 1.0], [6.4, 3.2, 4.5, 1.5, 1.0], [6.9, 3.1, 4.9, 1.5, 1.0], [5.5, 2.3, 4.0, 1.3, 1.0], [6.5, 2.8, 4.6, 1.5, 1.0],
    [5.7, 2.8, 4.5, 1.3, 1.0], [6.3, 3.3, 4.7, 1.6, 1.0], [4.9, 2.4, 3.3, 1.0, 1.0], [6.6, 2.9, 4.6, 1.3, 1.0], [5.2, 2.7, 3.9, 1.4, 1.0],
    [5.0, 2.0, 3.5, 1.0, 1.0], [5.9, 3.0, 4.2, 1.5, 1.0], [6.0, 2.2, 4.0, 1.0, 1.0], [6.1, 2.9, 4.7, 1.4, 1.0], [5.6, 2.9, 3.6, 1.3, 1.0],
    [6.7, 3.1, 4.4, 1.4, 1.0], [5.6, 3.0, 4.5, 1.5, 1.0], [5.8, 2.7, 4.1, 1.0, 1.0], [6.2, 2.2, 4.5, 1.5, 1.0], [5.6, 2.5, 3.9, 1.1, 1.0],
    [5.9, 3.2, 4.8, 1.8, 1.0], [6.1, 2.8, 4.0, 1.3, 1.0], [6.3, 2.5, 4.9, 1.5, 1.0], [6.1, 2.8, 4.7, 1.2, 1.0], [6.4, 2.9, 4.3, 1.3, 1.0],
    [6.6, 3.0, 4.4, 1.4, 1.0], [6.8, 2.8, 4.8, 1.4, 1.0], [6.7, 3.0, 5.0, 1.7, 1.0], [6.0, 2.9, 4.5, 1.5, 1.0], [5.7, 2.6, 3.5, 1.0, 1.0],
    [5.5, 2.4, 3.8, 1.1, 1.0], [5.5, 2.4, 3.7, 1.0, 1.0], [5.8, 2.7, 3.9, 1.2, 1.0], [6.0, 2.7, 5.1, 1.6, 1.0], [5.4, 3.0, 4.5, 1.5, 1.0],
    [6.0, 3.4, 4.5, 1.6, 1.0], [6.7, 3.1, 4.7, 1.5, 1.0], [6.3, 2.3, 4.4, 1.3, 1.0], [5.6, 3.0, 4.1, 1.3, 1.0], [5.5, 2.5, 4.0, 1.3, 1.0],
    [5.5, 2.6, 4.4, 1.2, 1.0], [6.1, 3.0, 4.6, 1.4, 1.0], [5.8, 2.6, 4.0, 1.2, 1.0], [5.0, 2.3, 3.3, 1.0, 1.0], [5.6, 2.7, 4.2, 1.3, 1.0],
    [5.7, 3.0, 4.2, 1.2, 1.0], [5.7, 2.9, 4.2, 1.3, 1.0], [6.2, 2.9, 4.3, 1.3, 1.0], [5.1, 2.5, 3.0, 1.1, 1.0], [5.7, 2.8, 4.1, 1.3, 1.0],
    [6.3, 3.3, 6.0, 2.5, 2.0], [5.8, 2.7, 5.1, 1.9, 2.0], [7.1, 3.0, 5.9, 2.1, 2.0], [6.3, 2.9, 5.6, 1.8, 2.0], [6.5, 3.0, 5.8, 2.2, 2.0],
    [7.6, 3.0, 6.6, 2.1, 2.0], [4.9, 2.5, 4.5, 1.7, 2.0], [7.3, 2.9, 6.3, 1.8, 2.0], [6.7, 2.5, 5.8, 1.8, 2.0], [7.2, 3.6, 6.1, 2.5, 2.0],
    [6.5, 3.2, 5.1, 2.0, 2.0], [6.4, 2.7, 5.3, 1.9, 2.0], [6.8, 3.0, 5.5, 2.1, 2.0], [5.7, 2.5, 5.0, 2.0, 2.0], [5.8, 2.8, 5.1, 2.4, 2.0],
    [6.4, 3.2, 5.3, 2.3, 2.0], [6.5, 3.0, 5.5, 1.8, 2.0], [7.7, 3.8, 6.7, 2.2, 2.0], [7.7, 2.6, 6.9, 2.3, 2.0], [6.0, 2.2, 5.0, 1.5, 2.0],
    [6.9, 3.2, 5.7, 2.3, 2.0], [5.6, 2.8, 4.9, 2.0, 2.0], [7.7, 2.8, 6.7, 2.0, 2.0], [6.3, 2.7, 4.9, 1.8, 2.0], [6.7, 3.3, 5.7, 2.1, 2.0],
    [7.2, 3.2, 6.0, 1.8, 2.0], [6.2, 2.8, 4.8, 1.8, 2.0], [6.1, 3.0, 4.9, 1.8, 2.0], [6.4, 2.8, 5.6, 2.1, 2.0], [7.2, 3.0, 5.8, 1.6, 2.0],
    [7.4, 2.8, 6.1, 1.9, 2.0], [7.9, 3.8, 6.4, 2.0, 2.0], [6.4, 2.8, 5.6, 2.2, 2.0], [6.3, 2.8, 5.1, 1.5, 2.0], [6.1, 2.6, 5.6, 1.4, 2.0],
    [7.7, 3.0, 6.1, 2.3, 2.0], [6.3, 3.4, 5.6, 2.4, 2.0], [6.4, 3.1, 5.5, 1.8, 2.0], [6.0, 3.0, 4.8, 1.8, 2.0], [6.9, 3.1, 5.4, 2.1, 2.0],
    [6.7, 3.1, 5.6, 2.4, 2.0], [6.9, 3.1, 5.1, 2.3, 2.0], [5.8, 2.7, 5.1, 1.9, 2.0], [6.8, 3.2, 5.9, 2.3, 2.0], [6.7, 3.3, 5.7, 2.5, 2.0],
    [6.7, 3.0, 5.2, 2.3, 2.0], [6.3, 2.5, 5.0, 1.9, 2.0], [6.5, 3.0, 5.2, 2.0, 2.0], [6.2, 3.4, 5.4, 2.3, 2.0], [5.9, 3.0, 5.1, 1.8, 2.0],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_classification_deterministic_at_seed() {
        let (a, ay) = make_classification(50, 3, 3, 42).unwrap();
        let (b, by) = make_classification(50, 3, 3, 42).unwrap();
        for i in 0..50 {
            assert_eq!(ay[i], by[i]);
            for j in 0..3 {
                assert_eq!(a[[i, j]], b[[i, j]]);
            }
        }
    }

    #[test]
    fn make_regression_recovers_linear_signal_under_no_noise() {
        let (x, y, w) = make_regression(20, 3, 0.0, 7).unwrap();
        // y = X·w exactly.
        for i in 0..20 {
            let mut s = 0.0_f64;
            for j in 0..3 {
                s += x[[i, j]] * w[j];
            }
            assert!((y[i] - s).abs() < 1e-10);
        }
    }

    #[test]
    fn load_iris_has_150_rows_of_shape_4_and_3_classes() {
        let (x, y, _names) = load_iris();
        assert_eq!(x.shape(), &[150, 4]);
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort();
        classes.dedup();
        assert_eq!(classes, vec![0, 1, 2]);
    }
}
