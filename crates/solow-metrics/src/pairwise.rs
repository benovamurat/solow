//! Pairwise distances and kernels — the reference `metrics.pairwise`
//! surface.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Distance / kernel metric.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairwiseMetric {
    /// Euclidean distance.
    Euclidean,
    /// Squared Euclidean.
    SqEuclidean,
    /// Manhattan (L1).
    Manhattan,
    /// Chebyshev (L∞).
    Chebyshev,
    /// Cosine distance `1 − cos(x, y)`.
    Cosine,
    /// Minkowski distance with a caller-supplied exponent `p`.
    Minkowski {
        /// Exponent in `d(x, y) = (Σ|xᵢ − yᵢ|^p)^{1/p}`.
        p: f64,
    },
    /// Hamming (fraction of components differing).
    Hamming,
}

/// Pairwise distance matrix `(n × m)` between rows of `a` and `b`.
pub fn pairwise_distances(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    metric: PairwiseMetric,
) -> Result<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(Error::Shape(
            "pairwise_distances: a and b must have the same number of columns".into(),
        ));
    }
    let n = a.nrows();
    let m = b.nrows();
    let d = a.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = distance(a.row(i).to_owned().as_slice().unwrap(),
                                    b.row(j).to_owned().as_slice().unwrap(),
                                    metric,
                                    d);
        }
    }
    Ok(out)
}

fn distance(a: &[f64], b: &[f64], metric: PairwiseMetric, d: usize) -> f64 {
    match metric {
        PairwiseMetric::Euclidean => {
            let mut s = 0.0_f64;
            for k in 0..d {
                let e = a[k] - b[k];
                s += e * e;
            }
            s.sqrt()
        }
        PairwiseMetric::SqEuclidean => {
            let mut s = 0.0_f64;
            for k in 0..d {
                let e = a[k] - b[k];
                s += e * e;
            }
            s
        }
        PairwiseMetric::Manhattan => {
            let mut s = 0.0_f64;
            for k in 0..d {
                s += (a[k] - b[k]).abs();
            }
            s
        }
        PairwiseMetric::Chebyshev => {
            let mut best = 0.0_f64;
            for k in 0..d {
                let e = (a[k] - b[k]).abs();
                if e > best {
                    best = e;
                }
            }
            best
        }
        PairwiseMetric::Cosine => {
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for k in 0..d {
                dot += a[k] * b[k];
                na += a[k] * a[k];
                nb += b[k] * b[k];
            }
            1.0 - dot / (na.sqrt() * nb.sqrt()).max(1e-30)
        }
        PairwiseMetric::Minkowski { p } => {
            let mut s = 0.0_f64;
            for k in 0..d {
                s += (a[k] - b[k]).abs().powf(p);
            }
            s.powf(1.0 / p.max(1e-30))
        }
        PairwiseMetric::Hamming => {
            let mut mismatches = 0_usize;
            for k in 0..d {
                if a[k] != b[k] {
                    mismatches += 1;
                }
            }
            mismatches as f64 / d as f64
        }
    }
}

/// RBF (Gaussian) kernel `exp(−γ ‖x − y‖²)`.
pub fn rbf_kernel(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>, gamma: f64) -> Result<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(Error::Shape("rbf_kernel: shape mismatch".into()));
    }
    let n = a.nrows();
    let m = b.nrows();
    let d = a.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for k in 0..d {
                let e = a[[i, k]] - b[[j, k]];
                s += e * e;
            }
            out[[i, j]] = (-gamma * s).exp();
        }
    }
    Ok(out)
}

/// Linear kernel `x·y`.
pub fn linear_kernel(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(Error::Shape("linear_kernel: shape mismatch".into()));
    }
    let n = a.nrows();
    let m = b.nrows();
    let d = a.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for k in 0..d {
                s += a[[i, k]] * b[[j, k]];
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

/// Polynomial kernel `(γ · x·y + coef0)^degree`.
pub fn polynomial_kernel(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    gamma: f64,
    coef0: f64,
    degree: i32,
) -> Result<Array2<f64>> {
    let mut out = linear_kernel(a, b)?;
    for v in out.iter_mut() {
        *v = (gamma * *v + coef0).powi(degree);
    }
    Ok(out)
}

/// Sigmoid kernel `tanh(γ · x·y + coef0)`.
pub fn sigmoid_kernel(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    gamma: f64,
    coef0: f64,
) -> Result<Array2<f64>> {
    let mut out = linear_kernel(a, b)?;
    for v in out.iter_mut() {
        *v = (gamma * *v + coef0).tanh();
    }
    Ok(out)
}

/// Laplacian kernel `exp(−γ ‖x − y‖₁)`.
pub fn laplacian_kernel(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    gamma: f64,
) -> Result<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(Error::Shape("laplacian_kernel: shape mismatch".into()));
    }
    let n = a.nrows();
    let m = b.nrows();
    let d = a.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for k in 0..d {
                s += (a[[i, k]] - b[[j, k]]).abs();
            }
            out[[i, j]] = (-gamma * s).exp();
        }
    }
    Ok(out)
}

/// Cosine similarity `x·y / (‖x‖ · ‖y‖)`.
pub fn cosine_similarity(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Result<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(Error::Shape("cosine_similarity: shape mismatch".into()));
    }
    let n = a.nrows();
    let m = b.nrows();
    let d = a.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for k in 0..d {
                dot += a[[i, k]] * b[[j, k]];
                na += a[[i, k]] * a[[i, k]];
                nb += b[[j, k]] * b[[j, k]];
            }
            out[[i, j]] = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        }
    }
    Ok(out)
}

/// Additive χ² kernel `Σᵢ 2xᵢyᵢ / (xᵢ + yᵢ)` for non-negative inputs.
pub fn chi2_kernel(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(Error::Shape("chi2_kernel: shape mismatch".into()));
    }
    let n = a.nrows();
    let m = b.nrows();
    let d = a.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for k in 0..d {
                let sum = a[[i, k]] + b[[j, k]];
                if sum > 0.0 {
                    s += 2.0 * a[[i, k]] * b[[j, k]] / sum;
                }
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

// Prevent unused import warning for Array1.
#[allow(dead_code)]
fn _touch(a: Array1<f64>) -> Array1<f64> {
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pairwise_euclidean_recovers_hand_distances() {
        let a = array![[0.0_f64, 0.0]];
        let b = array![[3.0, 4.0]];
        let d = pairwise_distances(a.view(), b.view(), PairwiseMetric::Euclidean).unwrap();
        assert!((d[[0, 0]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn rbf_kernel_returns_1_on_the_diagonal() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let k = rbf_kernel(a.view(), a.view(), 0.5).unwrap();
        assert!((k[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((k[[1, 1]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_gives_1_on_identical_vectors() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        let s = cosine_similarity(a.view(), a.view()).unwrap();
        assert!((s[[0, 0]] - 1.0).abs() < 1e-12);
    }
}
