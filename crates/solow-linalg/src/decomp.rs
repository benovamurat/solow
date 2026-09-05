//! Matrix decompositions, implemented in pure Rust.
//!
//! All routines operate on dense `f64` matrices ([`ndarray::Array2`]). The
//! algorithms are chosen for numerical robustness and verifiability rather than
//! peak performance: Cholesky and Householder/Jacobi methods that are stable and
//! easy to validate against an authoritative reference.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

/// Lower-triangular Cholesky factor `L` such that `A = L Lᵀ`.
///
/// `A` must be symmetric positive-definite. Returns [`Error::Singular`] otherwise.
pub fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::Shape("cholesky: matrix must be square".into()));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(Error::Singular(
                        "cholesky: matrix is not positive definite".into(),
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// An LU factorization with partial (row) pivoting: `P A = L U`.
#[derive(Clone, Debug)]
pub struct Lu {
    /// Combined `L` (unit lower, implicit ones on the diagonal) and `U` (upper).
    pub lu: Array2<f64>,
    /// Row permutation: `piv[i]` is the source row placed in position `i`.
    pub piv: Vec<usize>,
    /// `(-1)^(number of row swaps)`, used by [`crate::solve::det`].
    pub sign: f64,
}

/// Compute the LU factorization of a square matrix with partial pivoting.
pub fn lu_factor(a: &Array2<f64>) -> Result<Lu> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::Shape("lu: matrix must be square".into()));
    }
    let mut lu = a.clone();
    let mut piv: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;

    for k in 0..n {
        // Partial pivot: largest magnitude in column k at or below the diagonal.
        let mut p = k;
        let mut max = lu[[k, k]].abs();
        for i in (k + 1)..n {
            let v = lu[[i, k]].abs();
            if v > max {
                max = v;
                p = i;
            }
        }
        if max == 0.0 {
            return Err(Error::Singular("lu: matrix is singular".into()));
        }
        if p != k {
            for j in 0..n {
                lu.swap([k, j], [p, j]);
            }
            piv.swap(k, p);
            sign = -sign;
        }
        let pivot = lu[[k, k]];
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / pivot;
            lu[[i, k]] = factor;
            for j in (k + 1)..n {
                let v = lu[[k, j]];
                lu[[i, j]] -= factor * v;
            }
        }
    }
    Ok(Lu { lu, piv, sign })
}

/// Reduced (economy) Householder QR of a tall-or-square matrix `A` (`m ≥ n`):
/// returns `(Q, R)` with `Q` of shape `m × n` (orthonormal columns) and `R`
/// upper-triangular of shape `n × n`, such that `A = Q R`.
pub fn qr(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    if m < n {
        return Err(Error::Shape(
            "qr: this reduced QR requires rows >= cols".into(),
        ));
    }
    let mut r = a.clone();
    // Accumulate Q as the product of Householder reflectors applied to I (m × n).
    let mut q = Array2::<f64>::eye(m);

    for k in 0..n {
        // Householder vector for column k, rows k..m.
        let mut norm = 0.0;
        for i in k..m {
            norm += r[[i, k]] * r[[i, k]];
        }
        let norm = norm.sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if r[[k, k]] >= 0.0 { -norm } else { norm };
        // v = x - alpha e_k
        let mut v = vec![0.0; m];
        v[k] = r[[k, k]] - alpha;
        for (i, vi) in v.iter_mut().enumerate().skip(k + 1) {
            *vi = r[[i, k]];
        }
        let mut vnorm2 = 0.0;
        for vi in v.iter().skip(k) {
            vnorm2 += vi * vi;
        }
        if vnorm2 == 0.0 {
            continue;
        }
        // Apply H = I - 2 v vᵀ / (vᵀv) to R (columns k..n).
        for j in k..n {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i] * r[[i, j]];
            }
            let beta = 2.0 * dot / vnorm2;
            for i in k..m {
                r[[i, j]] -= beta * v[i];
            }
        }
        // Apply H to Q (all columns): Q := Q H.
        for j in 0..m {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i] * q[[j, i]];
            }
            let beta = 2.0 * dot / vnorm2;
            for i in k..m {
                q[[j, i]] -= beta * v[i];
            }
        }
    }

    // Economy slice: first n columns of Q, first n rows of R.
    let q_econ = q.slice(ndarray::s![.., 0..n]).to_owned();
    let r_econ = r.slice(ndarray::s![0..n, 0..n]).to_owned();
    Ok((q_econ, r_econ))
}

/// Singular value decomposition `A = U diag(s) Vᵀ` (economy / thin form).
///
/// For an `m × n` input, let `k = min(m, n)`. Returns `(U, s, Vt)` with `U` of
/// shape `m × k`, `s` of length `k` (descending, non-negative), and `Vt` of
/// shape `k × n`. Computed with a one-sided (Hestenes) Jacobi sweep, which has
/// excellent relative accuracy.
pub fn svd(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    if m >= n {
        Ok(svd_tall(a))
    } else {
        // svd(A) from svd(Aᵀ): if Aᵀ = U' S V'ᵀ then A = V' S U'ᵀ.
        let at = a.t().to_owned();
        let (up, s, vtp) = svd_tall(&at); // up: n×m, vtp: m×m
        let u = vtp.t().to_owned(); // m×m
        let vt = up.t().to_owned(); // m×n
        Ok((u, s, vt))
    }
}

/// One-sided Jacobi SVD for `m ≥ n`.
fn svd_tall(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let mut u = a.clone(); // becomes U·diag(s)
    let mut v = Array2::<f64>::eye(n);
    let eps = f64::EPSILON;
    let max_sweeps = 100;

    for _sweep in 0..max_sweeps {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0; // ‖col p‖²
                let mut beta = 0.0; // ‖col q‖²
                let mut gamma = 0.0; // col p · col q
                for i in 0..m {
                    let up = u[[i, p]];
                    let uq = u[[i, q]];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if gamma == 0.0 || gamma.abs() <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let up = u[[i, p]];
                    let uq = u[[i, q]];
                    u[[i, p]] = c * up - s * uq;
                    u[[i, q]] = s * up + c * uq;
                }
                for i in 0..n {
                    let vp = v[[i, p]];
                    let vq = v[[i, q]];
                    v[[i, p]] = c * vp - s * vq;
                    v[[i, q]] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }

    // Singular values are the column norms of U; normalize to get U's columns.
    let mut s = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut nrm = 0.0;
        for i in 0..m {
            nrm += u[[i, j]] * u[[i, j]];
        }
        s[j] = nrm.sqrt();
    }
    for j in 0..n {
        if s[j] > 0.0 {
            for i in 0..m {
                u[[i, j]] /= s[j];
            }
        }
    }

    // Sort by descending singular value.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| s[y].total_cmp(&s[x]));

    let s_sorted = Array1::from_iter(idx.iter().map(|&j| s[j]));
    let mut u_sorted = Array2::<f64>::zeros((m, n));
    let mut v_sorted = Array2::<f64>::zeros((n, n));
    for (newj, &oldj) in idx.iter().enumerate() {
        for i in 0..m {
            u_sorted[[i, newj]] = u[[i, oldj]];
        }
        for i in 0..n {
            v_sorted[[i, newj]] = v[[i, oldj]];
        }
    }
    let vt = v_sorted.t().to_owned();
    (u_sorted, s_sorted, vt)
}

/// Eigendecomposition of a real symmetric matrix via the cyclic Jacobi method.
///
/// Returns `(w, V)` where `w` holds the eigenvalues in **ascending** order
/// (matching the reference `eigh`) and the columns of `V` are the corresponding
/// orthonormal eigenvectors, so that `A = V diag(w) Vᵀ`.
pub fn eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::Shape("eigh: matrix must be square".into()));
    }
    let mut a = a.clone();
    let mut v = Array2::<f64>::eye(n);
    let max_sweeps = 100;

    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius norm (squared).
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[[p, q]] * a[[p, q]];
            }
        }
        if off <= f64::MIN_POSITIVE {
            break;
        }
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq == 0.0 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                // A := Jᵀ A J  (column rotation then row rotation).
                for k in 0..n {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - s * akq;
                    a[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[[p, k]];
                    let aqk = a[[q, k]];
                    a[[p, k]] = c * apk - s * aqk;
                    a[[q, k]] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors: V := V J.
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }

    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        w[i] = a[[i, i]];
    }
    // Ascending sort.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| w[x].total_cmp(&w[y]));
    let w_sorted = Array1::from_iter(idx.iter().map(|&i| w[i]));
    let mut v_sorted = Array2::<f64>::zeros((n, n));
    for (nj, &oj) in idx.iter().enumerate() {
        for i in 0..n {
            v_sorted[[i, nj]] = v[[i, oj]];
        }
    }
    Ok((w_sorted, v_sorted))
}
