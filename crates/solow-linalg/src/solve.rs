//! Linear solves, inverses, determinants, pseudoinverse, and least squares.

use crate::decomp::{lu_factor, svd, Lu};
use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

/// Solve `L y = Pb` then `U x = y` for a single right-hand side, given an LU
/// factorization.
fn lu_solve_vec(lu: &Lu, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let m = &lu.lu;
    let mut x = Array1::<f64>::zeros(n);
    // Apply the row permutation.
    for i in 0..n {
        x[i] = b[lu.piv[i]];
    }
    // Forward substitution (unit lower triangular).
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..i {
            sum -= m[[i, j]] * x[j];
        }
        x[i] = sum;
    }
    // Back substitution (upper triangular).
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= m[[i, j]] * x[j];
        }
        x[i] = sum / m[[i, i]];
    }
    x
}

/// Solve the square linear system `A x = b`.
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::Shape("solve: matrix must be square".into()));
    }
    if b.len() != n {
        return Err(Error::Shape("solve: rhs length mismatch".into()));
    }
    let lu = lu_factor(a)?;
    Ok(lu_solve_vec(&lu, b))
}

/// Solve `A X = B` for a matrix right-hand side.
pub fn solve_matrix(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::Shape("solve_matrix: matrix must be square".into()));
    }
    if b.nrows() != n {
        return Err(Error::Shape("solve_matrix: rhs row mismatch".into()));
    }
    let lu = lu_factor(a)?;
    let cols = b.ncols();
    let mut x = Array2::<f64>::zeros((n, cols));
    for c in 0..cols {
        let col = b.column(c).to_owned();
        let sol = lu_solve_vec(&lu, &col);
        for i in 0..n {
            x[[i, c]] = sol[i];
        }
    }
    Ok(x)
}

/// Inverse of a square matrix.
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::Shape("inv: matrix must be square".into()));
    }
    let lu = lu_factor(a)?;
    let mut inverse = Array2::<f64>::zeros((n, n));
    let mut e = Array1::<f64>::zeros(n);
    for c in 0..n {
        e.fill(0.0);
        e[c] = 1.0;
        let col = lu_solve_vec(&lu, &e);
        for i in 0..n {
            inverse[[i, c]] = col[i];
        }
    }
    Ok(inverse)
}

/// Determinant of a square matrix (product of LU pivots times the swap sign).
pub fn det(a: &Array2<f64>) -> Result<f64> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::Shape("det: matrix must be square".into()));
    }
    match lu_factor(a) {
        Ok(lu) => {
            let mut d = lu.sign;
            for i in 0..n {
                d *= lu.lu[[i, i]];
            }
            Ok(d)
        }
        // A singular matrix has determinant zero.
        Err(Error::Singular(_)) => Ok(0.0),
        Err(e) => Err(e),
    }
}

/// Moore–Penrose pseudoinverse via the SVD, together with the singular values.
///
/// Singular values at or below `rcond * max(s)` are treated as zero. The default
/// `rcond` of `1e-15` matches the reference's extended pseudoinverse.
pub fn pinv(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>)> {
    pinv_rcond(a, 1e-15)
}

/// [`pinv`] with an explicit relative cutoff.
pub fn pinv_rcond(a: &Array2<f64>, rcond: f64) -> Result<(Array2<f64>, Array1<f64>)> {
    let (m, n) = a.dim();
    let (u, s, vt) = svd(a)?;
    let k = s.len();
    let smax = s.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = rcond * smax;

    let mut sinv = Array1::<f64>::zeros(k);
    for i in 0..k {
        if s[i] > cutoff {
            sinv[i] = 1.0 / s[i];
        }
    }
    // pinv (n × m) = V diag(sinv) Uᵀ, with V = Vtᵀ.
    let mut p = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0;
            for r in 0..k {
                acc += vt[[r, i]] * sinv[r] * u[[j, r]];
            }
            p[[i, j]] = acc;
        }
    }
    Ok((p, s))
}

/// Numerical rank of a matrix using the reference convention
/// `tol = max(s) * max(m, n) * eps`.
pub fn matrix_rank(a: &Array2<f64>) -> Result<usize> {
    let (m, n) = a.dim();
    let (_, s, _) = svd(a)?;
    let smax = s.iter().cloned().fold(0.0_f64, f64::max);
    let tol = smax * (m.max(n) as f64) * f64::EPSILON;
    Ok(s.iter().filter(|&&x| x > tol).count())
}

/// Minimum-norm least-squares solution to `A x ≈ b` via the pseudoinverse.
pub fn lstsq(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let (m, _n) = a.dim();
    if b.len() != m {
        return Err(Error::Shape("lstsq: rhs length mismatch".into()));
    }
    let (p, _s) = pinv(a)?;
    Ok(p.dot(b))
}

/// Back-substitution solving the upper-triangular system `R x = b`.
fn solve_upper(r: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = b.clone();
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= r[[i, j]] * x[j];
        }
        x[i] = s / r[[i, i]];
    }
    x
}

/// Inverse of an upper-triangular matrix via column-wise back substitution.
fn inv_upper(r: &Array2<f64>) -> Array2<f64> {
    let n = r.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for c in 0..n {
        for i in (0..n).rev() {
            let mut s = if i == c { 1.0 } else { 0.0 };
            for j in (i + 1)..n {
                s -= r[[i, j]] * inv[[j, c]];
            }
            inv[[i, c]] = s / r[[i, i]];
        }
    }
    inv
}

/// Fast least-squares for a **tall, full-column-rank** design via Householder QR.
///
/// Returns `Some((x, (AᵀA)⁻¹))` — the least-squares solution and the normalized
/// covariance — or `None` if `A` is wide or not numerically full column rank, in
/// which case the caller should fall back to the SVD-based [`pinv`]. The reflectors
/// are applied directly to `b`; the orthogonal factor `Q` is never materialized, so
/// this is `O(m n²)` time and `O(m n)` memory (unlike forming a full `m × m` `Q`).
///
/// For a full-rank design this yields the same least-squares solution as the
/// pseudoinverse to within rounding, but far faster on tall matrices.
pub fn lstsq_qr(a: &Array2<f64>, b: &Array1<f64>) -> Result<Option<(Array1<f64>, Array2<f64>)>> {
    let (m, n) = a.dim();
    if m < n {
        return Ok(None);
    }
    if b.len() != m {
        return Err(Error::Shape("lstsq_qr: rhs length mismatch".into()));
    }

    // Column-major working buffer: column j occupies `buf[j*m .. j*m + m]`. Householder
    // reflectors touch sub-columns (contiguous slices), so the O(m n²) inner loops are
    // cache-friendly and auto-vectorize.
    let mut buf = vec![0.0_f64; m * n];
    for j in 0..n {
        let col = &mut buf[j * m..j * m + m];
        for (i, c) in col.iter_mut().enumerate() {
            *c = a[[i, j]];
        }
    }
    let mut qtb = b.to_vec();
    let mut v = vec![0.0_f64; m];

    for k in 0..n {
        let base_k = k * m;
        let mut norm = 0.0;
        for &x in &buf[base_k + k..base_k + m] {
            norm += x * x;
        }
        let norm = norm.sqrt();
        if norm == 0.0 {
            return Ok(None); // rank deficient
        }
        let akk = buf[base_k + k];
        let alpha = if akk >= 0.0 { -norm } else { norm };
        v[k] = akk - alpha;
        v[(k + 1)..m].copy_from_slice(&buf[base_k + k + 1..base_k + m]);
        let mut vnorm2 = 0.0;
        for &vi in &v[k..m] {
            vnorm2 += vi * vi;
        }
        if vnorm2 == 0.0 {
            continue;
        }
        let two_over = 2.0 / vnorm2;
        let vk = &v[k..m];
        for j in k..n {
            let col = &mut buf[j * m + k..j * m + m];
            let mut dot = 0.0;
            for (c, &x) in col.iter().zip(vk) {
                dot += c * x;
            }
            let beta = two_over * dot;
            for (c, &x) in col.iter_mut().zip(vk) {
                *c -= beta * x;
            }
        }
        let qb = &mut qtb[k..m];
        let mut dot = 0.0;
        for (q, &x) in qb.iter().zip(vk) {
            dot += q * x;
        }
        let beta = two_over * dot;
        for (q, &x) in qb.iter_mut().zip(vk) {
            *q -= beta * x;
        }
    }

    // Conditioning check on the R diagonal; bail to SVD if near-singular.
    let mut dmax = 0.0_f64;
    let mut dmin = f64::INFINITY;
    for i in 0..n {
        let d = buf[i * m + i].abs();
        dmax = dmax.max(d);
        dmin = dmin.min(d);
    }
    // Fall back to the SVD pseudoinverse for ill-conditioned or (near-)rank-deficient
    // designs — condition number ≳ 1e8. There the QR solution loses accuracy and the R
    // diagonal is an unreliable rank indicator, whereas the SVD reliably reveals the
    // rank and gives the correct minimum-norm least-squares solution, matching the
    // reference (which always uses the pseudoinverse). Well-conditioned designs — the
    // overwhelming majority — keep the fast QR path.
    if dmax == 0.0 || dmin <= dmax * 1e-8 {
        return Ok(None);
    }

    // Extract R (top n×n; R[i][j] = buf[j*m + i] for i ≤ j) and Qᵀb's first n entries.
    let mut rtop = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        for i in 0..=j {
            rtop[[i, j]] = buf[j * m + i];
        }
    }
    let qtb_top = Array1::from_iter(qtb[..n].iter().copied());

    let x = solve_upper(&rtop, &qtb_top);
    let rinv = inv_upper(&rtop);
    let ncp = rinv.dot(&rinv.t());
    Ok(Some((x, ncp)))
}

/// Solve a symmetric-positive-definite system `A x = b` via Cholesky (`A = L Lᵀ`).
pub fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let l = crate::decomp::cholesky(a)?;
    let n = b.len();
    // Forward solve L y = b.
    let mut y = b.clone();
    for i in 0..n {
        let mut s = y[i];
        for j in 0..i {
            s -= l[[i, j]] * y[j];
        }
        y[i] = s / l[[i, i]];
    }
    // Backward solve Lᵀ x = y.
    let mut x = y;
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= l[[j, i]] * x[j];
        }
        x[i] = s / l[[i, i]];
    }
    Ok(x)
}
